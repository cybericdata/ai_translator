import os
import re
from dotenv import load_dotenv

import bs4
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, MessagesState
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USER_AGENT = os.getenv("USER_AGENT")
WEB_URL1 = os.getenv("WEB_URL1")
WEB_URL2 = os.getenv("WEB_URL2")
WEB_URL3 = os.getenv("WEB_URL3")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment.")

if not USER_AGENT:
  raise ValueError("USER_AGENT is not set. Please set it in your environment variables.")

llm = init_chat_model("llama3-8b-8192", model_provider="groq")


loader = WebBaseLoader(
    web_paths=(WEB_URL1, WEB_URL2, WEB_URL3)
)

docs = loader.load()

assert len(docs) > 0

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

for doc in docs:
    if doc.page_content:
        doc.page_content = clean_text(doc.page_content)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

vector_store = InMemoryVectorStore(embeddings)

vector_store.add_documents(documents=all_splits)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use five sentences maximum and keep the answer as concise as possible, for questions with a list make it a bullet point.
    Always say "Thanks!, Is there anything else about HIV/AIDS you will like to know?" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)
    formatted_prompt = prompt_template.format(context=docs_content, question=state["question"])
    response = llm.invoke(formatted_prompt)
    return {"answer": response.content}



# print(f'Context: {result["context"]}\n\n')
# print(f'Answer: {result["answer"]}')

def get_hiv_rag_agent(query: str) -> str:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = graph.invoke({"question": query})
    return result

# if __name__ == "__main__":
#     query = {"question": "What is HIV and AIDS?"}
#     result = get_hiv_rag_agent(query)
#     print(f"Answer: {result['answer']}")