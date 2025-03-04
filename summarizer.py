from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import chain
from typing import List


from utility import utils

folder_path  = './data'
documents = utils.load_documents(folder_path)

print(f"loaded {len(documents)}")
# loader =  PyPDFLoader(file_path)

# docs = loader.load()

# print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])
# using chroma
vector_chroma_store = Chroma(embedding_function=embeddings)

# print(embeddings)

# vector_store = InMemoryVectorStore(embeddings)

# ids = vector_store.add_documents(documents=all_splits)
ids = vector_chroma_store.add_documents(documents=all_splits)

# print(ids)

embedding = embeddings.embed_query("What is Tuberculosis (TB) as an infectious disease?")

results = vector_chroma_store.similarity_search_by_vector(embedding)

@chain
def retriever(query: str) -> List[Document]:
    return vector_chroma_store.similarity_search(query, k=1)

rst = retriever.batch([
    "What is Tuberculosis (TB) as an infectious disease?",
    "What is the symptoms of Tuberculosis?",
    "What are the treatment options for Tuberculosis?",
    "Who is responsible for Tuberculosis?"
    # Add more queries as needed!
])

print(f"result from query: {rst}")

# for i, result in enumerate(results, 1):
#     print(f"Result {i}:")
#     print(f"Source: {result.metadata.get('source', 'Unknown')}")
#     print(f"Content: {result.page_content}")
#     print()

