import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment.")

model = init_chat_model("llama3-8b-8192", model_provider="groq")

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template), ("user","{text}")])

prompt = prompt_template.invoke({"language": "French", "text": "How old are you?"})
#testing out the model
# messages = [
#     SystemMessage("Translate the following from English into Hausa"),
#     HumanMessage("The sky is blue"), 
# ]
prompt = prompt.to_messages()

message = model.invoke(prompt)
print(message.content)

