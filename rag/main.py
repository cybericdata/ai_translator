from fastapi import FastAPI, File, UploadFile, HTTPException
from model_schema import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from agent import get_hiv_rag_agent
from db import insert_application_logs, get_chat_history
import os
import uuid
import logging
import shutil

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    print(query_input)
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    print(query_input)
    chat_history = get_chat_history(session_id)
    result = get_hiv_rag_agent(query_input.question)
    # answer = rag_chain.invoke({
    #     "input": query_input.question,
    #     "chat_history": chat_history
    # })['answer']

    insert_application_logs(session_id, query_input.question, result['answer'], query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {result}")
    return QueryResponse(answer=result['answer'], session_id=session_id, model=query_input.model)

