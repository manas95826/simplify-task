from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# FastAPI setup
app = FastAPI()

# Initialize components once for efficiency
loader = WebBaseLoader(
    web_paths=("https://supportyourapp.com/",),
)
docs = loader.load()

# Loading the mixtral model from groq client
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", api_key='gsk_ntOqjghFfAEabmSRltBrW')

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Set up the retriever and chain
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Pydantic model for request body
class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: QueryRequest):
    # Invoke the chain and return the response
    response = rag_chain.invoke(request.question)
    return {"response": response}

# To run the server, use uvicorn:
# uvicorn app_name:app --reload
