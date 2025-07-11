#import necessary packages
from PyPDF2 import PdfReader
import pandas as pd
import base64
from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

def get_text_from_pdf(pdf_path): #pdf_path contains list of pdf files
    text=""
    for pdf in pdf_path:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            text += content
    return text

#split text into chunks
def split_text_into_chunks(text,chunk_size=1000,chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

#embed text chunks
def embed_text_chunks():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.getenv("OPENAI_API_KEY"))
    return embeddings

#store text chunks in vector store
def store_text_in_vector_store(chunks, embeddings):
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_store")
    return vector_store

