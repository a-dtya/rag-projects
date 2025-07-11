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
from langchain_core.output_parsers.pydantic import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List

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


class MCQ(BaseModel): #example MCQ: What is the capital of France? Options: ["Paris", "London", "Berlin", "Madrid"]
    question: str = Field(...,description="The MCQ question")
    options: List[str] = Field(...,min_items=3,max_items=3,description="The options for the MCQ")
    answer: str = Field(...,description="The answer to the MCQ (it should be one of the options)")


class MCQList(BaseModel): #example MCQList: [MCQ(question="What is the capital of France?", options=["Paris", "London", "Berlin"], answer="Paris"), MCQ(question="What is the capital of Germany?", options=["Paris", "London", "Berlin"], answer="Berlin")] 
    mcqs: List[MCQ] = Field(...,description="The list of MCQs")

def mcqlist_parser(): #to enforce strict JSON formatting for output
    parser = PydanticOutputParser(return_id="id",pydantic_object=MCQList)
    return parser
    
def get_prompt(): #prompt for MCQ generation
    parser = mcqlist_parser()
    prompt = PromptTemplate(
    template="""
        You are an MCQ generator for campaign training.

        Generate 5 multiple-choice questions from the following campaign material. Each question must have exactly 3 options and one correct answer.

        {format_instructions}

        Campaign context:
        {context}
    """,
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt
