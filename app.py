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
from langchain.chains.llm import LLMChain
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
def load_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.getenv("OPENAI_API_KEY"))
    return embeddings

#store text chunks in vector store
def store_text_in_vector_store(chunks, embeddings):
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_store")
    return vector_store

def load_knowledge_base():
    embeddings = load_embeddings()
    vector_store = FAISS.load_local("faiss_store", embeddings)
    return vector_store

def get_relevant_context(query):
    knowledge_base = load_knowledge_base()
    relevant_context = knowledge_base.similarity_search(query) #default k=4
    return "\n\n".join([doc.page_content for doc in relevant_context])


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

def load_model():
    model = ChatOpenAI(model_name="gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY"))
    model_with_structure = model.with_structured_output(MCQList)
    return model_with_structure
    
def generate_mcqs(context):
    model = load_model()
    prompt = get_prompt()
    chain = LLMChain(llm=model,prompt=prompt)
    response = chain.invoke({"context": context})
    return response
    
def load_mcq_generator(pdf_path):
    campaign_material = get_text_from_pdf(pdf_path)
    kb_store = load_knowledge_base()
    kb_context = get_relevant_context(campaign_material)
    combined_context = campaign_material + "\nRelated information:\n" + kb_context
    return generate_mcqs(combined_context)
    