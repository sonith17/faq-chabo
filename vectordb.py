from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import argparse
from dotenv import load_dotenv

def add_documents_to_faiss(vectorstore, documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    for text in texts:
        vectorstore.add_texts([text.page_content], [text.metadata])
    return vectorstore

def create_faiss_index(documentation_path):
    loader = PyPDFLoader(documentation_path)
    page = loader.load()
    spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    documents = spliter.split_documents(page)
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable is not set.")
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key,model="mistral-embed")
    vectorstore = FAISS.from_documents(documents,embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a FAISS index from a PDF file.")
    parser.add_argument("--doc_path", type=str, help="Path to the PDF documentation file.")
    documentation_path = parser.parse_args().doc_path
    vectorstore = create_faiss_index(documentation_path)
    print("FAISS index created and saved locally.")