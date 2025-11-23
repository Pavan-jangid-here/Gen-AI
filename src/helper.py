import os
import numpy as np
from PyPDF2 import PdfReader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# If using ConversationalRetrievalChain (legacy):
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv



VDB_DIR = "faiss_stores"

def create_faiss_store(VDB_DIR):
    os.makedirs(VDB_DIR, exist_ok=True)



# ========== Utility Functions ==========
def list_vector_stores():
    return [f.replace(".faiss", "") for f in os.listdir(VDB_DIR) if f.endswith(".faiss")]

def extract_pdf_text(files):
    texts = []
    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        texts.append(text)
    return texts


def split_text(pdf_texts):
    full_text = "\n".join(pdf_texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    return chunks

def Embedding_VectorStore(chunks):
    embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vdb = FAISS.from_texts(chunks, embedder)
    return vdb

def save_vector(vdb, vectorstore_name):
    VDB_DIR = "faiss_stores"
    os.makedirs(VDB_DIR, exist_ok=True)

    vdb_path = os.path.join(VDB_DIR, f"{vectorstore_name}.faiss")
    vdb.save_local(vdb_path)
    return

def load_vdb(vectorstore_path):
    embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vdb = FAISS.load_local(vectorstore_path, embedder, allow_dangerous_deserialization=True)
    return vdb

def get_context(vdb, user_input):
    docs = vdb.similarity_search(user_input)
    context = "\n".join([doc.page_content for doc in docs])
    return context


