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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")




VDB_DIR = "faiss_stores"

def create_faiss_store(VDB_DIR):
    os.makedirs(VDB_DIR, exist_ok=True)



# ========== Utility Functions ==========
def list_vector_stores():
    return [f.replace(".faiss", "") for f in os.listdir(VDB_DIR) if f.endswith(".faiss")]

# def extract_pdf_text(files):
#     texts = []
#     for file in files:
#         reader = PdfReader(file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() or ""
#         texts.append(text)
#     return texts


# def split_text(pdf_texts):
#     full_text = "\n".join(pdf_texts)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_text(full_text)
#     return chunks

def extract_and_chunk_pdfs(pdf_files):
    chunks = []
    for file in pdf_files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        # Split text into chunks for each PDF
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        file_chunks = splitter.split_text(text)
        # Annotate each chunk with filename metadata
        for chunk in file_chunks:
            doc = Document(page_content=chunk, metadata={"source": file})
            chunks.append(doc)
    return chunks

# def Embedding_VectorStore(chunks):
#     embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
#     vdb = FAISS.from_texts(chunks, embedder)
#     return vdb



# def Embedding_VectorStore(chunks):
#     # Use a LangChain-compatible embeddings wrapper
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     # Pass text chunks and embeddings object to FAISS.from_texts
#     vdb = FAISS.from_texts(chunks, embedding=embeddings)
#     return vdb

def Embedding_VectorStore(documents):
    # Use LangChain-compatible HF embeddings wrapper
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Use FAISS.from_documents to preserve metadata
    vdb = FAISS.from_documents(documents, embedding=embeddings)
    return vdb



def save_vector(vdb, vectorstore_name):
    VDB_DIR = "faiss_stores"
    os.makedirs(VDB_DIR, exist_ok=True)

    vdb_path = os.path.join(VDB_DIR, f"{vectorstore_name}.faiss")
    vdb.save_local(vdb_path)
    return

# def load_vdb(vectorstore_path):
#     embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
#     vdb = FAISS.load_local(vectorstore_path, embedder, allow_dangerous_deserialization=True)
#     return vdb

def load_vdb(vectorstore_path):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vdb = FAISS.load_local(vectorstore_path, embedder, allow_dangerous_deserialization=True)
    return vdb


# def get_context(vdb, user_input):
#     docs = vdb.similarity_search(user_input)
#     context = "\n".join([doc.page_content for doc in docs])
#     return context

def get_context(vdb, user_input):
    docs = vdb.similarity_search(user_input)
    context_output = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        context_output.append(f"PDF: {source}\n{doc.page_content}")
    context = "\n\n".join(context_output)
    return context
