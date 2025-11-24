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
import pandas as pd
from io import BytesIO
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

# def extract_and_chunk_pdfs(pdf_files):
#     chunks = []
#     for file in pdf_files:
#         reader = PdfReader(file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() or ""
#         # Split text into chunks for each PDF
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         file_chunks = splitter.split_text(text)
#         # Annotate each chunk with filename metadata
#         for chunk in file_chunks:
#             doc = Document(page_content=chunk, metadata={"source": file})
#             chunks.append(doc)
#     return chunks



def extract_and_chunk_files(file_objs):
    """
    file_objs: list of either string paths or Streamlit UploadedFile objects
    """
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file_obj in file_objs:
        # If it's an UploadedFile, get its attributes & content
        if hasattr(file_obj, "name") and hasattr(file_obj, "read"):
            file_name = file_obj.name
            file_ext = os.path.splitext(file_name)[1].lower()
        else:
            file_name = os.path.basename(file_obj)
            file_ext = os.path.splitext(file_obj)[1].lower()
        
        # ========== PDF ==========
        if file_ext == ".pdf":
            try:
                # Accept both file path and UploadedFile
                if hasattr(file_obj, "read"):
                    # Seek to beginning!
                    file_bytes = file_obj.read()
                    pdf_stream = BytesIO(file_bytes)
                    reader = PdfReader(pdf_stream)
                else:
                    reader = PdfReader(file_obj)
                
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                file_chunks = splitter.split_text(text)
                for chunk in file_chunks:
                    doc = Document(page_content=chunk, metadata={
                        "source": file_name, "file_type": "pdf"
                    })
                    chunks.append(doc)
            except Exception as e:
                print(f"PDF error: {file_name}: {e}")
        
        # ========== CSV/Excel ==========
        elif file_ext in [".csv", ".xlsx", ".xls"]:
            try:
                if hasattr(file_obj, "read"):
                    file_obj.seek(0)  # for safety
                    if file_ext == ".csv":
                        df = pd.read_csv(file_obj)
                    else:
                        df = pd.read_excel(file_obj)
                else:
                    if file_ext == ".csv":
                        df = pd.read_csv(file_obj)
                    else:
                        df = pd.read_excel(file_obj)
                for idx, row in df.iterrows():
                    text_parts = [f"{col}: {row[col]}" for col in df.columns]
                    content = "\n".join(map(str, text_parts))
                    doc = Document(page_content=content, metadata={
                        "source": file_name,
                        "file_type": file_ext.strip(".").lower(),
                        "row_index": int(idx)
                    })
                    chunks.append(doc)
            except Exception as e:
                print(f"DATA error: {file_name}: {e}")
        else:
            print(f"Unsupported file type: {file_name}")
    
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
