import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load your environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Gemini model for chat & Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)
embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

st.title("Universe Chatbot (Gemini powered)")

# Session management
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# File upload
uploaded_files = st.file_uploader("Attach PDF files", type=["pdf"], accept_multiple_files=True)

# Persistent FAISS vector store file
VECTORSTORE_PATH = "faiss_store"

# Utility: Extract text from multiple PDFs
def extract_pdf_text(files):
    texts = []
    for file in files:
        reader = PdfReader(file)
        # Concatenate all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        texts.append(text)
    return texts

# Process the files and load into persistent FAISS vector store
if uploaded_files:
    # 1. PDF to text
    pdf_texts = extract_pdf_text(uploaded_files)
    full_text = "\n".join(pdf_texts)
    # 2. Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    # 3. Embedding + Vector DB
    if os.path.exists(VECTORSTORE_PATH):
        vdb = FAISS.load_local(VECTORSTORE_PATH, embedder,
                               allow_dangerous_deserialization=True)
    else:
        vdb = FAISS.from_texts(chunks, embedder)
        vdb.save_local(VECTORSTORE_PATH)
    st.success("Uploaded PDFs processed and stored in vector DB.")

# User input
user_input = st.text_input("You:", "")

if st.button("Send"):
    st.session_state["chat_messages"].append({"role": "user", "content": user_input})

    # If we have a vector DB, do similarity search
    answer = ""
    if os.path.exists(VECTORSTORE_PATH):
        vdb = FAISS.load_local(VECTORSTORE_PATH, embedder,
                               allow_dangerous_deserialization=True)
        docs = vdb.similarity_search(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        # Send context + query to Gemini
        full_query = f"Context:\n{context}\n\nQuestion:{user_input}"
        response = llm.invoke(full_query)
        answer = response.content if hasattr(response, 'content') else str(response)
    else:
        response = llm.invoke(user_input)
        answer = response.content if hasattr(response, 'content') else str(response)

    st.session_state["chat_messages"].append({"role": "bot", "content": answer})

# Show chat history
for msg in st.session_state["chat_messages"]:
    role = "You" if msg["role"] == "user" else "Universe"
    st.markdown(f"**{role}:** {msg['content']}")

