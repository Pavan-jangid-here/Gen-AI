import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import math

# Load your environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)
embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

st.title("Universe Chatbot (Gemini powered)")

# Session management
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "current_vectorstore" not in st.session_state:
    st.session_state["current_vectorstore"] = None

# Directory to store FAISS DBs
VDB_DIR = "faiss_stores"
os.makedirs(VDB_DIR, exist_ok=True)

# Utility: List vector stores (main + batches)
def list_vector_stores():
    return [f.replace(".faiss", "") for f in os.listdir(VDB_DIR) if f.endswith(".faiss")]

# Utility: Extract text from multiple PDFs
def extract_pdf_text(files):
    texts = []
    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        texts.append(text)
    return texts

# Batch creation parameters
MAX_CHUNKS_PER_RUN = st.number_input(
    "Max chunks per batch (lower if you hit quota errors, usually 50~100)",
    min_value=1, max_value=500, value=50
)

uploaded_files = st.file_uploader("Attach PDF files", type=["pdf"], accept_multiple_files=True)
vectorstore_name = st.text_input("Name your new vector store", "")
create_vector_btn = st.button("Create Batched Vector Store")

def create_batched_vector_stores(chunks, embedder, base_name, vdb_dir, max_per_run):
    total_batches = math.ceil(len(chunks) / max_per_run)
    last_successful_batch = None
    for batch_idx in range(total_batches):
        start = batch_idx * max_per_run
        end = min((batch_idx + 1) * max_per_run, len(chunks))
        batch_chunks = chunks[start:end]
        vdb_name = f"{base_name}_{batch_idx+1}"
        vdb_path = os.path.join(vdb_dir, f"{vdb_name}.faiss")
        try:
            vdb = FAISS.from_texts(batch_chunks, embedder)
            vdb.save_local(vdb_path)
            st.success(f"Batch {batch_idx+1}/{total_batches} saved as '{vdb_name}'")
            last_successful_batch = batch_idx+1
        except Exception as e:
            st.warning(
                f"Error processing batch {batch_idx+1}: {e}\n"
                "You may be hitting a rate/quota error. Please wait and retry remaining batches later."
            )
            break
    return last_successful_batch

if create_vector_btn and uploaded_files and vectorstore_name:
    pdf_texts = extract_pdf_text(uploaded_files)
    full_text = "\n".join(pdf_texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    # Batched creation!
    create_batched_vector_stores(chunks, embedder, vectorstore_name, VDB_DIR, MAX_CHUNKS_PER_RUN)
    st.session_state["current_vectorstore"] = f"{vectorstore_name}_1"  # Set to first batch by default

# Merging utility
def merge_faiss_files(base_name, vdb_dir):
    batch_files = sorted([
        f for f in os.listdir(vdb_dir)
        if f.startswith(base_name + "_") and f.endswith(".faiss")
    ])
    final_store_path = os.path.join(vdb_dir, f"{base_name}.faiss")
    if not batch_files:
        st.warning("No batch files found to merge.")
        return
    base_vdb = FAISS.load_local(os.path.join(vdb_dir, batch_files[0]), embedder,
                                allow_dangerous_deserialization=True)
    for batch_file in batch_files[1:]:
        new_vdb = FAISS.load_local(os.path.join(vdb_dir, batch_file), embedder,
                                   allow_dangerous_deserialization=True)
        base_vdb.merge_from(new_vdb)
    base_vdb.save_local(final_store_path)
    st.success(f"Merged all batches into '{base_name}'.")

# UI for merging
st.markdown("### Merge Batched Vector Stores")
merge_base_name = st.text_input("Base store name to merge batches (e.g. Boiler Research Paper)")
merge_btn = st.button("Merge All Batches")
if merge_btn and merge_base_name:
    merge_faiss_files(merge_base_name, VDB_DIR)

# List available vector stores (main + batches)
st.markdown("### Available Vector Stores")
vector_store_list = list_vector_stores()
selected_vectorstore = st.selectbox("Choose a vector store", vector_store_list)
if selected_vectorstore:
    st.session_state["current_vectorstore"] = selected_vectorstore

vectorstore_path = None
if st.session_state["current_vectorstore"]:
    vectorstore_path = os.path.join(VDB_DIR, f"{st.session_state['current_vectorstore']}.faiss")

# User input
user_input = st.text_input("You:", "")
if st.button("Send"):
    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    answer = ""
    if vectorstore_path and os.path.exists(vectorstore_path):
        vdb = FAISS.load_local(vectorstore_path, embedder,
                              allow_dangerous_deserialization=True)
        docs = vdb.similarity_search(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        full_query = f"You are an Expert in field of Boiler of thermal power plant. Here is Context:\n{context}\n\nQuestion:{user_input}"
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
