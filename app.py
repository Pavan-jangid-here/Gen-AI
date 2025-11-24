import streamlit as st
from src.helper import list_vector_stores, Embedding_VectorStore, save_vector, VDB_DIR, create_faiss_store, load_vdb, get_context, extract_and_chunk_pdfs
# from src.vector import get_vector_db
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def main():
    # ========== Custom CSS for UI & Layout ==========
    st.markdown("""
    <style>
    html, body, [class*="css"]  {font-family: 'Montserrat', sans-serif;}
    .stApp {
        background: linear-gradient(120deg, #212E39 0%, #7289da 100%);
    }
                
    /* Set overall page width */
    .main, .block-container {
        max-width: 1500px !important;  /* You can adjust this (e.g., 1200px, 700px) */
        margin-left: auto !important;
        margin-right: auto !important;
    }
                
    .main-title {
        font-size: 3rem !important;
        color: #7289da;
        text-align: center;
        margin-top: 30px;
    }
    .create-vector-title, .ask-question-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #7289da;
        margin: 0.35em 0;
    }
    .chat-bubble-user {
        background: #7289da;
        color: white;
        border-radius: 1em 1em 0 1em;
        padding: 1.3em;
        margin: 0.4em 0;
        width: fit-content;
        max-width: 60%;
        align-self: flex-end;
        box-shadow: 0 2px 12px rgba(50,50,100,0.1);
        animation: fadeInUp 0.6s;
    }
    .chat-bubble-bot {
        background: #fff;
        color: #212E39;
        border-radius: 1em 1em 1em 0;
        padding: 1.3em;
        margin: 0.4em 0;
        width: fit-content;
        max-width: 60%;
        align-self: flex-start;
        box-shadow: 0 2px 12px rgba(50,50,100,0.18);
        border-left: 6px solid #7289da;
        animation: fadeInUp 0.6s;
    }
    @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px);}
    to { opacity: 1; transform: translateY(0);}
    }
    </style>
    """, unsafe_allow_html=True)


    # ======================================================== Title =========================================================================
    st.markdown(f"<div class='main-title'>RAG Chatbot</div>", unsafe_allow_html=True)

    # ============================================================ Session State =============================================================
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "current_vectorstore" not in st.session_state:
        st.session_state["current_vectorstore"] = None


    # ==================================================== CREATE VECTOR SECTION =============================================================

    create_faiss_store(VDB_DIR)

    with st.expander("🗂️ Create Vector Store"):
    # st.markdown('<div class="create-vector-title">Create Vector</div>', unsafe_allow_html=True)
        
        with st.container():
            col1, colA, col2 = st.columns([3,2,1])
            with col1:
                uploaded_files = st.file_uploader(label = "", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
            with colA:
                vectorstore_name = st.text_input("", placeholder="Name of Vector Store", key="vector_name", label_visibility="collapsed")
            with col2:
                create_vector_btn = st.button("Create", key="create_vector_small")


    st.markdown("")
    st.markdown("")

    # ==================================================== END OF CREATE VECTOR SECTION =============================================================


    # ==================================================== Processing of PDF to Vector Store ========================================================

    if create_vector_btn and uploaded_files and vectorstore_name:
        # pdf_texts = extract_pdf_text(uploaded_files)
        # chunks = split_text(pdf_texts)
        documents = extract_and_chunk_pdfs(uploaded_files)

        try:
            vdb = Embedding_VectorStore(documents)
            save_vector(vdb, vectorstore_name)
            st.success(f"✅ Vector store **{vectorstore_name}** created successfully.")
            st.session_state["current_vectorstore"] = vectorstore_name

        except Exception as e:
            st.error(f"❌ Error creating vector store: {e}")

    vector_store_list = list_vector_stores()

    # ============================================= End of Processing of PDF to Vector Store ========================================================

    # =========================================================== ASK A QUESTION SECTION ============================================================

    st.markdown('<div class="ask-question-title">Ask a Question</div>', unsafe_allow_html=True)
    with st.container():
        colq, cold, colb = st.columns([3,1,1])
        with colq:
            user_input = st.text_input("", placeholder="Your query...", key="user_query", label_visibility="collapsed")
        with cold:
            selected_vectorstore = st.selectbox("", vector_store_list, key="vs_dropdown", label_visibility="collapsed")
        with colb:
            send_btn = st.button("➡️", key="send_right_arrow")

    # Update active vectorstore if changed
    if selected_vectorstore:
        st.session_state["current_vectorstore"] = selected_vectorstore

    # ====================================================== End of ASK A QUESTION SECTION ============================================================

    vectorstore_path = None
    if st.session_state["current_vectorstore"]:
        vectorstore_path = os.path.join(VDB_DIR, f"{st.session_state['current_vectorstore']}.faiss")

    if send_btn and user_input.strip():
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        answer = ""
        try:
            if vectorstore_path and os.path.exists(vectorstore_path):
                vdb = load_vdb(vectorstore_path)
                context = get_context(vdb, user_input)
                full_query = f"You are an Expert in field of Boiler of thermal power plant. Here is Context:\n{context}\n\nQuestion:{user_input}, give the PDF source at the end of given response."

                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)
                response = llm.invoke(full_query)
                answer = response.content if hasattr(response, 'content') else str(response)
            else:
                response = llm.invoke(user_input)
                answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            answer = f"❌ Error: {e}"
        st.session_state["chat_messages"].append({"role": "bot", "content": answer})

    st.markdown("")
    st.markdown("")

    # ========== Chat History (Horizontal Bubbles) ==========
    if st.session_state["chat_messages"]:
        st.markdown("### 💎 Chat Universe")
        # Render chat as columns, latest first
        for i, msg in enumerate(reversed(st.session_state["chat_messages"])):  # latest-on-top
            colu, colb = st.columns([2,2])
            with colu if msg["role"] == "user" else colb:
                roleicon = "🧑‍💼" if msg["role"] == "user" else "🌌"
                cssclass = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
                st.markdown(f"<div class='{cssclass}'>{roleicon} {msg['content']}</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
