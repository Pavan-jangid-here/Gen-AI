import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_relevant_context_and_answer
from src.vector import get_vector_db

def main():
    st.set_page_config("Information Retrieval")
    st.header("Boiler-Tube-Leakage-System 💡")

    # Session state for vector DB and file status
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "files_ready" not in st.session_state:
        st.session_state.files_ready = False

    # Sidebar: Upload and process files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Submit with button.",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing ..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_db(text_chunks)
                    st.session_state.vector_store = vector_store
                    st.session_state.files_ready = True
                    st.success("Files processed and context ready!")
            else:
                st.warning("Please upload at least one PDF.")

    # Main Q&A area
    user_question = st.text_input("Ask a Question !!!")
    if st.button("Ask"):
        if st.session_state.files_ready and st.session_state.vector_store:
            answer, context_chunks = get_relevant_context_and_answer(
                st.session_state.vector_store, user_question
            )
            st.write("## Answer:")
            st.write(answer)
            st.write("### Relevant Context Chunks:")
            for idx, chunk in enumerate(context_chunks, 1):
                st.write(f"{idx}. {chunk}")
        else:
            st.warning("Upload and process files first (in sidebar)!")

if __name__ == "__main__":
    main()
