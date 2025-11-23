from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_vector_db(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db_location = "./chrome_langchain_db"
    # This ensures persistence across sessions
    add_documents = not Chroma(collection_name="Boiler_tube_leakage_docs", persist_directory=db_location).get(ids=["0"]) # Attempt to read something, if fail, add

    documents = []
    ids = []

    # Chunks become Documents for Chroma
    for i, chunk in enumerate(text_chunks):
        document = Document(page_content=chunk, id=str(i))
        documents.append(document)
        ids.append(str(i))

    # Always create the vector DB
    vector_store = Chroma(
        collection_name="Boiler_tube_leakage_docs",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    # Add only if DB is empty/new
    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

    # retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return vector_store

