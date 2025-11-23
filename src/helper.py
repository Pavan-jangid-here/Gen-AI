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


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store_new(text_chunks):
    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    embeddings = model.encode(text_chunks)

    # Save embeddings and texts locally as .npy and .txt files
    np.save('embeddings.npy', embeddings)

    with open('text_chunks.txt', 'w', encoding='utf-8') as f:
        for chunk in text_chunks:
            f.write(chunk.replace('\n', ' ') + '\n')


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    except Exception as e:
        print(f"Error Occured: {e}")
    return vector_store


# def get_conversational_chain(vector_store):
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key = GOOGLE_API_KEY)

#     memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)

#     conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vector_store.as_retriever(), memory = memory) 

#     return conversation_chain


def get_relevant_context_and_answer(vector_store, user_question, k=5):
    """Retrieve relevant chunks and get answer using LLM."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(user_question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # Build prompt for answer generation
    template = """
        You are an expert in answering questions about boiler tube leakage.
        Here is some context from technical documents: {context}
        Here is the question: {question}
        Answer precisely based only on the context.
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)
    answer = llm.invoke(prompt.format(context=context, question=user_question))
    # Streamlit expects string for display; context_chunks for inspection
    return answer.content if hasattr(answer, "content") else answer, [doc.page_content for doc in relevant_docs]

