# RAG Chatbot with Streamlit and Google Generative AI

This is an interactive retrieval-augmented generation (RAG) chatbot application built with Streamlit. It leverages OpenAI-like embeddings from Google Generative AI via Langchain to create vector stores from PDF documents and perform context-aware question answering for expert domains such as thermal power plant boilers.

## Features

- Upload multiple PDF files and create named vector stores for document embeddings.
- Perform semantic search on vector stores to retrieve relevant context.
- Ask questions to the chatbot, which answers using Google Generative AI with retrieved contextual information.
- Maintains interactive chat history with styled UI in Streamlit.
- Uses FAISS for efficient similarity search on vector embeddings.

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```


3. **Set up environment variables**:

Create a `.env` file in the root directory with your Google API key:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

4. **Run the app**:

```bash
streamlit run app.py
```

## Usage

- Use the "Create Vector Store" section to upload PDF files and name the vector store.
- The app extracts text from PDFs, splits it into chunks, and creates a FAISS vector store.
- Once created, select a vector store in the dropdown.
- Enter your question in the "Ask a Question" section and submit.
- The chatbot returns answers based on the context retrieved from the selected vector store.
- Chat history is displayed below the input with user and bot messages styled as chat bubbles.

## Project Structure

- `app.py`: Main Streamlit app orchestrating UI, session state, and chatbot logic.
- `src/helper.py`: Helper functions for PDF text extraction, text splitting, embedding creation, vector store persistence, and semantic search.
- `faiss_stores/`: Directory where FAISS vector store files are saved.
- `.env`: Environment variables containing the Google API key.

## Dependencies

- streamlit
- PyPDF2
- langchain (with google gen ai extensions)
- sentence-transformers
- faiss
- python-dotenv

## Notes

- Google Generative AI is used for both embedding creation and chat completion.
- Make sure your Google API credentials have appropriate permissions.
- The app includes custom CSS for an enhanced UI experience.

---

