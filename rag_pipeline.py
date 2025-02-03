import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
import fitz  # PyMuPDF for PDF processing

# Load environment variables (API keys)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  # Load your GROQ API key from .env

# Initialize the Groq client
groq_client = Groq(api_key=groq_api_key)

# Initialize Sentence-Transformer (MiniLM model)
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models as well

# Function to load documents (now includes PDF support)
def load_documents(file):
    if file.type == "application/pdf":
        # PDF loading
        text = ""
        pdf_document = fitz.open(stream=file.read())
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        return text.splitlines()
    else:
        # For text files, just read lines
        return file.getvalue().decode("utf-8").splitlines()

# Function to embed documents using Sentence Transformers (MiniLM)
def embed_documents(documents):
    embeddings = model.encode(documents)  # Use MiniLM to encode the documents
    return np.array(embeddings)

# Function to build and initialize the RAG pipeline
def build_rag_pipeline(documents):
    # Embed documents using Sentence Transformers
    embedded_docs = embed_documents(documents)

    # Initialize FAISS for efficient similarity search
    dim = embedded_docs.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dim)
    index.add(embedded_docs)

    # Function for querying RAG pipeline
    def rag_chain(query):
        # Embed the query using Sentence Transformers
        query_embedding = model.encode([query])

        # Retrieve similar documents using FAISS
        distances, indices = index.search(query_embedding, k=3)
        retrieved_docs = [documents[i] for i in indices[0]]

        # Combine the retrieved documents into context
        context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nAnswer the question: {query}"

        # Query the Groq API for chat completion (using the Llama model)
        response = query_with_llm(prompt)
        return response
    
    return rag_chain

# Function to query the LLM model via Groq API (for chat completion)
def query_with_llm(context):
    try:
        # Use the Groq API to send the context and get the response from the LLM
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": context}],
            model="llama3-8b-8192"  # Using Llama model for completions
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return "Sorry, I couldn't generate a response."
