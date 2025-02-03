import streamlit as st
from rag_pipeline import load_documents, build_rag_pipeline

# Streamlit UI setup
st.title("RAG Chatbot - Document Query System")

st.write("""
    This is a Retrieval-Augmented Generation (RAG) chatbot. 
    It answers questions based on the documents provided. 
    You can upload a PDF or text file, and the system will allow you to ask questions based on it.
""")

# File upload (supports PDF or text)
uploaded_file = st.file_uploader("Choose a PDF or text file to upload", type=["pdf", "txt"])

if uploaded_file is not None:
    # Load the document
    document_content = load_documents(uploaded_file)

    # Initialize RAG pipeline
    rag_pipeline = build_rag_pipeline(document_content)

    # User query input
    query = st.text_input("Ask a question about the document:")

    if query:
        # Get the response from the RAG pipeline
        response = rag_pipeline(query)
        st.write("Answer: ", response)
