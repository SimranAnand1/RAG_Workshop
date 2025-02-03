import streamlit as st
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader  # Updated for PDF handling
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

import os
from dotenv import load_dotenv

load_dotenv()

def load_pdfs_from_folder(folder_path):
    """
    Loads all PDF files from a given folder.
    
    Args:
        folder_path (str): Path to the folder containing PDFs.

    Returns:
        list: A list of documents loaded from all PDFs.
    """
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):  # Filter for PDF files
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())  # Add all documents from the PDF
    return documents

def main():
    st.title("RAG Agent with Browser Search and Document Retrieval")
    st.write("Interact with the Agentic AI RAG chatbot powered LangChain with added browser search and document retrieval features.")

    # Sidebar for customization
    st.sidebar.title('Settings')
    system_prompt = st.sidebar.text_input("System prompt:", value="You are a helpful assistant.")
    model = st.sidebar.selectbox('Choose a model', ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma2-9b-it', 'deepseek-r1-distill-llama-70b'])
    memory_length = st.sidebar.slider('Memory length:', 1, 10, value=5)

    # Initialize conversation memory
    memory = ConversationBufferWindowMemory(k=memory_length, memory_key="chat_history", return_messages=True)

    # User input
    user_input = st.text_input("Your message:")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load previous chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Initialize Groq chat model
    groq_chat = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model_name=model)

    # Initialize embeddings and document retrieval
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load multiple PDFs from a folder
    folder_path = "./documents"  # Update this to the path where your PDFs are stored
    docs = load_pdfs_from_folder(folder_path)
    
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=groq_chat,
        retriever=retriever,
        return_source_documents=True
    )

    # Initialize browser search tool
    search_tool = DuckDuckGoSearchRun()

    if user_input:
        # Define the chat prompt template
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        # Create a conversation chain
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory
        )

        # Handle input types
        if "search:" in user_input.lower():
            query = user_input.split("search:", 1)[1].strip()
            search_results = search_tool.run(query)
            st.write("Search Results:", search_results)
        elif "retrieve:" in user_input.lower():
            query = user_input.split("retrieve:", 1)[1].strip()
            results = retrieval_qa_chain.invoke({"query": query})
            st.write("Retrieved Answer:", results['result'])
            st.write("Source Documents:", [doc.metadata["source"] for doc in results['source_documents']])
        else:
            # Regular conversation
            response = conversation.predict(human_input=user_input)
            st.session_state.chat_history.append({'human': user_input, 'AI': response})
            st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
