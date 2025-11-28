import os
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
groq_api_key = os.getenv("GROQ_API_KEY", "")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# -------------------------------
# LLM + Prompt setup
# -------------------------------
if not groq_api_key:
    st.warning("GROQ_API_KEY is not set in .env. ChatGroq will fail without it.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# -------------------------------
# Helper: Create embeddings & vector store
# -------------------------------
def create_vector_embedding(documents):
    """
    Create FAISS vector store from documents and cache it in session_state.
    """
    if "vectors" not in st.session_state:
        with st.spinner("Processing documents... Please wait."):
            try:
                # Load embeddings model (HuggingFace, correct ID)
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                st.error(f"Failed to load embeddings model: {e}")
                return

            # Split documents
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            st.session_state.final_documents = (
                st.session_state.text_splitter.split_documents(documents)
            )

            # Build FAISS index
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings,
            )

        st.success("Documents processed and indexed successfully!")
    else:
        st.info("Documents are already indexed. Ready for querying!")

# -------------------------------
# Custom CSS Styles for improved UI
# -------------------------------
st.markdown(
    """
    <style>
    .main-header {text-align: center; font-size: 32px; color: #007BFF; margin-top: 10px;}
    .sub-header {text-align: center; font-size: 18px; color: #6c757d; margin-bottom: 20px;}
    .sidebar-header {color: #007BFF; font-size: 22px; font-weight: bold;}
    .search-btn {background-color: #007BFF; color: white; font-weight: bold; border-radius: 5px;}
    .response-time {color: #28a745; font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# App Title
# -------------------------------
st.markdown(
    "<div class='main-header'>HELPER GPT - Document Search Assistant</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-header'>Upload your PDF documents and ask questions. Get accurate answers in seconds.</div>",
    unsafe_allow_html=True,
)

# -------------------------------
# Sidebar: Upload Documents
# -------------------------------
st.sidebar.markdown(
    "<div class='sidebar-header'>Upload Documents</div>",
    unsafe_allow_html=True,
)
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

documents = []

if uploaded_files:
    total_files = len(uploaded_files)
    progress_bar = st.sidebar.progress(0.0)

    for idx, uploaded_file in enumerate(uploaded_files):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_path)
        documents.extend(loader.load())

        progress_bar.progress((idx + 1) / total_files)

    # Create vector embeddings after documents are uploaded
    create_vector_embedding(documents)
    st.sidebar.success(f"{total_files} document(s) uploaded and processed!")

    # Clean up temp files
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------
# User prompt input section
# -------------------------------
st.markdown("### Enter your query below")
user_prompt = st.text_input("Ask a question related to the uploaded documents:")

# -------------------------------
# Search button + logic
# -------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button(
        "Search",
        key="search",
        help="Click to search documents",
        use_container_width=True,
    ):
        if not user_prompt:
            st.warning("Please enter a query before searching.")
        else:
            if "vectors" not in st.session_state:
                st.error("Please upload and process documents first.")
            else:
                with st.spinner("Searching for the best response..."):
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    start = time.process_time()
                    try:
                        response = retrieval_chain.invoke({"input": user_prompt})
                        response_time = time.process_time() - start

                        st.write(
                            f"<div class='response-time'>Response time: {response_time:.2f} seconds</div>",
                            unsafe_allow_html=True,
                        )

                        # Display the result
                        st.markdown("### Answer")
                        st.write(response.get("answer", "No answer key in response."))

                        # Expander for document similarity search results
                        context_docs = response.get("context", [])
                        if context_docs:
                            with st.expander("See related documents"):
                                for i, doc in enumerate(context_docs):
                                    st.write(doc.page_content)
                                    st.write("------------------------")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Powered by LangChain & Streamlit</div>",
    unsafe_allow_html=True,
)
