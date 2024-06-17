import streamlit as st
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer


# Load embedding model and cache it
@st.cache_resource
def load_embedding_model():
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return emb_model



def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    """
    Chunk large documents into smaller segments.

    Parameters:
    - docs (str or list): Input documents to be chunked. If a single string, it will be treated as one document.
                          If a list of strings, each element will be treated as a separate document.
    - chunk_size (int): Size of each chunk in characters (default is 800).
    - chunk_overlap (int): Number of characters to overlap between adjacent chunks (default is 50).

    Returns:
    - list: List of chunked documents, where each element represents a chunk.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs

def Doc_QA():
    st.title("Doc Quest")
    pdf_file = st.file_uploader("Upload the query document", type="pdf")
    emb_model = load_embedding_model()

    text = st.text_input("Input something")
    if text:
        embeddings = emb_model.encode([text])
        st.write(f"Embedding shape for '{text}': {embeddings.shape}")
    else:
        st.error("Enter some text")


    if pdf_file is not None:

        # Saving the uploaded PDF file
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(pdf_file.read())

        # Use PyPDFLoader to load and split the PDF
        loader = PyPDFLoader("uploaded_pdf.pdf")
        pages = loader.load_and_split()

        # Calling the function to create chunks
        chunks = chunk_data(docs=pages)
        raw_text = chunks[3].page_content

        embeddings = emb_model.encode([raw_text])
        st.write(f"Embedding shape for '{raw_text}': {embeddings[0]}")






Doc_QA()