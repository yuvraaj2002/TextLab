import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os

from dotenv import load_dotenv
load_dotenv()

# Load embedding model and cache it
@st.cache_resource
def load_embedding_model():
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return emb_model

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs

def Doc_QA():
    st.title("Doc Quest")
    pdf_file = st.file_uploader("Upload the query document", type="pdf")
    emb_model = load_embedding_model()

    # Initialize Pinecone client
    pc = Pinecone(api_key=os.environ['PINECONE_API'])
    index_name = "docquest"

    if pdf_file is not None:
        # Check if index exists, delete if it does
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)

        # Save the uploaded PDF file
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(pdf_file.read())

        # Use PyPDFLoader to load and split the PDF
        loader = PyPDFLoader("uploaded_pdf.pdf")
        docs = loader.load_and_split()

        # Calling the function to create chunks
        chunks = chunk_data(docs=docs)

        data = []
        for i, chunk in enumerate(chunks):
            actual_text = chunk.page_content
            metadata = {'text': f'{actual_text}'}
            embedding = emb_model.encode(actual_text).tolist()
            data.append((f"{i}", embedding, metadata))

        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        # Get the index reference
        index = pc.Index(index_name)

        # Upsert data in batches
        batch_size = 50
        total_batches = (len(data) + batch_size - 1) // batch_size
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            batch = data[start_idx:end_idx]
            index.upsert(vectors=batch, namespace="ns1")

        st.success("Document has been processed and upserted to the index!")

Doc_QA()
