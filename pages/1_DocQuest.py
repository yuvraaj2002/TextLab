import streamlit as st
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec


# Load embedding model and cache it
@st.cache_resource
def load_embedding_model():
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return emb_model


def setup_pinecone():
    # Setting up pinecone
    pc = Pinecone(api_key="c7bcca3b-f55e-4c48-b6fd-ed090e75997f")
    index_name = "docquest"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index(index_name)
    return index


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


    if pdf_file is not None:

        # Saving the uploaded PDF file
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(pdf_file.read())

        # Use PyPDFLoader to load and split the PDF
        loader = PyPDFLoader("uploaded_pdf.pdf")
        docs = loader.load_and_split()

        # Calling the function to create chunks
        chunks = chunk_data(docs=docs)

        # Prepare data for upserting in Pinecone
        data = []
        for i, chunk in enumerate(chunks):
            actual_text = chunk.page_content
            metadata = {'text': f'{actual_text}'}
            embedding = emb_model.encode(actual_text)
            data.append({"id": f"{i}", "values": embedding, "metadata": metadata})

        # Creating pandas dataframe
        data_df = pd.DataFrame(data)

        # Calling the function to setup the pinecone object
        index = setup_pinecone()

        # Upsert data into Pinecone index in batches
        batch_size = 50
        total_batches = (len(data_df) + batch_size - 1) // batch_size
        for i in (range(total_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data_df))
            batch = data_df.iloc[start_idx:end_idx]
            items = [(row["id"], row["values"], row["metadata"]) for _, row in batch.iterrows()]
            index.upsert(items=items)

        st.success("Document has been processed and upserted to the index!")




Doc_QA()