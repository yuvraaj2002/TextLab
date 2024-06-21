import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_pinecone import PineconeVectorStore
# from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import ServerlessSpec

from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os
import io
import base64
from dotenv import load_dotenv
load_dotenv()


st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.7rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)




@st.cache_resource
def load_models():

    embedding_model = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-de")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ['GEMINI_API'])
    
    return embedding_model,llm




def split_load_data(pdf_data,index_name,embedding_model):

    # Create a BytesIO object
    file_object = io.BytesIO(pdf_data)  
    reader = PdfReader(file_object)

    # Extract text from all pages
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(pdf_text)]

    pinecone = PineconeVectorStore.from_documents(
    docs, embedding_model, index_name=index_name,pinecone_api_key=os.environ['PINECONE_API_KEY'])
    st.success("Document has been processed and stored in Vector database")
    return pinecone



def get_answer(vdb_contxt_text, query_text, llm):

    template = f"""
        Given the query '{{query_text}}', and after reviewing the information retrieved from the vector database:
        {{vdb_contxt_text}}
        Please provide a concise and informative answer that addresses the query effectively.
    """

    # Define the input variable names
    input_variables = ["query_text", "vdb_contxt_text"]

    # Create the prompt template
    prompt = PromptTemplate(input_variables=input_variables, template=template)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Pass the actual values for the variables
    response = chain.invoke({"query_text": query_text, "vdb_contxt_text": vdb_contxt_text})
    return response





def Doc_QA():
     

    # Calling the function to load the Whisper model, LLM and embedding model
    embedding_model,llm = load_models()

    col1,col2 = st.columns(spec=(1.5,1), gap="large")
    with col1:
        st.markdown(
        "<h1 style='text-align: left; font-size: 50px;'>Chat with Documents📃</h1>",
        unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 20px; text-align: left;'>Welcome to our advanced PDF Query Module, where you can harness the power of Large Language Models (LLMs) and vector databases to efficiently query your PDF documents. This module simplifies the process of interacting with your PDF files, making information retrieval both quick and intuitive.Experience the ease of querying and extracting information from your PDF documents like never before, powered by cutting-edge AI and database technology..</p>",
            unsafe_allow_html=True,
        )
        pdf_file = st.file_uploader("Upload the query document", type="pdf")
    

    if pdf_file is not None:
        with col2:
            st.write("")
            st.write("")
            pdf_data = pdf_file.read()
            b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
            pdf_display = f'<embed src="data:application/pdf;base64,{b64_pdf}" width="690" height="750" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)

        with col1:
            # Calling the function to extract the pdf data and create chunks
            pinecone = split_load_data(pdf_data,"docquest",embedding_model)
            response = None
            with st.container(border=True):
                query_text = st.text_input("Enter your query : ")
                if query_text:
                    result = pinecone.similarity_search(query_text)[:1]
                    vdb_context_text = result[0].page_content

                    # Calling the function to get the answer from the LLM
                    response = get_answer(vdb_context_text,query_text,llm)

            if response is not None:
                with st.container(border=True):
                    st.markdown(
                        f"<p style='font-size: 20px;'>{response}</p>",
                        unsafe_allow_html=True
                    )



Doc_QA()
