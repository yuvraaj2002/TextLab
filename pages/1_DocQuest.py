import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

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
    model = Ollama(model="llama3")
    embedding_model = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-de")
    
    # Setup for Pinecone vector database
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index_name = "docquest"

    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
    else:
        # Creating a new index
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    return (model,embedding_model)




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



def get_answer(context_text, query_text, llm):

    template = f"""
        Given the query '{query_text}', and after reviewing the information retrieved from the vector database:
        {context_text}
        Please provide a concise and informative answer that addresses the query effectively.
    """

    prompt = PromptTemplate(input_variables=["query_text", "context_text"], template=template)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    response = chain.invoke({"context_text": context_text, "query_text": query_text})
    return response





def Doc_QA():
     

    # Calling the function to load the Whisper model, LLM and embedding model
    llm,embedding_model = load_models()

    col1,col2 = st.columns(spec=(1.5,1), gap="large")
    with col1:
        st.markdown(
        "<h1 style='text-align: left; font-size: 50px;'>Chat with Documents</h1>",
        unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 20px; text-align: left;'>In times of tough market situations, fake job postings and scams often spike, posing a significant threat to job seekers. To combat this, I've developed a user-friendly module designed to protect individuals from falling prey to such fraudulent activities. This module requires users to input details about the job posting they're considering. Behind the sceeptive.</p>",
            unsafe_allow_html=True,
        )
        pdf_file = st.file_uploader("Upload the query document", type="pdf")
    

    if pdf_file is not None:
        with col2:
            st.write("")
            st.write("")
            pdf_data = pdf_file.read()
            b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
            pdf_display = f'<embed src="data:application/pdf;base64,{b64_pdf}" width="690" height="740" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)
        with col1:

            # Calling the function to extract the pdf data and create chunks
            pinecone = split_load_data(pdf_data,"docquest",embedding_model)
            query_text = st.text_input("Enter your query : ")
            if query_text:
                result = pinecone.similarity_search(query_text)[:1]
                context_text = result[0].page_content

                # Calling the function to get the answer from the LLM
                response = get_answer(context_text,query_text,llm)
                st.write(response)



Doc_QA()
