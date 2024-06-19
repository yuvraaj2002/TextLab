import streamlit as st
import tempfile
import whisper
from pytube import YouTube
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
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


# Function to extract the video ID from the YouTube URL
def get_youtube_id(url):
    import re
    regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    match = re.match(regex, url)
    return match.group(6) if match else None


@st.cache_resource
def load_models():
    model = Ollama(model="llama3")
    embedding_model = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-de")
    whisper_model = whisper.load_model("base")
    return [whisper_model,model,embedding_model]


def get_video_transcript(whisper_model, youtube_url):

    transcription = ""
    youtube = YouTube(youtube_url)
    audio = youtube.streams.filter(only_audio=True).first()

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        # Always write the transcription to the file, overwriting if it exists
        with open("./video_content.txt", "w") as file:
            file.write(transcription)

    return transcription


def chat_with_utube():

    st.markdown(
        "<h1 style='text-align: left; font-size: 50px;'>Chat with Youtube</h1>",
        unsafe_allow_html=True,
    )

    # Calling the function to load the Whisper model, LLM and embedding model
    whisper,llm,embedding_model = load_models()
    

    col1,col2 = st.columns(spec=(1.3,1), gap="large")
    chat_bt = False
    transcrip_bt = False

    with col1:
        st.markdown(
            "<p style='font-size: 20px; text-align: left;'>In times of tough market situations, fake job postings and scams often spike, posing a significant threat to job seekers. To combat this, I've developed a user-friendly module designed to protect individuals from falling prey to such fraudulent activities. This module requires users to input details about the job posting they're considering. Behind the scenes, two  provided information. Once completed, users receive a clear indication of whether the job posting is genuine or potentially deceptive.</p>",
            unsafe_allow_html=True,
        )
        youtube_link = st.text_input("Enter the Youtube video URL")
        if youtube_link:
            video_id = get_youtube_id(youtube_link)
            if video_id:
                st.markdown(f"""
                    <iframe width="950" height="420" src="https://www.youtube.com/embed/{video_id}" 
                    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen></iframe>
                """, unsafe_allow_html=True)
            else:
                st.error("Invalid YouTube URL")

        st.write("")
        transcrip_col,chat_col = st.columns(spec=(1,1), gap="large")
        with transcrip_col:
            transcrip_bt = st.button("Get transcript",use_container_width=True)
            if transcrip_bt:
                transcription = get_video_transcript(whisper,youtube_link)
                with chat_col:
                    st.success("Video transcription extracted succesfully")
                    st.write(transcription)

        with chat_col:
            chat_bt = st.button("Start Chat",use_container_width=True)
            
            
    with col2:
        if transcrip_bt:
            pass
            

chat_with_utube()