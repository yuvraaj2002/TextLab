import streamlit as st
import tempfile
import whisper
from pytube import YouTube
from moviepy.editor import VideoFileClip
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
    try:
        # Initialize pytube to fetch YouTube video
        yt = YouTube(youtube_url)
        
        # Get the highest quality audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        # Download audio stream to a temporary file
        temp_file = audio_stream.download(filename="extracted_audio")
        
        # Convert to mp3 using moviepy
        video_clip = VideoFileClip(temp_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile("./audio.mp3")
        
        # Close and delete the temporary file
        audio_clip.close()
        video_clip.close()
        os.remove(temp_file)
        
        print("Audio saved as audio.mp3")
        
    except Exception as e:
        print(f"Error: {str(e)}")

    if os.path.exists("./extracted_audio"):
        st.audio("./extracted_audio", format="audio/mpeg", loop=True)
        os.remove("./extracted_audio")
        # result = whisper_model.transcribe("./extracted_audio")['text']
        # transcription = transcription + result 
        pass
    else:
        st.error("Audio can't be transcripted")

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
                with col1:
                    st.success("Video transcription extracted succesfully")
                    st.write(transcription)

        with chat_col:
            chat_bt = st.button("Start Chat",use_container_width=True)
            
            
    with col2:
        if transcrip_bt:
            pass
            

chat_with_utube()