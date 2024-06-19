import streamlit as st

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.5rem;
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



def chat_with_utube():

    st.markdown(
        "<h1 style='text-align: center; font-size: 55px;'>Chat with Youtube</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
            "<p style='font-size: 22px; text-align: center;padding-left: 2rem;padding-right: 2rem;'>In times of tough market situations, fake job postings and scams often spike, posing a significant threat to job seekers. To combat this, I've developed a user-friendly module designed to protect individuals from falling prey to such fraudulent activities. This module requires users to input details about the job posting they're considering. Behind the scenes, two powerful AI models thoroughly analyze the provided information. Once completed, users receive a clear indication of whether the job posting is genuine or potentially deceptive.</p>",
            unsafe_allow_html=True,
        )
    youtube_link = st.text_input("Enter the Youtube video URL")
   

    col1,col2 = st.columns(spec=(0.7,1), gap="large")
    chat_button = False

    with col1:
        if youtube_link:
            video_id = get_youtube_id(youtube_link)
            if video_id:
                st.markdown(f"""
                    <iframe width="700" height="394" src="https://www.youtube.com/embed/{video_id}" 
                    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen></iframe>
                """, unsafe_allow_html=True)
            else:
                st.error("Invalid YouTube URL")

        st.write("")
        with st.container(border=True):
            transcrip_col,chat_col = st.columns(spec=(1,1), gap="large")
            with transcrip_col:
                transcrip_bt = st.button("Get transcript",use_container_width=True)
            with chat_col:
                chat_bt = st.button("Start Chat",use_container_width=True)
            
    with col2:
        if chat_button:
            pass

chat_with_utube()