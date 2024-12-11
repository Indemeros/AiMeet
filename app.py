import streamlit as st
import whisper
from transformers import pipeline
from pydub import AudioSegment
from io import BytesIO
import numpy as np
from scipy.io import wavfile

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Load Hugging Face Summarization pipeline
@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarization_model()

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    try:
        # Convert audio file to WAV format
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        temp_wav = BytesIO()
        audio.export(temp_wav, format="wav")
        temp_wav.seek(0)

        # Convert WAV to NumPy array
        temp_wav.seek(0)
        sample_rate, audio_data = wavfile.read(temp_wav)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize

        # Transcribe using Whisper
        transcription = whisper_model.transcribe(audio_data)
        if not transcription.get("text"):
            raise ValueError("Transcription returned empty text.")
        return transcription["text"]
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {e}")

# Function to summarize text using Hugging Face
def summarize_text(transcription_text):
    try:
        transcription_text = transcription_text.strip()
        if len(transcription_text.split()) < 50:
            raise ValueError("Transcription text is too short for summarization.")

        # Truncate text to fit model's input size
        max_input_length = 1024
        transcription_text = transcription_text[:max_input_length]

        # Generate summary
        summary = summarizer(
            transcription_text, max_length=150, min_length=50, do_sample=False
        )
        if not summary or not summary[0].get("summary_text"):
            raise ValueError("Summarization returned empty text.")
        return summary[0]["summary_text"]
    except Exception as e:
        raise RuntimeError(f"Error during summarization: {e}")

# Streamlit App Code
st.title("AiMeet: Meeting Notes Generator")
st.subheader("Upload your meeting audio file to generate structured meeting notes.")

# Audio file uploader
uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_audio is not None:
    st.info("Audio file uploaded successfully!")

    # Step 1: Transcription
    with st.spinner("Transcribing audio..."):
        try:
            transcription_text = transcribe_audio(uploaded_audio)
            st.success("Transcription completed!")
            st.text_area("Transcription", transcription_text, height=200)
        except Exception as e:
            st.error(f"Error in transcription: {e}")
            transcription_text = None

    if transcription_text:
        # Step 2: Summarization
        with st.spinner("Generating meeting notes..."):
            try:
                meeting_notes = summarize_text(transcription_text)
                st.success("Meeting notes generated!")
                st.markdown(meeting_notes, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in summarization: {e}")
                meeting_notes = None

        # Step 3: Download meeting notes
        if meeting_notes:
            notes_file = BytesIO(meeting_notes.encode("utf-8"))
            st.download_button(
                label="Download Meeting Notes",
                data=notes_file,
                file_name="meeting_notes.txt",
                mime="text/plain"
            )

st.sidebar.header("About the App")
st.sidebar.info(
    """
    **AiMeet: Meeting Notes Generator** simplifies the process of creating 
    structured meeting notes. Upload an audio file, and the app will transcribe and summarize 
    it for you using Whisper and Hugging Face.
    """
)
