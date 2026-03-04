import os
import pathlib

ffmpeg_path = pathlib.Path(__file__).parent / "ffmpeg" / "bin"
os.environ["PATH"] += os.pathsep + str(ffmpeg_path)

import streamlit as st
import whisper
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import tempfile

st.title("AI Grammar Scoring System")

st.write("Upload an audio file to evaluate grammar quality.")

# Load models
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    models = joblib.load("grammar_scoring_models.pkl")
    scaler = joblib.load("scaler.pkl")
    return whisper_model, embedder, models, scaler

whisper_model, embedder, models, scaler = load_models()

# Upload audio
uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    st.write("Transcribing audio...")

    result = whisper_model.transcribe(audio_path)
    text = result["text"]

    st.write("### Transcribed Text")
    st.write(text)

    # Generate embedding
    embedding = embedder.encode(text)
    X = np.array([embedding])
    X_scaled = scaler.transform(X)

    # Ensemble prediction
    preds = np.mean([m.predict(X_scaled) for m in models], axis=0)
    score = np.clip(preds[0], 0, 5)

    st.write("### Predicted Grammar Score")
    st.success(round(score,2))