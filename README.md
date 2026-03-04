# 🎙️ AI Grammar Scoring Engine

An AI-powered system that evaluates **spoken English grammar quality from audio recordings**.  
The system transcribes speech, extracts linguistic features, and predicts a **grammar quality score** using machine learning.

This project demonstrates how **speech recognition, NLP, and machine learning** can be combined to automatically assess spoken grammar quality.

---

# 🚀 Features

- Upload `.wav` audio files
- Speech-to-text transcription using **OpenAI Whisper**
- Linguistic feature extraction using **NLTK**
- Sentence embeddings using **Sentence Transformers**
- Grammar scoring using **XGBoost ensemble models**
- Interactive **Streamlit web interface**
- Real-time grammar quality prediction

---

## 🏗 System Architecture

Audio Input (.wav)  
 ↓  
Speech Recognition (Whisper)  
↓  
Text Transcription  
↓  
Linguistic Feature Extraction (NLTK)  
↓  
Sentence Embeddings (Sentence Transformers)  
↓  
Feature Scaling  
↓  
Machine Learning Model (XGBoost)  
↓  
Grammar Score Prediction

---

# 🧠 Model Pipeline

## 1. Speech Recognition
Audio is transcribed using:
OpenAI Whisper (base model)

Audio → Text

---

## 2. Linguistic Feature Extraction
Using **NLTK**, the following features are extracted:

- Total words
- Number of nouns
- Number of verbs
- Number of adjectives
- Average word length

---

## 3. Sentence Embeddings
Semantic features are generated using:
sentence-transformers/all-mpnet-base-v2

This produces **768-dimensional embeddings** capturing semantic meaning.

---

## 4. Feature Combination
Final input features are:

Linguistic Features  →  Sentence Embeddings  

---

## 5. Model Training
The final model uses:
XGBoost Regressor

Validation method:
5-Fold Cross Validation

Model performance:
Average CV MSE ≈ **0.77**

---

# 🌐 Streamlit Demo

The project includes a **Streamlit web application** where users can upload an audio file and receive a predicted grammar score.

Workflow:
Upload Audio → Transcription → Feature Extraction → Grammar Score Prediction

---

# 🧪 Dataset

The model was trained on a dataset containing:

Training data:
444 audio samples

Test data:
204 audio samples

Each audio file has a **human-annotated grammar quality score**.

---

# 📈 Future Improvements

Potential enhancements:

- Transformer-based grammar scoring
- Fine-tuning Whisper for speech quality
- More advanced linguistic features
- Larger dataset
- Real-time microphone input
- Cloud deployment

---

# 👨‍💻 Author

Atharva Dharmadhikari

