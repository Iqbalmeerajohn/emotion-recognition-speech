# 🎤 Speech Emotion Recognition
### Multi-Head Attention + Federated Learning | 92% Accuracy | Top 30 / 350 Projects @ GITAM

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-92%25-brightgreen?style=flat)
![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-blue?style=flat)

---

## 🧠 Overview

A production-ready **Speech Emotion Recognition (SER)** system that classifies human emotions from audio input in real time. Built using deep learning with **Multi-Head Attention mechanisms** and trained using a **Federated Learning** framework to preserve data privacy — a technique increasingly critical in healthcare, mental health, and enterprise AI applications.

This project was **selected among the Top 30 out of approximately 350 projects** at the GITAM University technical poster presentation (2026).

---

## 🏆 Key Achievements

| Metric | Result |
|--------|--------|
| **Accuracy** | 92% on RAVDESS dataset |
| **Recognition** | Top 30 / ~350 projects @ GITAM Tech Poster 2026 |
| **Dataset** | RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) |
| **Emotions Classified** | 8 classes (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised) |
| **Architecture** | Multi-Head Attention + Deep Neural Network |
| **Privacy Approach** | Federated Learning for distributed, privacy-preserving training |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SYSTEM OVERVIEW                    │
├──────────────────┬──────────────────────────────────┤
│  FRONTEND        │  BACKEND                         │
│  (Streamlit)     │  (FastAPI)                       │
│                  │                                  │
│  • WAV Upload    │  • REST API /predict             │
│  • Mic Record    │  • TensorFlow Model Inference    │
│  • Confidence    │  • MFCC Feature Extraction       │
│    Bar Chart     │  • Emotion Classification        │
└──────────────────┴──────────────────────────────────┘

AUDIO INPUT  →  MFCC Extraction (120 features, 94 frames)
             →  Multi-Head Attention Layers
             →  Dense Classification Head
             →  8-Class Emotion Output + Confidence Scores
```

**Model Training Approach:**
- Audio preprocessed using **librosa** (MFCC extraction: 120 coefficients, 94 time steps)
- **Federated Learning** simulated across partitioned data clients — model aggregated via FedAvg
- Final global model (`global_federated_model_grouped.keras`) achieves **92% accuracy** on held-out RAVDESS test set

---

## 📁 Project Structure

```
emotion-recognition-speech/
├── app.py                    # FastAPI backend — REST API for emotion prediction
├── streamlit_app.py          # Streamlit frontend — upload or record audio
├── utils/
│   ├── audio_processing.py   # MFCC feature extraction using librosa
│   └── emotion_labels.py     # Emotion class index → label mapping
├── model/
│   └── global_federated_model_grouped.keras   # Trained TF model (add manually)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `pip` or `conda`
- Trained model file: `model/global_federated_model_grouped.keras`

### Installation

```bash
# Clone the repository
git clone https://github.com/Iqbalmeerajohn/emotion-recognition-speech.git
cd emotion-recognition-speech

# Install dependencies
pip install -r requirements.txt
```

### Run the Backend (FastAPI)

```bash
uvicorn app:app --reload
# API available at http://127.0.0.1:8000
# Swagger docs at http://127.0.0.1:8000/docs
```

### Run the Frontend (Streamlit)

```bash
# In a separate terminal
streamlit run streamlit_app.py
# UI available at http://localhost:8501
```

---

## 🎯 Usage

**Option 1 — Upload a WAV File:**
1. Open the Streamlit UI at `http://localhost:8501`
2. Click **"Upload a WAV file"** and select a `.wav` audio file
3. The system sends the file to FastAPI, runs MFCC extraction + model inference
4. View the predicted emotion, confidence score, and full probability bar chart

**Option 2 — Record from Microphone:**
1. Adjust the recording duration slider (2–10 seconds)
2. Click **"Start Recording"**
3. Speak naturally — the system records, processes, and predicts in real time

**REST API directly:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@your_audio.wav"
```

Sample response:
```json
{
  "predicted_emotion": "Happy",
  "confidence": 0.9123,
  "all_probabilities": {
    "Neutral": 0.0210,
    "Calm": 0.0180,
    "Happy": 0.9123,
    "Sad": 0.0050,
    "Angry": 0.0200,
    "Fearful": 0.0100,
    "Disgust": 0.0067,
    "Surprised": 0.0070
  }
}
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **Deep Learning** | TensorFlow / Keras |
| **Architecture** | Multi-Head Attention + Dense Layers |
| **Training Paradigm** | Federated Learning (FedAvg) |
| **Audio Processing** | librosa (MFCC extraction) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend UI** | Streamlit |
| **Dataset** | RAVDESS |
| **Language** | Python 3.10+ |

---

## 🗺️ Future Roadmap

- [ ] Deploy backend on AWS Lambda / Google Cloud Run
- [ ] Add real Federated Learning with Flower (`flwr`) framework
- [ ] Extend to multilingual speech emotion recognition
- [ ] Add speaker diarization support
- [ ] Model quantization for edge deployment (mobile/IoT)
- [ ] Build a REST SDK for third-party integration

---

## 👨‍💻 Author

**Sheik Iqbal Meera John**  
B.Tech CSE (Data Science) — GITAM University, 2026  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/sheik-iqbal-meera-john-056191253)  
[![GitHub](https://img.shields.io/badge/GitHub-Iqbalmeerajohn-181717?style=flat&logo=github)](https://github.com/Iqbalmeerajohn)

---

> *"Privacy-preserving AI is not the future — it is the present."*
