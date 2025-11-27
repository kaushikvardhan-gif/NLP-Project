from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
import tempfile

app = Flask(__name__)
CORS(app)

MODEL_DIR = "trained_emotion_model"
NUM_LABELS = 28  # GoEmotions

if not os.path.exists(MODEL_DIR):
    raise Exception("❌ Model not found! Run: python train_emotion_model.py")

print("\n📦 Loading Emotion Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print("✅ Emotion model loaded!")

# ------------ Load Whisper for voice → text ----------
print("🎤 Loading Whisper model (tiny, GPU)...")
whisper_model = WhisperModel("tiny", device="cuda", compute_type="float16")
print("✅ Whisper loaded!\n")

# GoEmotions labels
labels = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]


# ---------- Emotion Prediction ----------
@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("sentence", "")

    if not text.strip():
        return jsonify({"predictions": []})

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].numpy()

    selected = np.where(probs > 0.5)[0].tolist()
    if len(selected) == 0:
        selected = [int(probs.argmax())]

    result = [(labels[i], float(probs[i])) for i in selected]

    return jsonify({"predictions": result})


# ---------- Voice to Text (audio → Whisper) ----------
@app.route("/speech", methods=["POST"])
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "no audio"}), 400

    audio_file = request.files["audio"]

    # save temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        path = tmp.name

    # whisper transcribe
    segments, _ = whisper_model.transcribe(path)
    text = " ".join([seg.text for seg in segments]).strip()

    return jsonify({"text": text})


# ---------- Serve UI ----------
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
