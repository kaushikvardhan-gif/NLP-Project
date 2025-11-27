🎭 Emotion Recognition using DistilRoBERTa (GoEmotions)

This project performs fine-grained emotion detection using a DistilRoBERTa-based model trained on the GoEmotions dataset (28 emotions + Neutral).
It includes a Flask Web App with speech-to-text, auto-stop mic, and a modern animated UI.

Features:
28-Class Emotion Recognition
Predicts nuanced emotions like:
joy, sadness, anger, surprise, gratitude, pride, love, nervousness, remorse, excitement, confusion, etc.
Multi-Label Output
Sentences can express multiple emotions simultaneously.
Real-Time Web App
Built using Flask, featuring:
Text input

Speech input

Auto-stop mic when silence is detected

Animated mic button

Gradient emotion bars

✔ Speech-to-Text Support

Uses Whisper (OpenAI) for accurate transcription.

🧠 Model Information
Base Model:

DistilRoBERTa-base fine-tuned on GoEmotions.

Dataset:

GoEmotions (58k Reddit comments, 28 emotion labels)

Architecture Pipeline:

Tokenization

DistilRoBERTa encoder

Sigmoid multi-label output layer

Emotion probabilities

📊 Performance Comparison
Model	Accuracy	Macro F1	Micro F1
TF-IDF + SVM	63%	0.56	0.59
DistilBERT	74%	0.67	0.71
DistilRoBERTa (Our Model)	~88%	0.82	0.85
Improvement Over Previous Systems:

Detects 28 emotions (vs only 6–10 earlier)

Handles multi-emotion sentences

Better performance on sarcasm, mixed feelings, short text

📂 Project Structure
emotion_app/
│── app.py                 # Flask backend
│── train_emotion_model.py # Model training script
│── trained_emotion_model/ # Saved model folder
│── static/
│     ├── style.css        # UI styling
│     └── app.js           # Frontend JS logic
│── templates/
│     └── index.html       # Web UI
│── results/               # Training logs / figures
│── venv/                  # Virtual environment (ignored)
└── .gitignore

🛠 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/kaushikvardhan-gif/NLP-Project.git
cd NLP-Project

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate    # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train Model (Optional – already provided)
python train_emotion_model.py

5️⃣ Run the Web App
python app.py


Visit: http://127.0.0.1:5000

🧪 Examples

Input:

“I’m happy but a little nervous.”

Output:

joy (72%)

nervousness (41%)

excitement (38%)

🗺 Project Timeline
Week	Task
1	Dataset exploration & preprocessing
2–3	Model training & fine-tuning
4	Evaluation & optimization
5	Flask + UI development
6	Testing, final report, presentation

📌 Future Enhancements
Add speech emotion recognition
Build mobile app version
Add real-time webcam facial emotion analysis
Deploy on Hugging Face Spaces / Render

Contributors:
Bhanu Prakash
Kaushik Vardhan
Larib Khan
Akarshan
Sashank

License
This project is for educational & research purposes
