import os
import time
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -----------------------------------------------------
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

BASE_URL = "https://api-inference.huggingface.co/models/"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# -----------------------------------------------------
# Load LOCAL MODEL from project folder
# -----------------------------------------------------

LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained-model")

local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
local_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
local_model.eval()

def predict_local_toxicity(text: str):
    inputs = local_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = local_model(**inputs).logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    label = int(probs.argmax())

    return label, float(probs[label])

# -----------------------------------------------------
def call_sentiment_model(text: str):
    resp = requests.post(BASE_URL + SENTIMENT_MODEL, headers=HEADERS, json={"inputs": text})

    if resp.status_code == 503:
        time.sleep(5)
        resp = requests.post(BASE_URL + SENTIMENT_MODEL, headers=HEADERS, json={"inputs": text})

    if resp.status_code == 200:
        return resp.json()

    return None

# -----------------------------------------------------
def analyze_text(text: str):
    text = text.strip()

    if not text:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "toxic": False,
            "toxicity_score": 0.0
        }

    # 1️⃣ Local Toxicity Model
    tox_label, tox_score = predict_local_toxicity(text)

    if tox_label == 1:
        return {
            "sentiment": "toxic",
            "confidence": tox_score,
            "toxic": True,
            "toxicity_score": tox_score
        }

    # 2️⃣ Remote Sentiment
    sent_resp = call_sentiment_model(text)

    if sent_resp and isinstance(sent_resp[0], list):
        best = max(sent_resp[0], key=lambda x: x["score"])
        label = best["label"].lower()
        score = float(best["score"])

        if "negative" in label:
            sentiment = "negative"
        elif "neutral" in label:
            sentiment = "neutral"
        else:
            sentiment = "positive"

        return {
            "sentiment": sentiment,
            "confidence": score,
            "toxic": False,
            "toxicity_score": tox_score
        }

    return {
        "sentiment": "neutral",
        "confidence": 0.0,
        "toxic": False,
        "toxicity_score": tox_score
    }
