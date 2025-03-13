# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
sentiment_pipeline = pipeline(
    model="models/sentiment_model",
    tokenizer="distilbert-base-uncased"
)

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(text: InputText):
    result = sentiment_pipeline(text.text)[0]
    return {
        "sentiment": result['label'],
        "confidence": round(result['score'], 4)
    }