# src/evaluate.py
import pandas as pd
from sklearn.metrics import classification_report
from transformers import pipeline

def evaluate_model():
    df = pd.read_csv('data/processed/processed_data.csv')
    pipe = pipeline(model="models/sentiment_model")
    
    # Predict
    preds = pipe(df['text'].tolist())
    y_true = df['sentiment'].map({'negative': 0, 'positive': 1})
    y_pred = [1 if p['label'] == 'positive' else 0 for p in preds]
    
    # Metrics
    print(classification_report(y_true, y_pred))