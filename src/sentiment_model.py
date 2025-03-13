# src/train_sentiment.py
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd

def train_model():
    # Load processed data
    df = pd.read_csv('data/processed/processed_data.csv')
    
    # Prepare dataset
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    x_train = tokenizer(
        df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='tf'
    )
    y_train = pd.get_dummies(df['sentiment']).values  # [negative, positive]

    # Build model
    model = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(
        x_train['input_ids'],
        y_train,
        epochs=3,
        batch_size=16,
        validation_split=0.2
    )

    # Save
    model.save_pretrained('models/sentiment_model')