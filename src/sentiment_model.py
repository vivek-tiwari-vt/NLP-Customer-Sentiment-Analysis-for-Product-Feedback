import os
import tensorflow as tf
import pandas as pd
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from typing import Tuple

def setup_gpu():
    """Configure GPU settings and memory growth"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU(s): {len(gpus)}")
        else:
            print("No GPU found. Using CPU.")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

def prepare_data(csv_path: str) -> Tuple[dict, tf.Tensor]:
    """
    Load and prepare data for training
    
    Args:
        csv_path: Path to the processed CSV file
    Returns:
        Tuple of tokenized inputs and labels
    """
    df = pd.read_csv(csv_path)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    x_train = tokenizer(
        df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='tf'
    )
    y_train = pd.get_dummies(df['sentiment']).values
    
    return x_train, y_train

def train_model():
    """Train and save the sentiment analysis model"""
    try:
        # Setup GPU configuration
        setup_gpu()
        
        # Prepare data
        print("Loading and preparing data...")
        x_train, y_train = prepare_data('data/processed/train.csv')
        
        # Build model
        print("Building model...")
        model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Train
        print("Starting training...")
        history = model.fit(
            x_train['input_ids'],
            y_train,
            epochs=3,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        # Save model
        print("Saving model...")
        model.save_pretrained('models/sentiment_model')
        print("Model saved successfully")
        
        return history
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None

if __name__ == "__main__":
    train_model()