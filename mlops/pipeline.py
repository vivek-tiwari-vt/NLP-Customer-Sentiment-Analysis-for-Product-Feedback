import mlflow
from src.preprocessing import load_and_preprocess_data
from src.sentiment_model import train_sentiment_model
from src.topic_modeling import analyze_topics

def run_mlops_pipeline():
    """Run the complete MLOps pipeline with MLflow tracking."""
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Sentiment_Analysis")

    with mlflow.start_run(run_name="Full_Pipeline"):
        # Preprocessing
        mlflow.log_param("step", "preprocessing")
        df = load_and_preprocess_data()
        mlflow.log_metric("dataset_size", len(df))
        
        # Model Training
        mlflow.log_param("step", "training")
        model, tokenizer = train_sentiment_model()
        mlflow.pytorch.log_model(model, "bert_model")
        
        # Topic Modeling
        mlflow.log_param("step", "topic_modeling")
        topics = analyze_topics()
        mlflow.log_artifact("data/processed/topics.txt")

if __name__ == "__main__":
    run_mlops_pipeline()