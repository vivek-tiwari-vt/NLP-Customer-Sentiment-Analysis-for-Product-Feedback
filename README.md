# NLP Customer Sentiment Analysis for Product Feedback

This project aims to classify sentiment in customer reviews and social media posts, identify key topics driving sentiment, and provide actionable insights through dashboards and marketing strategies.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Dashboard](#dashboard)
- [API](#api)
- [MLOps Pipeline](#mlops-pipeline)
- [Testing](#testing)
- [Dependencies](#dependencies)

## Project Overview
This project leverages Natural Language Processing (NLP) techniques to analyze customer feedback. It includes sentiment analysis, topic modeling, and a dashboard for visualizing insights. The project is structured to support MLOps practices, including model training, evaluation, and deployment.

## Project Structure
```
NLP-Customer-Sentiment-Analysis-for-Product-Feedback/
├── data/
│   ├── raw/
│   ├── processed/
├── deployment/
│   ├── Procfile
│   ├── requirements.txt
│   ├── runtime.txt
├── mlops/
│   ├── __init__.py
│   ├── mlflow_tracking.py
│   ├── pipeline.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── dashboard.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── sentiment_model.py
│   ├── topic_modeling.py
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
├── .gitignore
├── README.md
├── setup.py
```

## Setup Instructions
1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/NLP-Customer-Sentiment-Analysis-for-Product-Feedback.git
    cd NLP-Customer-Sentiment-Analysis-for-Product-Feedback
    ```

2. **Create a virtual environment and activate it:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r deployment/requirements.txt
    ```

## Usage
### Model Training
To train the sentiment analysis model, run:
```sh
python src/sentiment_model.py
```

### Evaluation
To evaluate the trained model, run:
```sh
python src/evaluate.py
```

### Dashboard
To run the Streamlit dashboard, execute:
```sh
streamlit run src/dashboard.py
```

### API
To start the FastAPI server, run:
```sh
uvicorn src.api:app --reload
```

### MLOps Pipeline
To run the complete MLOps pipeline with MLflow tracking, execute:
```sh
python mlops/pipeline.py
```

## Testing
To run the tests, use:
```sh
pytest
```

## Dependencies
- Python 3.8.10
- fastapi==0.68.1
- uvicorn==0.15.0
- transformers==4.11.3
- torch==1.9.0
- pandas==1.3.3
- nltk==3.6.3
- scikit-learn==1.0.1
- streamlit==1.0.0
- mlflow==1.20.2
- pytest==6.2.5
