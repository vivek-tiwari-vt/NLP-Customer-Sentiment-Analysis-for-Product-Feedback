from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_sentiment():
    response = client.post("/predict", json={"text": "I love this product"})
    assert response.status_code == 200
    assert response.json()["sentiment"] in ["positive", "negative"]