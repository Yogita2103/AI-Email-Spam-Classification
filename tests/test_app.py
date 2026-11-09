import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import app  # noqa: E402

MODEL_EXISTS = os.path.exists(os.path.join(ROOT, "model", "spam_pipeline.joblib"))
needs_model = pytest.mark.skipif(not MODEL_EXISTS, reason="Run python src/train.py first")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_home_page_loads(client):
    assert client.get("/").status_code == 200


def test_empty_text_rejected(client):
    res = client.post("/predict", json={"text": "   "})
    assert res.status_code == 400


@needs_model
def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"


@needs_model
def test_predict_spam(client):
    res = client.post("/predict", json={
        "text": "WIN a FREE prize now!!! Click http://claim.biz to get your cash money"})
    assert res.status_code == 200
    assert res.get_json()["label"] == "Spam"


@needs_model
def test_metrics(client):
    data = client.get("/metrics").get_json()
    assert 0 <= data["accuracy"] <= 1
