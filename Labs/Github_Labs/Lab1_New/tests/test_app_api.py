import numpy as np
import types
import pytest

from flask import Flask

# Import app module
# Ensure this path is correct for your project layout
from src import app as api


class DummyScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)  # identity-ish for test


class DummyModel:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, -1] > 10).astype(int)  # simple rule for test
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = np.clip((X[:, -1] / 20.0), 0, 1)
        return np.vstack([1 - p1, p1]).T


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    # Replace loaded artifacts on the imported app module
    monkeypatch.setattr(api, "model", DummyModel())
    monkeypatch.setattr(api, "scaler", DummyScaler())


@pytest.fixture
def client():
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_with_list(client):
    # Correct 11-length vector
    body = {"features": [7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 12.0]}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    js = r.get_json()
    assert "prediction" in js and "label" in js


def test_predict_with_dict(client):
    # Dict in any order, API reorders internally
    body = {
        "features": {
            "alcohol": 12.0,
            "sulphates": 0.56,
            "pH": 3.51,
            "density": 0.9978,
            "total sulfur dioxide": 34,
            "free sulfur dioxide": 11,
            "chlorides": 0.076,
            "residual sugar": 1.9,
            "citric acid": 0.0,
            "volatile acidity": 0.7,
            "fixed acidity": 7.4,
        }
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    js = r.get_json()
    assert js["label"] in ("Good", "Regular")


def test_predict_bad_payload(client):
    r = client.post("/predict", json={"foo": "bar"})
    assert r.status_code == 400
