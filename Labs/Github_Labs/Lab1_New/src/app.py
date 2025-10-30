from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path

# Package-relative import for pytest (importing `src.app`), with fallback for direct runs.
try:
    from .wine_model import load_model, predict_wine_quality
except ImportError:
    from wine_model import load_model, predict_wine_quality


FEATURE_ORDER = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

app = Flask(__name__)

# Resolve model paths relative to project root (parent of src/)
BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "model.pkl"
SCALER_PATH = BASE / "scaler.pkl"

# Load trained artifacts if present; tests will monkeypatch `model`/`scaler`.
model = None
scaler = None
try:
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model, scaler = load_model(str(MODEL_PATH), str(SCALER_PATH))
except Exception:
    # Leave as None; tests will patch; API returns 503 if not loaded
    pass


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    """
    Request JSON formats supported:

    1) Ordered list (11 numbers):
       { "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4] }

    2) Named dict (keys in any order, will be reordered internally):
       {
         "features": {
           "alcohol": 9.4, "sulphates": 0.56, "pH": 3.51, "density": 0.9978,
           "total sulfur dioxide": 34, "free sulfur dioxide": 11, "chlorides": 0.076,
           "residual sugar": 1.9, "citric acid": 0.0, "volatile acidity": 0.7, "fixed acidity": 7.4
         }
       }
    """
    payload = request.get_json(silent=True) or {}
    if "features" not in payload:
        return jsonify({"error": "Missing 'features'"}), 400

    # Ensure artifacts are loaded
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Train first or provide model.pkl/scaler.pkl."}), 503

    feats = payload["features"]

    try:
        # Case 1: list/array of length 11 in correct order
        if isinstance(feats, (list, tuple)):
            arr = np.array(feats, dtype=float).reshape(1, -1)
            if arr.shape[1] != len(FEATURE_ORDER):
                return jsonify({"error": f"Expected {len(FEATURE_ORDER)} values in order {FEATURE_ORDER}"}), 400

        # Case 2: dict of named features (order-insensitive)
        elif isinstance(feats, dict):
            missing = [name for name in FEATURE_ORDER if name not in feats]
            if missing:
                return jsonify({"error": f"Missing keys: {missing}"}), 400
            ordered = [feats[name] for name in FEATURE_ORDER]
            arr = np.array(ordered, dtype=float).reshape(1, -1)

        else:
            return jsonify({"error": "Unsupported 'features' format; use list or dict"}), 400

        # Predict
        pred = predict_wine_quality(model, scaler, arr)
        label = "Good" if int(pred[0]) == 1 else "Regular"

        return jsonify(
            {
                "prediction": int(pred[0]),
                "label": label,
                "feature_order": FEATURE_ORDER,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use a different port than MLflow UI (which often uses 5000)
    app.run(host="0.0.0.0", port=5001)
