import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    # Load model and scaler
    model = joblib.load("../model/wine_model.pkl")
    scaler = joblib.load("../model/wine_scaler.pkl")
    
    # Scale features and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return y_pred