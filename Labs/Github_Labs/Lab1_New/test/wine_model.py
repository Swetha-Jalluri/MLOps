"""
Wine Quality Prediction Model
This module contains functions for loading wine quality data,
training a Random Forest model, and making predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_wine_data(wine_type='red'):
    """
    Load wine quality dataset from UCI repository
    
    Args:
        wine_type (str): 'red' or 'white' wine dataset
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # URLs for wine quality datasets
    urls = {
        'red': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        'white': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    }
    
    try:
        # Load data from URL
        data = pd.read_csv(urls[wine_type], sep=';')
        
        # Separate features and target
        X = data.drop('quality', axis=1)
        y = data['quality']
        
        # Convert to binary classification (good wine: quality >= 7)
        y = (y >= 7).astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return sample data for testing
        np.random.seed(42)
        X_train = np.random.randn(100, 11)
        X_test = np.random.randn(25, 11)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 25)
        return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """
    Scale features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        tuple: Scaled X_train and X_test
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, n_estimators=100):
    """
    Train a Random Forest Classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
    
    Returns:
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Dictionary containing metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics


def predict_wine_quality(model, scaler, features):
    """
    Predict wine quality for new samples
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        features: Feature array
    
    Returns:
        array: Predictions (0: regular wine, 1: good wine)
    """
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    return predictions


def get_feature_importance(model, feature_names):
    """
    Get feature importance from the trained model
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
    
    Returns:
        dict: Feature importance scores
    """
    importance = model.feature_importances_
    feature_importance = {
        name: score for name, score in zip(feature_names, importance)
    }
    # Sort by importance
    sorted_importance = dict(sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    return sorted_importance


def save_model(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    """
    Save the trained model and scaler
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        model_path: Path to save model
        scaler_path: Path to save scaler
    
    Returns:
        tuple: Paths where files were saved
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def load_model(model_path='model.pkl', scaler_path='scaler.pkl'):
    """
    Load a saved model and scaler
    
    Args:
        model_path: Path to model file
        scaler_path: Path to scaler file
    
    Returns:
        tuple: Loaded model and scaler
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def full_pipeline(wine_type='red'):
    """
    Run the complete training pipeline
    
    Args:
        wine_type: Type of wine ('red' or 'white')
    
    Returns:
        dict: Results containing model, metrics, and paths
    """
    # Load data
    X_train, X_test, y_train, y_test = load_wine_data(wine_type)
    
    # Preprocess
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    # Train
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Get feature importance
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    importance = get_feature_importance(model, feature_names)
    
    # Save model
    model_path, scaler_path = save_model(model, scaler)
    
    results = {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_importance': importance,
        'model_path': model_path,
        'scaler_path': scaler_path
    }
    
    return results


if __name__ == "__main__":
    # Run the pipeline when script is executed directly
    print("Training Wine Quality Model...")
    results = full_pipeline('red')
    
    print("\nModel Performance:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTop 5 Important Features:")
    for i, (feature, importance) in enumerate(results['feature_importance'].items()):
        if i >= 5:
            break
        print(f"{feature}: {importance:.4f}")