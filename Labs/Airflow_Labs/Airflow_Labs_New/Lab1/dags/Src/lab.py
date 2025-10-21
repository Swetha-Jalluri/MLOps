import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import os

# Load data - using sklearn's iris dataset
def load_data():
    print("Loading Iris dataset...")
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    print(f"Data loaded with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Target classes: {np.unique(iris.target)}")
    return pickle.dumps(df)

# Preprocess the data
def data_preprocessing(data):
    print("Preprocessing data...")
    df = pickle.loads(data)
    
    print(f"Initial data shape: {df.shape}")
    
    # Drop any missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Convert to binary: class 2 = 1 (Virginica), else = 0
    y_binary = (y == 2).astype(int)
    
    print(f"Class distribution: {np.bincount(y_binary)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Combine back
    preprocessed_df = pd.DataFrame(X_scaled, columns=X.columns)
    preprocessed_df['target'] = y_binary
    
    print(f"Preprocessed data shape: {preprocessed_df.shape}")
    print(f"Target distribution:\n{preprocessed_df['target'].value_counts()}")
    
    return pickle.dumps(preprocessed_df)

# Build and save Random Forest model
def build_save_model(data, filename):
    print("Building Random Forest model...")
    df = pickle.loads(data)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{'='*50}\n")
    
    # Save model
    model_dir = '/opt/airflow/dags/model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'train_accuracy': model.score(X_train, y_train)
    }
    return pickle.dumps(metrics)

# Load model and show feature importance
def load_model_elbow(filename, metrics_data):
    print("Loading model and analyzing feature importance...")
    
    model_path = os.path.join('/opt/airflow/dags/model', filename)
    model = joblib.load(model_path)
    metrics = pickle.loads(metrics_data)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS - IRIS CLASSIFICATION (Virginica Detection)")
    print(f"{'='*50}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    
    # Feature importance
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance Ranking:")
    print(feature_importance_df.to_string())
    print(f"{'='*50}\n")
    
    return "Model analysis complete!"
