"""
Wine Quality Prediction Model
This module contains functions for loading wine quality data,
training a Random Forest model, and making predictions.
"""

# ======================
# Imports
# ======================
import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # non-interactive backend suitable for CI/tests
import matplotlib.pyplot as plt


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Optional: keep runs in project folder and use a named experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("wine-quality-lab")


# ======================
# Data loading & preprocessing
# ======================
def load_wine_data(wine_type: str = "red"):
    """
    Load wine quality dataset from UCI repository.

    Args:
        wine_type (str): 'red' or 'white'

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    urls = {
        "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    }
    try:
        data = pd.read_csv(urls[wine_type], sep=";")
        X = data.drop("quality", axis=1)
        y = (data["quality"] >= 7).astype(int)  # binary label: good (>=7)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback synthetic data (keeps pipeline runnable offline)
        np.random.seed(42)
        X_train = np.random.randn(100, 11)
        X_test = np.random.randn(25, 11)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 25)
        return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """
    Scale features using StandardScaler.

    Returns:
        tuple: X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ======================
# Modeling helpers
# ======================
def train_model(X_train, y_train, n_estimators: int = 100):
    """
    Train a Random Forest Classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def tune_hyperparameters(X_train, y_train, scoring: str = "f1", cv_splits: int = 3, n_jobs: int = -1):
    """
    Run GridSearchCV to find better RandomForest hyperparameters.

    Returns:
        best_estimator_: fitted RandomForestClassifier
        best_params_: dict
        cv_results_df: pandas DataFrame of CV results
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(grid.cv_results_)
    return grid.best_estimator_, grid.best_params_, cv_results_df


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance. Returns dict of metrics.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }


def predict_wine_quality(model, scaler, features):
    """
    Predict wine quality for new samples (0=regular, 1=good).
    """
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)


def get_feature_importance(model, feature_names):
    """
    Compute sorted feature importances from a fitted RF model.
    """
    importance = model.feature_importances_
    pairs = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return {name: score for name, score in pairs}


# ======================
# Visualization helpers (logged to MLflow)
# ======================
def plot_feature_importance(importance_dict, out_path: str = "feature_importance.png"):
    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(9, 5))
    sns.barplot(x=values, y=labels)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    try:
        mlflow.log_artifact(out_path)
    except Exception:
        pass


def plot_confusion_matrix_fig(y_true, y_pred, out_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    try:
        mlflow.log_artifact(out_path)
    except Exception:
        pass


def plot_roc_curve_fig(y_true, y_score, out_path: str = "roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    try:
        mlflow.log_artifact(out_path)
        mlflow.log_metric("roc_auc", float(roc_auc))
    except Exception:
        pass


# ======================
# Persistence
# ======================
def save_model(model, scaler, model_path: str = "model.pkl", scaler_path: str = "scaler.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def load_model(model_path: str = "model.pkl", scaler_path: str = "scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ======================
# Main pipeline
# ======================
def full_pipeline(wine_type: str = "red", n_estimators: int = 100, tune: bool = False):
    """
    Run the complete training pipeline with optional hyperparameter tuning and MLflow tracking.

    Args:
        wine_type: 'red' or 'white'
        n_estimators: used when tune=False
        tune: if True, runs GridSearchCV and uses the best estimator
    """
    with mlflow.start_run():
        # Log params
        mlflow.log_param("wine_type", wine_type)
        mlflow.log_param("tune", tune)
        mlflow.log_param("n_estimators_initial", n_estimators)

        # Data
        X_train, X_test, y_train, y_test = load_wine_data(wine_type)
        X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

        # Train
        if tune:
            model, best_params, cv_results_df = tune_hyperparameters(
                X_train_scaled, y_train, scoring="f1", cv_splits=3, n_jobs=-1
            )
            for k, v in best_params.items():
                mlflow.log_param(f"best_{k}", v)
            try:
                cv_csv = "gridsearch_results.csv"
                cv_results_df.to_csv(cv_csv, index=False)
                mlflow.log_artifact(cv_csv)
            except Exception:
                pass
        else:
            model = train_model(X_train_scaled, y_train, n_estimators=n_estimators)

        # Metrics
        metrics = evaluate_model(model, X_test_scaled, y_test)
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass

        # Feature importance (compute before plotting)
        if hasattr(X_train, "columns"):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        importance = get_feature_importance(model, feature_names)

        # Visualizations (logged as artifacts)
        plot_feature_importance(importance)
        y_pred = model.predict(X_test_scaled)
        plot_confusion_matrix_fig(y_test, y_pred)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            plot_roc_curve_fig(y_test, y_proba)

        # Save artifacts
        model_path, scaler_path = save_model(model, scaler)
        fi_json_path = "feature_importance.json"
        with open(fi_json_path, "w", encoding="utf-8") as f:
            json.dump(importance, f, indent=2)
        mlflow.log_artifact(fi_json_path)

        # Log model with signature & input example (helps downstream serving)
        try:
            signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
            input_example = X_train_scaled[:1]
            # artifact_path is widely supported; if your MLflow warns, switching to name=... may work.
            mlflow.sklearn.log_model(
                model,
                artifact_path="rf_model",
                signature=signature,
                input_example=input_example,
            )
        except Exception:
            pass

        try:
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(scaler_path)
        except Exception:
            pass

        return {
            "model": model,
            "scaler": scaler,
            "metrics": metrics,
            "feature_importance": importance,
            "model_path": model_path,
            "scaler_path": scaler_path,
        }


# ======================
# Script entrypoint
# ======================
if __name__ == "__main__":
    print("Training Wine Quality Model (with tuning)...")
    results = full_pipeline("red", n_estimators=100, tune=True)

    print("\nModel Performance:")
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nTop 5 Important Features:")
    for i, (feature, importance) in enumerate(results["feature_importance"].items()):
        if i >= 5:
            break
        print(f"{feature}: {importance:.4f}")
