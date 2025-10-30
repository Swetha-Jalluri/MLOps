import numpy as np
import pandas as pd
import builtins
import types
import os

import pytest

# Import the module under test
# If your project uses "src" package structure, ensure tests run from repo root:
from src.wine_model import (
    load_wine_data,
    preprocess_data,
    train_model,
    tune_hyperparameters,
    evaluate_model,
    predict_wine_quality,
    get_feature_importance,
    plot_feature_importance,
    plot_confusion_matrix_fig,
    plot_roc_curve_fig,
    full_pipeline,
)
import mlflow


@pytest.fixture(autouse=True)
def no_external_mlflow(monkeypatch, tmp_path):
    """
    Prevent tests from writing to a real mlruns folder or needing UI deps.
    """
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)

    class DummyRun:
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(mlflow, "start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **k: None)

    # stub model logging (no-op)
    class DummySklearnModule(types.SimpleNamespace):
        def log_model(self, *a, **k): return None
    monkeypatch.setattr(mlflow, "sklearn", DummySklearnModule())


@pytest.fixture
def small_dataset():
    # 11 features like the UCI dataset
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 11)), columns=[f"f{i}" for i in range(11)])
    y = pd.Series(rng.integers(0, 2, size=60))
    X_train, X_test = X.iloc[:48], X.iloc[48:]
    y_train, y_test = y.iloc[:48], y.iloc[48:]
    return X_train, X_test, y_train, y_test


def test_preprocess_shapes(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    X_trs, X_tes, scaler = preprocess_data(X_train, X_test)
    assert X_trs.shape == (48, 11)
    assert X_tes.shape == (12, 11)
    assert hasattr(scaler, "transform")


def test_train_and_eval(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    X_trs, X_tes, scaler = preprocess_data(X_train, X_test)
    model = train_model(X_trs, y_train, n_estimators=20)  # tiny for speed
    metrics = evaluate_model(model, X_tes, y_test)
    for key in ("accuracy", "precision", "recall", "f1_score"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_predict_quality(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    X_trs, X_tes, scaler = preprocess_data(X_train, X_test)
    model = train_model(X_trs, y_train, n_estimators=10)
    preds = predict_wine_quality(model, scaler, X_test.iloc[:2].to_numpy())
    assert preds.shape == (2,)


def test_feature_importance_len(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    X_trs, X_tes, _ = preprocess_data(X_train, X_test)
    model = train_model(X_trs, y_train, n_estimators=10)
    names = X_train.columns.tolist()
    imp = get_feature_importance(model, names)
    assert isinstance(imp, dict)
    assert len(imp) == len(names)


def test_plots_created(tmp_path, small_dataset, monkeypatch):
    # prevent mlflow.log_artifact side-effects
    monkeypatch.setattr(mlflow, "log_artifact", lambda *a, **k: None)

    # feature importance plot
    imp = {f"f{i}": 1.0 / (i + 1) for i in range(11)}
    p1 = tmp_path / "fi.png"
    plot_feature_importance(imp, out_path=str(p1))
    assert p1.exists() and p1.stat().st_size > 0

    # confusion matrix
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    p2 = tmp_path / "cm.png"
    plot_confusion_matrix_fig(y_true, y_pred, out_path=str(p2))
    assert p2.exists() and p2.stat().st_size > 0

    # ROC curve
    y_score = np.array([0.1, 0.9, 0.6, 0.8])
    p3 = tmp_path / "roc.png"
    plot_roc_curve_fig(y_true, y_score, out_path=str(p3))
    assert p3.exists() and p3.stat().st_size > 0


def test_tuning_returns_best_estimator(small_dataset, monkeypatch):
    # tiny grid for speed
    from src import wine_model as wm
    monkeypatch.setattr(
        wm, "tune_hyperparameters",
        lambda X, y, **kw: (train_model(X, y, n_estimators=5), {"n_estimators": 5}, pd.DataFrame({"mean_test_score": [0.5]}))
    )
    X_train, X_test, y_train, y_test = small_dataset
    X_trs, X_tes, scaler = preprocess_data(X_train, X_test)
    model, best_params, df = wm.tune_hyperparameters(X_trs, y_train)
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params


def test_full_pipeline_smoke(monkeypatch):
    """
    Smoke test the whole pipeline without internet:
    monkeypatch load_wine_data to return synthetic data.
    """
    from src import wine_model as wm

    def fake_loader(_type="red"):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(80, 11)))
        y = pd.Series(rng.integers(0, 2, size=80))
        return X.iloc[:64], X.iloc[64:], y.iloc[:64], y.iloc[64:]

    monkeypatch.setattr(wm, "load_wine_data", fake_loader)
    out = wm.full_pipeline("red", n_estimators=10, tune=False)
    assert "model" in out and "metrics" in out and "feature_importance" in out
