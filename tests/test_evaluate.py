# tests/test_evaluate.py
"""
Educational Goal:
- Why this test exists in an MLOps system: Ensure the evaluation module strictly honors the pipeline contract.
- Responsibility: Verify that valid inputs produce mathematically sound metrics, and invalid inputs crash early.
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.evaluate import evaluate_model


@pytest.fixture
def dummy_classification_data():
    """Provides a minimal dataset for classification testing."""
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1], name="target")

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    return model, X, y


@pytest.fixture
def dummy_regression_data():
    """Provides a minimal dataset for regression testing."""
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 1.0, 1.0]})
    y = pd.Series([0.0, 1.0, 2.0, 3.0], name="target")

    model = DummyRegressor(strategy="mean")
    model.fit(X, y)

    return model, X, y


# --------------------------------------------------------
# 1) HAPPY PATH: Mathematical Soundness
# --------------------------------------------------------
def test_evaluate_classification_returns_dict_of_metrics(dummy_classification_data):
    """Classification evaluation must return a dict with pr_auc and roc_auc between 0.0 and 1.0."""
    model, X, y = dummy_classification_data

    metrics = evaluate_model(model=model, X_eval=X,
                             y_eval=y, problem_type="classification")

    assert isinstance(metrics, dict)
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert isinstance(metrics["pr_auc"], float)
    assert isinstance(metrics["roc_auc"], float)
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_regression_returns_dict_with_rmse(dummy_regression_data):
    """Regression evaluation must return a dict with rmse >= 0.0."""
    model, X, y = dummy_regression_data

    metrics = evaluate_model(model=model, X_eval=X,
                             y_eval=y, problem_type="regression")

    assert isinstance(metrics, dict)
    assert "rmse" in metrics
    assert isinstance(metrics["rmse"], float)
    assert metrics["rmse"] >= 0.0


# --------------------------------------------------------
# 2) CONTRACT CHECKS: Duck Typing & Configurations
# --------------------------------------------------------
def test_raises_if_model_has_no_predict_proba_for_classification(dummy_classification_data):
    """Crash immediately if the loaded classification artifact cannot predict probabilities."""
    _, X, y = dummy_classification_data

    class BadArtifact:
        def predict(self, X):
            return [0] * len(X)

    with pytest.raises(TypeError, match="Fatal: classification model must implement predict_proba"):
        evaluate_model(model=BadArtifact(), X_eval=X,
                       y_eval=y, problem_type="classification")


# --------------------------------------------------------
# 3) FAIL FAST: Empty Data & Shape Guardrails
# --------------------------------------------------------
def test_raises_on_none_or_empty_data(dummy_classification_data):
    """Crash if evaluation receives None or 0 rows of data."""
    model, X, y = dummy_classification_data

    with pytest.raises(ValueError, match="Fatal: X_eval is empty"):
        evaluate_model(model=model, X_eval=None, y_eval=y,
                       problem_type="classification")

    X_empty = X.iloc[0:0]
    with pytest.raises(ValueError, match="Fatal: X_eval is empty"):
        evaluate_model(model=model, X_eval=X_empty, y_eval=y,
                       problem_type="classification")


def test_raises_on_length_mismatch(dummy_classification_data):
    """Crash if the feature rows do not perfectly align with the target labels."""
    model, X, y = dummy_classification_data

    y_short = y.iloc[:-1]

    with pytest.raises(ValueError, match="do not match y_eval rows"):
        evaluate_model(model=model, X_eval=X, y_eval=y_short,
                       problem_type="classification")
