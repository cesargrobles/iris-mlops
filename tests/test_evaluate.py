"""
Tests for src/evaluate.py
Covers: normal classification, empty data, invalid problem_type
"""

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.evaluate import evaluate_model


# --- Helper: creates a simple trained pipeline for testing ---
# Why: we need a real fitted model to pass to evaluate_model
def make_trained_pipeline():
    """Creates and fits a minimal sklearn Pipeline on iris-like data."""
    X = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.2, 5.8, 6.3],
        "sepal_width":  [3.5, 3.0, 2.9, 2.7, 3.3],
    })
    y = pd.Series(["setosa", "setosa", "virginica", "virginica", "virginica"])

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, solver="liblinear"))
    ])
    pipeline.fit(X, y)
    return pipeline, X, y


# --- Test 1: normal classification case ---
def test_evaluate_classification_returns_valid_score():
    """
    Should return a float between 0 and 1
    when given a valid trained model and test data.
    """
    model, X, y = make_trained_pipeline()
    score = evaluate_model(model, X, y, problem_type="classification")

    assert isinstance(score, float), "Score should be a float"
    assert 0.0 <= score <= 1.0, "F1 score should be between 0 and 1"


# --- Test 2: empty DataFrame ---
def test_evaluate_raises_on_empty_dataframe():
    """
    Should raise ValueError immediately
    when X_test or y_test is empty.
    """
    model, _, _ = make_trained_pipeline()
    X_empty = pd.DataFrame()
    y_empty = pd.Series([], dtype=str)

    with pytest.raises(ValueError):
        evaluate_model(model, X_empty, y_empty, problem_type="classification")


# --- Test 3: invalid problem_type ---
def test_evaluate_raises_on_invalid_problem_type():
    """
    Should raise ValueError when problem_type
    is not 'classification' or 'regression'.
    """
    model, X, y = make_trained_pipeline()

    with pytest.raises(ValueError):
        evaluate_model(model, X, y, problem_type="clustering")