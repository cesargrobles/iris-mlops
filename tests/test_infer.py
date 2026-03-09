import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.infer import run_inference


def make_dummy_model(n_predictions=1):
    """Create a mock model for testing"""
    model = MagicMock()
    predictions = np.array(['setosa'] * n_predictions)
    probas = np.array([[0.9, 0.05, 0.05]] * n_predictions)
    model.predict.return_value = predictions
    model.predict_proba.return_value = probas
    return model


def test_predict_returns_prediction():
    model = make_dummy_model(n_predictions=1)
    X = pd.DataFrame({
        'sepal_length': [5.1],
        'sepal_width': [3.5],
        'petal_length': [1.4],
        'petal_width': [0.2]
    })
    prediction = run_inference(model, X)
    assert prediction is not None


def test_predict_output_length():
    model = make_dummy_model(n_predictions=2)
    X = pd.DataFrame({
        'sepal_length': [5.1, 6.2],
        'sepal_width': [3.5, 3.4],
        'petal_length': [1.4, 5.4],
        'petal_width': [0.2, 2.3]
    })
    prediction = run_inference(model, X)
    assert len(prediction) == 2
    