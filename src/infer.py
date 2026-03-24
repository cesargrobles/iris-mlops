"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""
import pandas as pd


def run_inference(model, X: pd.DataFrame):
    """
    Run predictions using a trained model.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    predictions = model.predict(X)

    results = pd.DataFrame({
        "prediction": predictions
    }, index=X.index)

    return results