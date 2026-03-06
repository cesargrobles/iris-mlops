"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: to measure how well
  the trained model performs on unseen data, in a single reusable place.
- Responsibility: receive a trained pipeline and test data, compute
  and print the right metric based on problem type.
- Pipeline contract:
    Input:  model (fitted sklearn Pipeline), X_test (DataFrame),
            y_test (Series), problem_type (str)
    Output: a single float (F1 score for classification, RMSE for regression)

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable will be imported from config.yml later
"""

# --- Imports ---
# pandas: to work with DataFrames
# sklearn metrics: to compute F1 and RMSE
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str
) -> float:
    """
    Inputs:
    - model: fitted sklearn Pipeline (preprocessor + estimator)
    - X_test: DataFrame with feature columns (never seen during training)
    - y_test: Series with true target labels
    - problem_type: "classification" or "regression"

    Outputs:
    - a single float: F1 score (classification) or RMSE (regression)

    Why this contract matters for reliable ML delivery:
    - Using held-out test data gives an honest measure of real performance
    - Returning a single float makes it easy to compare runs and log metrics
    - Supporting both problem types makes the module reusable across projects
    """

    print("[ evaluate.py ] Starting model evaluation...")  # TODO: replace with logging later

    # --- Fail fast: crash immediately if test data is empty ---
    # Why: empty test sets produce misleading perfect scores silently
    if X_test.empty or y_test.empty:
        raise ValueError("X_test or y_test is empty. Cannot evaluate on empty data.")

    # --- Generate predictions ---
    # Why: we call predict on the full Pipeline so preprocessing is applied
    # exactly the same way as during training — no leakage, no drift
    y_pred = model.predict(X_test)
    print(f"[ evaluate.py ] Predictions generated for {len(y_pred)} samples.")  # TODO: replace with logging later

    # --- Compute metric based on problem type ---
    # Why: routing by string keeps this module flexible and config-driven
    if problem_type == "classification":
        # weighted F1 works for binary AND multiclass (e.g. iris has 3 classes)
        score = f1_score(y_test, y_pred, average="weighted")
        print(f"[ evaluate.py ] F1 Score (weighted): {score:.4f}")  # TODO: replace with logging later

    elif problem_type == "regression":
        # RMSE: same units as the target, easy to interpret
        score = mean_squared_error(y_test, y_pred, squared=False)
        print(f"[ evaluate.py ] RMSE: {score:.4f}")  # TODO: replace with logging later

    else:
        raise ValueError(
            f"Unknown problem_type: '{problem_type}'. "
            "Use 'classification' or 'regression'."
        )

    return score