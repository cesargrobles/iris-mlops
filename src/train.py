"""
Educational Goal:
- Why this module exists in an MLOps system: Training is where the model
  learns from data.  Wrapping the estimator inside a scikit-learn Pipeline
  together with the preprocessor guarantees that the SAME feature transforms
  are applied during training AND inference — eliminating a whole class of
  production bugs.
- Responsibility (separation of concerns): Receive training data and an
  unfitted preprocessor, then return a FITTED Pipeline.  No evaluation, no
  saving — those are separate modules.
- Pipeline contract (inputs and outputs):
  Input  -> X_train, y_train, preprocessor, problem_type
  Output -> a fitted sklearn Pipeline (preprocessor + estimator)

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    problem_type: str,
):
    """
    Inputs:
    - X_train: Feature DataFrame (training split only).
    - y_train: Target Series (training split only).
    - preprocessor: An UNFITTED ColumnTransformer from features.py.
    - problem_type: "regression" or "classification".
    Outputs:
    - A FITTED sklearn Pipeline (preprocessor + estimator).
    Why this contract matters for reliable ML delivery:
    - The Pipeline fits the preprocessor on X_train during .fit() and
      applies the same learned transforms during .predict().  This makes
      the saved model artifact fully self-contained — no separate
      preprocessing code needed at inference time.
    """
    print(f"[train] Training model. Problem type: {problem_type}")  # TODO: replace with logging later
    print(f"[train] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")  # TODO: replace with logging later

    # ------------------------------------------------------------------
    # Baseline estimator selection
    # ------------------------------------------------------------------
    if problem_type == "classification":
        estimator = LogisticRegression(max_iter=500)
    else:
        estimator = Ridge()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
  
    # Using RandomForestClassifier to match the Jupyter notebook
    # n_estimators=100 and random_state=42 kept consistent with notebook parameters
    
    from sklearn.ensemble import RandomForestClassifier
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    model.fit(X_train, y_train)
    print("[train] Model training complete.")
    return model