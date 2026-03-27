# src/train.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Encapsulate training so models are reproducible and swappable
  without rewiring the pipeline.
- Responsibility (separation of concerns): Combines the feature recipe and algorithm into a single Pipeline artifact.
- Pipeline contract: Inputs are the train split, problem type, and preprocessor. Output is a fully fitted Pipeline artifact.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Optional

import pandas as pd
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.logger import get_logger

try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__)


def _normalize_problem_type(problem_type: Optional[str]) -> str:
    """
    Inputs:
    - problem_type: Raw problem type string
    Outputs:
    - normalized: "classification" or "regression"

    Why this contract matters for reliable ML delivery:
    - Strict normalization avoids silent configuration errors and makes failures actionable
    """
    return (problem_type or "").strip().lower()


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
    wandb_cfg: dict | None = None,
) -> Pipeline:
    """
    Inputs:
    - X_train: Training features (already split, no target column)
    - y_train: Training target
    - preprocessor: ColumnTransformer recipe (should not be fitted here)
    - problem_type: "classification" or "regression"
    Outputs:
    - pipeline: Trained scikit-learn Pipeline object

    Why this contract matters for reliable ML delivery:
    - Fitting happens on training data only, preventing leakage and inflated performance estimates
    - A single fitted pipeline artifact ensures training and inference run the exact same steps
    """
    logger.info(f"[train.train_model] Training model pipeline for problem_type={problem_type}")

    # 1) Fail-fast structural guardrails
    if X_train is None or len(X_train) == 0:
        raise ValueError("Fatal: X_train is empty. Cannot train a model.")

    if y_train is None or len(y_train) == 0:
        raise ValueError("Fatal: y_train is empty. Cannot train a model.")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Fatal: X_train rows ({len(X_train)}) do not match y_train rows ({len(y_train)})."
        )

    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"Fatal: preprocessor must be a ColumnTransformer. Got type={type(preprocessor)}"
        )

    # 2) Model selection
    pt = _normalize_problem_type(problem_type)

    if pt == "classification":
        model = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",
        )
    elif pt == "regression":
        model = LinearRegression()
    else:
        raise ValueError(
            f"Fatal: Unsupported problem_type '{problem_type}'. Use 'classification' or 'regression'."
        )

    # 3) Build the deployable artifact
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # 4) Execute training
    pipeline.fit(X_train, y_train)

    if (
    wandb_cfg
    and wandb_cfg.get("enabled", False)
    and wandb is not None
    and hasattr(wandb, "init")
    ):
        run = wandb.init(
        project=wandb_cfg.get("project", "iris-mlops"),
        entity=wandb_cfg.get("entity"),
        reinit=True,
    )
        run.config.update(wandb_cfg)
        wandb.log({"artifact/accuracy": 0})  # placeholder: update from evaluation step
        run.finish()

    return pipeline
