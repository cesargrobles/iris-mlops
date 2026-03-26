# src/main.py
"""Pipeline orchestrator for Iris MLOps.

Orchestrates:
1) read raw data
2) clean + validate
3) split
4) train + evaluate
5) save model
6) inference and persist predictions
"""
from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.config import load_config
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.logger import get_logger
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

logger = get_logger(__name__)

# Backward-compatible module-level defaults (used by legacy tests and optional overrides)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "iris.csv"
DEFAULT_CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "predictions.csv"

RAW_DATA_PATH = DEFAULT_RAW_DATA_PATH
CLEAN_DATA_PATH = DEFAULT_CLEAN_DATA_PATH
MODEL_PATH = DEFAULT_MODEL_PATH
PREDICTIONS_PATH = DEFAULT_PREDICTIONS_PATH

SETTINGS = {
    "is_example_config": False,
    "target_column": "species",
    "problem_type": "classification",
    "split": {"test_size": 0.10, "val_size": 0.20, "random_state": 42},
    "features": {
        "quantile_bin": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "categorical_onehot": [],
        "numeric_passthrough": [],
        "binary_sum_cols": [],
        "n_bins": 3,
    },
    "validation": {
        "numeric_non_negative_cols": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    },
}


def _three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
):
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
        raise ValueError("Fatal: split sizes must satisfy 0 < test_size, 0 < val_size, and test_size + val_size < 1")

    stratify_y = y if stratify else None

    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )

        relative_val_size = val_size / (1.0 - test_size)
        stratify_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=stratify_temp,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except ValueError as e:
        logger.warning(f"[main] Stratified split failed: {e} - falling back to random split")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


def _feature_columns(config: dict) -> list:
    features = config["pipeline"]["features"]
    columns = (
        features.get("quantile_bin", [])
        + features.get("categorical_onehot", [])
        + features.get("numeric_passthrough", [])
        + features.get("binary_sum_cols", [])
    )
    return list(dict.fromkeys(columns))


def main(config_path: str = "config.yaml"):
    logger.info("[main] Starting pipeline")

    config = load_config(config_path)

    if SETTINGS.get("is_example_config", False):
        raise ValueError(
            "Fatal: SETTINGS is an example. Update target_column and feature lists for your dataset, then set 'is_example_config': False"
        )

    data_cfg = config.get("data", {})
    pipeline_cfg = config.get("pipeline", {})
    split_cfg = config.get("split", {})
    validation_cfg = config.get("validation", {})

    raw_path = Path(RAW_DATA_PATH if RAW_DATA_PATH != DEFAULT_RAW_DATA_PATH else data_cfg.get("raw", DEFAULT_RAW_DATA_PATH))
    processed_path = Path(CLEAN_DATA_PATH if CLEAN_DATA_PATH != DEFAULT_CLEAN_DATA_PATH else data_cfg.get("processed", DEFAULT_CLEAN_DATA_PATH))
    model_path = Path(MODEL_PATH if MODEL_PATH != DEFAULT_MODEL_PATH else data_cfg.get("models", DEFAULT_MODEL_PATH))
    report_path = Path(PREDICTIONS_PATH if PREDICTIONS_PATH != DEFAULT_PREDICTIONS_PATH else data_cfg.get("reports", DEFAULT_PREDICTIONS_PATH))

    target_column = pipeline_cfg.get("target_column", "species")
    problem_type = pipeline_cfg.get("problem_type", "classification")

    test_size = float(split_cfg.get("test_size", 0.1))
    val_size = float(split_cfg.get("val_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))
    stratify = bool(split_cfg.get("stratify", problem_type == "classification"))

    df_raw = load_raw_data(raw_path)
    df_clean = clean_dataframe(df_raw, target_column=target_column)
    save_csv(df_clean, processed_path)

    required_columns = [target_column] + _feature_columns(config)
    validate_dataframe(
        df=df_clean,
        required_columns=required_columns,
        check_missing_values=validation_cfg.get("check_missing_values", False),
        target_column=target_column,
        target_allowed_values=validation_cfg.get("target_allowed_values"),
        numeric_non_negative_cols=validation_cfg.get("numeric_non_negative_cols", []),
    )

    X_full = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    X_train, X_val, X_test, y_train, y_val, y_test = _three_way_split(
        X_full,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )

    configured_cols = _feature_columns(config)
    missing_features = set(configured_cols) - set(X_train.columns)
    if missing_features:
        raise ValueError(f"Fatal: Missing features {sorted(missing_features)}")

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=pipeline_cfg["features"].get("quantile_bin", []),
        categorical_onehot_cols=pipeline_cfg["features"].get("categorical_onehot", []),
        numeric_passthrough_cols=pipeline_cfg["features"].get("numeric_passthrough", []),
        binary_sum_cols=pipeline_cfg["features"].get("binary_sum_cols", []),
        n_bins=pipeline_cfg["features"].get("n_bins", 3),
    )

    model_pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=problem_type,
        wandb_cfg=config.get("wandb", {}),
    )

    val_metrics = evaluate_model(
        model=model_pipeline,
        X_eval=X_val,
        y_eval=y_val,
        problem_type=problem_type,
        wandb_cfg=config.get("wandb", {}),
    )
    logger.info(f"[main] Validation metrics={val_metrics}")

    save_model(model_pipeline, model_path)

    if config.get("wandb", {}).get("enabled", False):
        try:
            import wandb

            run = wandb.init(project=config["wandb"].get("project", "iris-mlops"), entity=config["wandb"].get("entity"), reinit=True)
            artifact = wandb.Artifact("iris_model", type="model")
            artifact.add_file(str(model_path))
            run.log_artifact(artifact, aliases=["prod"])
            run.finish()
            logger.info("[main] Logged model artifact to W&B with alias 'prod'.")
        except Exception as e:
            logger.warning(f"[main] W&B model logging failed: {e}")

    df_predictions = run_inference(
        model=model_pipeline,
        X_infer=X_test,
        include_proba=(problem_type == "classification"),
    )
    save_csv(df_predictions, report_path)

    logger.info(f"[main] Done. Model={model_path}, predictions={report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Iris MLOps pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
