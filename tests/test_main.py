# tests/test_main.py
"""
Educational Goal:
- Why these tests exist in an MLOps system: Verify that the orchestrator is mathematically correct (splits)
  and operationally correct (creates artifacts).
- Responsibility: Unit test the split logic, run a safe end-to-end pipeline in an isolated temp directory,
  and enforce separation of concerns.
"""

import copy
from pathlib import Path

import joblib
import pandas as pd
import pytest

import src.main as main_module


# --------------------------------------------------------
# 1) UNIT TESTS: 3-WAY SPLIT MATH
# --------------------------------------------------------
@pytest.fixture
def dummy_split_data():
    """Predictable dataset for split testing (100 rows, imbalanced target)."""
    X = pd.DataFrame({"feature1": range(100), "feature2": range(100)})
    y = pd.Series([0] * 80 + [1] * 20, name="target")
    return X, y


def test_three_way_split_correct_sizes(dummy_split_data):
    """Ensure the math correctly partitions a 100-row dataset."""
    X, y = dummy_split_data

    X_train, X_val, X_test, y_train, y_val, y_test = main_module._three_way_split(
        X, y, test_size=0.10, val_size=0.20, random_state=42, stratify=False
    )

    assert len(X_test) == 10
    assert len(X_val) == 20
    assert len(X_train) == 70


def test_three_way_split_stratification_preserves_ratio(dummy_split_data):
    """Ensure classification splits preserve the 80/20 class imbalance."""
    X, y = dummy_split_data

    _, _, _, _, _, y_test = main_module._three_way_split(
        X, y, test_size=0.10, val_size=0.20, random_state=42, stratify=True
    )

    # With 80/20 overall and 10 test rows, we expect exactly 2 positive labels in the test set
    assert (y_test == 1).sum() == 2


def test_three_way_split_invalid_sizes_raises(dummy_split_data):
    """Crash if configuration asks for impossible split ratios."""
    X, y = dummy_split_data

    with pytest.raises(ValueError, match="split sizes must satisfy"):
        main_module._three_way_split(
            X, y, test_size=0.60, val_size=0.50, random_state=42, stratify=False
        )


# --------------------------------------------------------
# 2) END-TO-END ORCHESTRATION TESTS
# --------------------------------------------------------
def _make_synthetic_iris_df(n_rows: int = 150) -> pd.DataFrame:
    """Build a minimal iris-like dataset matching SETTINGS expectations."""
    import numpy as np
    rng = np.random.RandomState(42)
    species = ["setosa", "versicolor", "virginica"]

    df = pd.DataFrame({
        "sepal_length": rng.uniform(4.3, 7.9, n_rows),
        "sepal_width": rng.uniform(2.0, 4.4, n_rows),
        "petal_length": rng.uniform(1.0, 6.9, n_rows),
        "petal_width": rng.uniform(0.1, 2.5, n_rows),
        "species": [species[i % 3] for i in range(n_rows)],
    })
    return df


def _patch_paths_and_settings(monkeypatch, tmp_path: Path):
    """Redirect all I/O paths to tmp_path. Uses deepcopy on SETTINGS to prevent cross-test state leakage."""
    raw_path = tmp_path / "data" / "raw" / "iris.csv"
    clean_path = tmp_path / "data" / "processed" / "clean.csv"
    model_path = tmp_path / "models" / "model.joblib"
    predictions_path = tmp_path / "reports" / "predictions.csv"

    monkeypatch.setattr(main_module, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(main_module, "CLEAN_DATA_PATH", clean_path)
    monkeypatch.setattr(main_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(main_module, "PREDICTIONS_PATH", predictions_path)

    patched_settings = copy.deepcopy(main_module.SETTINGS)
    patched_settings["is_example_config"] = False
    monkeypatch.setattr(main_module, "SETTINGS", patched_settings)

    return raw_path, clean_path, model_path, predictions_path


def test_main_end_to_end_creates_clean_model_and_predictions_artifacts(tmp_path, monkeypatch):
    """Proves the pipeline cleans data, trains a model, and saves artifacts."""
    raw_path, clean_path, model_path, predictions_path = _patch_paths_and_settings(
        monkeypatch, tmp_path
    )

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_iris_df(n_rows=150).to_csv(raw_path, index=False)

    main_module.main()

    # Cleaned data artifact
    assert clean_path.exists()
    df_clean = pd.read_csv(clean_path)
    assert "species" in df_clean.columns

    # Model artifact
    assert model_path.exists()
    model = joblib.load(model_path)
    assert hasattr(model, "predict")

    # Predictions artifact
    assert predictions_path.exists()
    df_preds = pd.read_csv(predictions_path)
    assert "prediction" in df_preds.columns


def test_main_calls_evaluate_and_infer_modules(tmp_path, monkeypatch):
    """Spy test: intercept calls to ensure the orchestrator delegates work to the correct modules."""
    raw_path, _, _, _ = _patch_paths_and_settings(monkeypatch, tmp_path)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _make_synthetic_iris_df().to_csv(raw_path, index=False)

    called = {"evaluate": 0, "infer": 0}
    original_evaluate = main_module.evaluate_model
    original_infer = main_module.run_inference

    def _spy_evaluate(*args, **kwargs):
        called["evaluate"] += 1
        return original_evaluate(*args, **kwargs)

    def _spy_infer(*args, **kwargs):
        called["infer"] += 1
        return original_infer(*args, **kwargs)

    monkeypatch.setattr(main_module, "evaluate_model", _spy_evaluate)
    monkeypatch.setattr(main_module, "run_inference", _spy_infer)

    main_module.main()

    assert called["evaluate"] == 1
    assert called["infer"] == 1


def test_main_raises_when_raw_data_missing(tmp_path, monkeypatch):
    """Pipeline must crash immediately if raw input data is missing."""
    raw_path, _, _, _ = _patch_paths_and_settings(monkeypatch, tmp_path)
    assert not raw_path.exists()

    with pytest.raises(FileNotFoundError):
        main_module.main()


def test_main_raises_when_example_config_enabled(tmp_path, monkeypatch):
    """Pipeline must crash if the instructor example block is active."""
    _patch_paths_and_settings(monkeypatch, tmp_path)

    patched_settings = copy.deepcopy(main_module.SETTINGS)
    patched_settings["is_example_config"] = True
    monkeypatch.setattr(main_module, "SETTINGS", patched_settings)

    with pytest.raises(ValueError, match="SETTINGS is an example"):
        main_module.main()
