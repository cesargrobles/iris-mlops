"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize I/O so the rest of the pipeline is easy to test and reuse.
- Responsibility (separation of concerns): Only reading/writing datasets and model artifacts.
- Pipeline contract (inputs and outputs): Paths in -> DataFrames/models out.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to a CSV on disk
    Outputs:
    - df: DataFrame loaded from CSV
    Why this contract matters for reliable ML delivery:
    - A single loader reduces “works on my machine” issues and keeps data reading consistent.
    """
    print(f"[utils.load_csv] Loading CSV: {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Standard read_csv is used here. Adjust if needed:
    # For CSV with semicolon: sep=";"
    # For specific encoding: encoding="utf-8"
    # For date parsing: parse_dates=[...]
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save
    - filepath: destination Path
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Materialized data artifacts make the pipeline debuggable and reproducible.
    """
    print(f"[utils.save_csv] Saving CSV: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Standard to_csv with index=False to avoid row numbers.
    # Adjust if needed:
    # To keep index: index=True
    # For float formatting: float_format="%.4f"
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: any Python object (typically sklearn Pipeline)
    - filepath: destination Path
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Persisting a single Pipeline artifact reduces training/serving skew.
    """
    print(f"[utils.save_model] Saving model: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Standard joblib.dump is used. For compression adjust compress parameter:
    # No compression (default): joblib.dump(model, filepath)
    # With compression: joblib.dump(model, filepath, compress=3)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to joblib artifact
    Outputs:
    - model: loaded object
    Why this contract matters for reliable ML delivery:
    - Loading the same artifact ensures inference uses the exact trained pipeline.
    """
    print(f"[utils.load_model] Loading model: {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Standard joblib.load is sufficient for local environments.
    # For versioning checks or compatibility validation, add checks here if needed.
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return joblib.load(filepath)