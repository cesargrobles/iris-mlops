"""
Educational Goal:
- Why this module exists in an MLOps system: Validation catches obvious issues early to avoid wasted training runs.
- Responsibility (separation of concerns): Fail fast on empty data and missing columns.
- Pipeline contract (inputs and outputs): df + required_columns -> bool (raises on failure)

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: DataFrame to validate
    - required_columns: columns that must exist
    Outputs:
    - True if valid (raises ValueError otherwise)
    Why this contract matters for reliable ML delivery:
    - Prevents silent failures (training on empty data, missing target, etc.).
    """
    print("[validate.validate_dataframe] Validating dataframe")  # TODO: replace with logging later

    if df is None or len(df) == 0:
        raise ValueError("Validation failed: DataFrame is empty.")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: missing columns: {missing}")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Check for excessive null values (>50% nulls in any column)
    null_thresholds = df.isnull().sum() / len(df)
    excessive_nulls = null_thresholds[null_thresholds > 0.5]
    if not excessive_nulls.empty:
        raise ValueError(f"Validation failed: columns with >50% nulls: {excessive_nulls.to_dict()}")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return True