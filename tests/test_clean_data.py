# tests/test_clean_data.py
"""
Educational Goal:
- Verify that the stateless data cleaning rules work as intended.
- Keep tests isolated from file I/O to test only the cleaning logic.
"""

import pandas as pd
from src.clean_data import clean_dataframe

TARGET_COLUMN = "species"


def test_clean_dataframe_happy_path_contract():
    """Verify column standardization, deduplication, target NaN handling, and index reset."""
    df_messy = pd.DataFrame(
        {
            "sepal length": [5.1, 4.9, 6.2, 5.1],
            "sepal width": [3.5, 3.0, 2.9, 3.5],
            "species": ["setosa", "versicolor", pd.NA, "setosa"],
        }
    )

    df_clean = clean_dataframe(df_messy, target_column=TARGET_COLUMN)

    # Column names standardized (spaces to underscores)
    assert "sepal_length" in df_clean.columns
    assert "sepal length" not in df_clean.columns

    # Target column preserved without NaNs
    assert TARGET_COLUMN in df_clean.columns
    assert df_clean[TARGET_COLUMN].isna().sum() == 0

    # Index reset
    assert list(df_clean.index) == list(range(len(df_clean)))

    # Duplicates and missing-target rows removed (row 3 is dup of row 0, row 2 has NaN target)
    assert len(df_clean) == 2


def test_clean_dataframe_inference_mode_does_not_require_target():
    """In inference mode (target_column=None), no target column is needed."""
    df_infer = pd.DataFrame(
        {
            "sepal length": [5.1, 4.9],
            "sepal width": [3.5, 3.0],
        }
    )

    df_clean = clean_dataframe(df_infer, target_column=None)

    assert "sepal_length" in df_clean.columns
    assert "species" not in df_clean.columns
    assert len(df_clean) == 2
