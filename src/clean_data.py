"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaningError(RuntimeError):
    """Raised when cleaning cannot be completed safely."""


@dataclass(frozen=True)
class CleanConfig:
    processed_path: Path
    drop_duplicates: bool = True
    dropna: bool = True
    reset_index: bool = True


def _build_clean_config(config: Dict[str, Any]) -> CleanConfig:
    if not isinstance(config, dict):
        raise DataCleaningError("Config must be a dictionary.")

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise DataCleaningError("Missing or invalid config section: 'data'.")

    processed = data_cfg.get("processed")
    if not processed or not isinstance(processed, str):
        raise DataCleaningError("Missing or invalid config key: data.processed (must be a string path).")

    cleaning_cfg = config.get("cleaning", {})
    if cleaning_cfg is None:
        cleaning_cfg = {}
    if not isinstance(cleaning_cfg, dict):
        raise DataCleaningError("Invalid config section: 'cleaning' must be a dict if provided.")

    return CleanConfig(
        processed_path=Path(processed),
        drop_duplicates=bool(cleaning_cfg.get("drop_duplicates", True)),
        dropna=bool(cleaning_cfg.get("dropna", True)),
        reset_index=bool(cleaning_cfg.get("reset_index", True)),
    )


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )
    return out


def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Starting clean_data | input_shape=%s", df.shape)
    """
    Clean and stabilize the raw dataset.

    - Standardizes column names (strip + spaces to underscores)
    - Drops duplicates and missing values (configurable)
    - Resets index (configurable)
    """
    if not isinstance(df, pd.DataFrame):
        raise DataCleaningError("Input must be a pandas DataFrame.")
    if df.shape[0] == 0:
        raise DataCleaningError("Input DataFrame is empty.")

    cc = _build_clean_config(config)

    initial_shape = df.shape
    out = _standardize_columns(df)

    if cc.drop_duplicates:
        out = out.drop_duplicates()

    if cc.dropna:
        out = out.dropna()

    if out.shape[0] == 0:
        raise DataCleaningError("Cleaning removed all rows (empty dataset). Check upstream data quality.")

    if cc.reset_index:
        out = out.reset_index(drop=True)

    logger.info("Cleaned data: initial_shape=%s final_shape=%s", initial_shape, out.shape)
    return out


def save_clean_data(df_clean: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """
    Persist the cleaned dataset to disk as the canonical processed artifact.

    Expects:
      config["data"]["processed"] -> output path for clean CSV
    """
    if not isinstance(df_clean, pd.DataFrame):
        raise DataCleaningError("df_clean must be a pandas DataFrame.")
    if df_clean.shape[0] == 0:
        raise DataCleaningError("df_clean is empty; refusing to save empty artifact.")

    cc = _build_clean_config(config)
    out_path = cc.processed_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_clean.to_csv(out_path, index=False)
    except Exception as exc:
        raise DataCleaningError(f"Failed to save cleaned data to '{out_path}': {exc}") from exc

    logger.info("Saved cleaned data: path=%s shape=%s", out_path, df_clean.shape)
    return out_path