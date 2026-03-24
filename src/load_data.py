"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""
from __future__ import annotations
import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class DataLoadingError(Exception):
    pass


def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Starting load_data with config: %s", config)
    """
    Load dataset dynamically based on config.
    """

    if "data" not in config:
        raise DataLoadingError("Missing 'data' section in config.")

    data_cfg = config["data"]
    source = data_cfg.get("source")

    if source is None and data_cfg.get("raw"):
        source = "csv"

    if source == "seaborn":
        dataset_name = data_cfg.get("dataset_name")
        if not dataset_name:
            raise DataLoadingError("Missing 'dataset_name' for seaborn source.")

        try:
            df = sns.load_dataset(dataset_name)
        except Exception as e:
            raise DataLoadingError(f"Failed to load seaborn dataset '{dataset_name}': {e}")

    elif source == "csv":
        raw_path = data_cfg.get("raw")
        if not raw_path:
            raise DataLoadingError("Missing 'raw' path for csv source.")

        path = Path(raw_path)
        if not path.exists():
            raise DataLoadingError(f"CSV file not found at {raw_path}")

        df = pd.read_csv(path)

    else:
        raise DataLoadingError("Unsupported data source. Use 'csv' or 'seaborn'.")

    if df.empty:
        raise DataLoadingError("Loaded dataset is empty.")

    logger.info(f"Loaded dataset from {source}, shape={df.shape}")
    return df