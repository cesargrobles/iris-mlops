"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadingError(RuntimeError):
    """Raised when data cannot be loaded due to config, IO, or parsing errors."""


@dataclass(frozen=True)
class LoadConfig:
    raw_path: Path
    file_type: str = "csv"
    encoding: str = "utf-8"
    sep: str = ","
    na_values: Optional[list[str]] = None


def _build_load_config(config: Dict[str, Any]) -> LoadConfig:
    if not isinstance(config, dict):
        raise DataLoadingError("Config must be a dictionary.")

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise DataLoadingError("Missing or invalid config section: 'data'.")

    raw = data_cfg.get("raw")
    if not raw or not isinstance(raw, str):
        raise DataLoadingError("Missing or invalid config key: data.raw (must be a string path).")

    return LoadConfig(
        raw_path=Path(raw),
        file_type=str(data_cfg.get("file_type", "csv")),
        encoding=str(data_cfg.get("encoding", "utf-8")),
        sep=str(data_cfg.get("sep", ",")),
        na_values=data_cfg.get("na_values", None),
    )


def _validate_file_path(path: Path) -> None:
    if not path.exists():
        raise DataLoadingError(f"Raw dataset not found: {path}")
    if not path.is_file():
        raise DataLoadingError(f"Raw dataset path is not a file: {path}")


def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load raw data according to config.yaml.

    Expects:
      config["data"]["raw"] -> path to CSV file
    """
    lc = _build_load_config(config)
    _validate_file_path(lc.raw_path)

    file_type = lc.file_type.lower().strip()
    if file_type != "csv":
        raise DataLoadingError(f"Unsupported file_type='{lc.file_type}'. Only 'csv' is supported.")

    try:
        df = pd.read_csv(
            lc.raw_path,
            encoding=lc.encoding,
            sep=lc.sep,
            na_values=lc.na_values,
        )
    except Exception as exc:
        raise DataLoadingError(f"CSV parsing failed for '{lc.raw_path}': {exc}") from exc

    if df.shape[0] == 0:
        raise DataLoadingError(f"Loaded dataset is empty (0 rows): {lc.raw_path}")

    logger.info("Loaded raw data: path=%s shape=%s", lc.raw_path, df.shape)
    return df