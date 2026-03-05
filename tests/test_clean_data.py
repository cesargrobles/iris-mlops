from pathlib import Path

import pandas as pd
import pytest

from src.clean_data import DataCleaningError, clean_data, save_clean_data


def test_clean_data_standardizes_columns(tmp_path: Path):
    df = pd.DataFrame({"rx ds": [10, 20], "OD": [0, 1]})
    cfg = {"data": {"processed": str(tmp_path / "clean.csv")}}
    out = clean_data(df, cfg)
    assert "rx_ds" in out.columns


def test_clean_data_raises_on_empty_input(tmp_path: Path):
    df = pd.DataFrame()
    cfg = {"data": {"processed": str(tmp_path / "clean.csv")}}
    with pytest.raises(DataCleaningError):
        clean_data(df, cfg)


def test_save_clean_data_writes_file(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out_path = tmp_path / "processed" / "clean.csv"
    cfg = {"data": {"processed": str(out_path)}}

    saved = save_clean_data(df, cfg)
    assert saved.exists()
    assert saved.is_file()