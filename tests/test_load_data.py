from pathlib import Path

import pandas as pd
import pytest

from src.load_data import DataLoadingError, load_data


def test_load_data_raises_on_missing_file(tmp_path: Path):
    cfg = {"data": {"raw": str(tmp_path / "missing.csv")}}
    with pytest.raises(DataLoadingError):
        load_data(cfg)


def test_load_data_loads_csv(tmp_path: Path):
    p = tmp_path / "dataset.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    cfg = {"data": {"raw": str(p), "file_type": "csv"}}

    df = load_data(cfg)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)