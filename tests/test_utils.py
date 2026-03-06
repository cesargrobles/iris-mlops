"""
Unit Tests for utils module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from sklearn.ensemble import RandomForestClassifier

from src.utils import load_csv, save_csv, save_model, load_model


class TestLoadCSV:
    """Test suite for load_csv function"""

    @pytest.fixture
    def temp_csv(self):
        """Create a temporary CSV file"""
        temp_dir = tempfile.mkdtemp()
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4.0, 5.0, 6.0],
            'c': ['x', 'y', 'z']
        })
        filepath = Path(temp_dir) / "test.csv"
        df.to_csv(filepath, index=False)
        yield filepath
        shutil.rmtree(temp_dir)

    def test_load_csv_returns_dataframe(self, temp_csv):
        """Test that load_csv returns a DataFrame"""
        result = load_csv(temp_csv)
        assert isinstance(result, pd.DataFrame)

    def test_load_csv_correct_data(self, temp_csv):
        """Test that loaded data is correct"""
        result = load_csv(temp_csv)
        assert result.shape == (3, 3)
        assert list(result.columns) == ['a', 'b', 'c']

    def test_load_csv_dtypes(self, temp_csv):
        """Test that loaded data has correct dtypes"""
        result = load_csv(temp_csv)
        assert pd.api.types.is_integer_dtype(result['a'])
        assert pd.api.types.is_float_dtype(result['b'])
        assert pd.api.types.is_object_dtype(result['c'])

    def test_load_csv_nonexistent_file(self):
        """Test error on nonexistent file"""
        with pytest.raises(FileNotFoundError):
            load_csv(Path("/nonexistent/path/file.csv"))


class TestSaveCSV:
    """Test suite for save_csv function"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_save_csv_creates_file(self, temp_dir):
        """Test that save_csv creates a file"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        filepath = temp_dir / "output.csv"
        save_csv(df, filepath)
        assert filepath.exists()

    def test_save_csv_correct_content(self, temp_dir):
        """Test that saved content is correct"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        filepath = temp_dir / "output.csv"
        save_csv(df, filepath)

        loaded = pd.read_csv(filepath)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_csv_creates_nested_dirs(self, temp_dir):
        """Test that save_csv creates nested directories"""
        df = pd.DataFrame({'a': [1, 2]})
        filepath = temp_dir / "nested" / "deep" / "output.csv"
        save_csv(df, filepath)
        assert filepath.exists()

    def test_save_csv_overwrites_existing(self, temp_dir):
        """Test that save_csv overwrites existing file"""
        filepath = temp_dir / "data.csv"

        df1 = pd.DataFrame({'a': [1, 2]})
        save_csv(df1, filepath)

        df2 = pd.DataFrame({'b': [3, 4, 5]})
        save_csv(df2, filepath)

        loaded = pd.read_csv(filepath)
        assert list(loaded.columns) == ['b']
        assert len(loaded) == 3

    def test_save_csv_with_nan(self, temp_dir):
        """Test saving DataFrame with NaN values"""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4, 5, 6]})
        filepath = temp_dir / "with_nan.csv"
        save_csv(df, filepath)

        loaded = pd.read_csv(filepath)
        assert loaded['a'].isna().sum() == 1


class TestSaveModel:
    """Test suite for save_model function"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_model(self):
        """Create a sample sklearn model"""
        return RandomForestClassifier(n_estimators=10, random_state=42)

    def test_save_model_creates_file(self, temp_dir, sample_model):
        """Test that save_model creates a file"""
        filepath = temp_dir / "model.joblib"
        save_model(sample_model, filepath)
        assert filepath.exists()

    def test_save_model_creates_nested_dirs(self, temp_dir, sample_model):
        """Test that save_model creates nested directories"""
        filepath = temp_dir / "models" / "v1" / "model.joblib"
        save_model(sample_model, filepath)
        assert filepath.exists()

    def test_save_model_file_size(self, temp_dir, sample_model):
        """Test that saved model has non-zero size"""
        filepath = temp_dir / "model.joblib"
        save_model(sample_model, filepath)
        assert filepath.stat().st_size > 0

    def test_save_model_overwrites_existing(self, temp_dir, sample_model):
        """Test that save_model overwrites existing file"""
        filepath = temp_dir / "model.joblib"
        save_model(sample_model, filepath)
        original_size = filepath.stat().st_size

        save_model(sample_model, filepath)
        assert filepath.stat().st_size > 0


class TestLoadModel:
    """Test suite for load_model function"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def saved_model(self, temp_dir):
        """Create and save a model"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        filepath = temp_dir / "model.joblib"
        save_model(model, filepath)
        return filepath

    def test_load_model_returns_object(self, saved_model):
        """Test that load_model returns an object"""
        result = load_model(saved_model)
        assert result is not None

    def test_load_model_is_same_type(self, saved_model):
        """Test that loaded model has correct type"""
        result = load_model(saved_model)
        assert isinstance(result, RandomForestClassifier)

    def test_load_model_nonexistent_file(self, temp_dir):
        """Test error on nonexistent model file"""
        with pytest.raises(FileNotFoundError):
            load_model(temp_dir / "nonexistent.joblib")

    def test_save_and_load_consistency(self, temp_dir):
        """Test that saved model can be loaded"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        filepath = temp_dir / "model.joblib"

        save_model(model, filepath)
        loaded = load_model(filepath)

        # Both should be RandomForestClassifier instances
        assert type(loaded) == type(model)
        assert hasattr(loaded, 'predict')
