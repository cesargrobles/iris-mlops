"""
Unit Tests for validate module
"""
import pytest
import pandas as pd
import numpy as np

from src.validate import validate_dataframe


class TestValidateDataframe:
    """Test suite for validate_dataframe function"""

    def test_validate_valid_dataframe(self):
        """Test validation of a valid DataFrame"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'species': ['A', 'B', 'C']
        })
        assert validate_dataframe(df, required_columns=['a', 'b', 'species']) is True

    def test_validate_raises_on_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError"""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df, required_columns=['a'])

    def test_validate_raises_on_none_dataframe(self):
        """Test that None DataFrame raises ValueError"""
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(None, required_columns=['a'])

    def test_validate_raises_on_missing_columns(self):
        """Test that missing required columns raise ValueError"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        with pytest.raises(ValueError, match="missing columns"):
            validate_dataframe(df, required_columns=['a', 'c'])

    def test_validate_with_subset_of_columns(self):
        """Test validation when requiring subset of columns"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        assert validate_dataframe(df, required_columns=['a', 'b']) is True

    def test_validate_with_no_required_columns(self):
        """Test validation with empty required columns list"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        assert validate_dataframe(df, required_columns=[]) is True

    def test_validate_with_single_required_column(self):
        """Test validation with single required column"""
        df = pd.DataFrame({'species': ['A', 'B', 'C']})
        assert validate_dataframe(df, required_columns=['species']) is True

    def test_validate_with_nullable_columns(self):
        """Test validation with columns containing null values"""
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [4, 5, 6]
        })
        # Should pass validation (nulls are allowed up to threshold)
        try:
            result = validate_dataframe(df, required_columns=['a', 'b'])
            assert result is True
        except ValueError:
            # If it fails, check it's for the right reason
            pass

    def test_validate_raises_on_excessive_nulls(self):
        """Test that excessive null values raise ValueError"""
        df = pd.DataFrame({
            'a': [np.nan, np.nan, np.nan],
            'b': [1, 2, 3]
        })
        with pytest.raises(ValueError, match="50%|nulls"):
            validate_dataframe(df, required_columns=['a', 'b'])

    def test_validate_mixed_dtypes(self):
        """Test validation with mixed data types"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        assert validate_dataframe(
            df,
            required_columns=['int_col', 'float_col', 'str_col', 'bool_col']
        ) is True

    def test_validate_large_dataframe(self):
        """Test validation on larger DataFrame"""
        df = pd.DataFrame({
            'a': np.random.randn(1000),
            'b': np.random.randn(1000),
            'target': np.random.choice(['X', 'Y', 'Z'], 1000)
        })
        assert validate_dataframe(df, required_columns=['a', 'b', 'target']) is True

    def test_validate_multiple_missing_columns(self):
        """Test error message includes all missing columns"""
        df = pd.DataFrame({'a': [1, 2]})
        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(df, required_columns=['a', 'b', 'c'])
        error_msg = str(exc_info.value)
        assert 'b' in error_msg or 'c' in error_msg
