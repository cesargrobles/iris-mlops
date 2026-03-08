"""
Unit Tests for main module
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import sys

from src.main import main, SETTINGS


class TestMain:
    """Test suite for main function"""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing"""
        temp_dir = tempfile.mkdtemp()
        original_cwd = Path.cwd()

        # Change to temp directory
        import os
        os.chdir(temp_dir)

        yield Path(temp_dir)

        # Restore original directory
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)

    def test_main_creates_directories(self, temp_workspace):
        """Test that main() creates necessary directories"""
        with patch('src.main.load_raw_data') as mock_load, \
             patch('src.main.clean_dataframe') as mock_clean, \
             patch('src.main.validate_dataframe') as mock_validate, \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer:

            # Setup mocks
            df = pd.DataFrame({
                'sepal_length': [5.1],
                'sepal_width': [3.5],
                'petal_length': [1.4],
                'petal_width': [0.2],
                'species': ['setosa']
            })
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['setosa']})

            main()

            # Check directories were created
            assert (Path(temp_workspace) / "data" / "raw").exists()
            assert (Path(temp_workspace) / "data" / "processed").exists()
            assert (Path(temp_workspace) / "models").exists()
            assert (Path(temp_workspace) / "reports").exists()

    def test_settings_has_required_keys(self):
        """Test that SETTINGS contains required configuration keys"""
        required_keys = ['problem_type', 'random_seed', 'test_size', 'target_column', 'paths', 'features']
        for key in required_keys:
            assert key in SETTINGS

    def test_settings_problem_type_valid(self):
        """Test that problem_type is valid"""
        assert SETTINGS['problem_type'] in ['classification', 'regression']

    def test_settings_paths_structure(self):
        """Test that SETTINGS paths are properly structured"""
        paths = SETTINGS['paths']
        required_paths = ['raw_data', 'clean_data', 'model', 'predictions']
        for path_key in required_paths:
            assert path_key in paths
            assert isinstance(paths[path_key], str)

    def test_settings_features_structure(self):
        """Test that SETTINGS features are properly structured"""
        features = SETTINGS['features']
        required_features = ['quantile_bin', 'categorical_onehot', 'numeric_passthrough', 'n_bins']
        for feature_key in required_features:
            assert feature_key in features

    def test_main_calls_load_raw_data(self):
        """Test that main calls load_raw_data"""
        with patch('src.main.load_raw_data') as mock_load, \
             patch('src.main.clean_dataframe') as mock_clean, \
             patch('src.main.validate_dataframe'), \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer:

            df = pd.DataFrame({
                'sepal_length': [5.1],
                'sepal_width': [3.5],
                'petal_length': [1.4],
                'petal_width': [0.2],
                'species': ['setosa']
            })
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['setosa']})

            main()

            mock_load.assert_called_once()

    def test_main_calls_clean_dataframe(self):
        """Test that main calls clean_dataframe"""
        with patch('src.main.load_raw_data') as mock_load, \
             patch('src.main.clean_dataframe') as mock_clean, \
             patch('src.main.validate_dataframe'), \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer:

            df = pd.DataFrame({
                'sepal_length': [5.1],
                'sepal_width': [3.5],
                'petal_length': [1.4],
                'petal_width': [0.2],
                'species': ['setosa']
            })
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['setosa']})

            main()

            mock_clean.assert_called_once()

    def test_main_calls_validate_dataframe(self):
        """Test that main calls validate_dataframe"""
        with patch('src.main.load_raw_data') as mock_load, \
             patch('src.main.clean_dataframe') as mock_clean, \
             patch('src.main.validate_dataframe') as mock_validate, \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer:

            df = pd.DataFrame({
                'sepal_length': [5.1],
                'sepal_width': [3.5],
                'petal_length': [1.4],
                'petal_width': [0.2],
                'species': ['setosa']
            })
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_validate.return_value = True
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['setosa']})

            main()

            mock_validate.assert_called_once()

    def test_main_calls_train_model(self):
        """Test that main calls train_model"""
        with patch('src.main.load_raw_data') as mock_load, \
             patch('src.main.clean_dataframe') as mock_clean, \
             patch('src.main.validate_dataframe'), \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer:

            df = pd.DataFrame({
                'sepal_length': [5.1],
                'sepal_width': [3.5],
                'petal_length': [1.4],
                'petal_width': [0.2],
                'species': ['setosa']
            })
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['setosa']})

            main()

            mock_train.assert_called_once()

    def test_main_calls_evaluate_model(self):
        """Test that main calls evaluate_model"""
        with patch('src.main.load_raw_data') as mock_load, \
             patch('src.main.clean_dataframe') as mock_clean, \
             patch('src.main.validate_dataframe'), \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer:

            df = pd.DataFrame({
                'sepal_length': [5.1],
                'sepal_width': [3.5],
                'petal_length': [1.4],
                'petal_width': [0.2],
                'species': ['setosa']
            })
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['setosa']})

            main()

            mock_evaluate.assert_called_once()

    def test_main_calls_run_inference(self):
        """Test that main calls run_inference"""
        with patch('src.main.load_raw_data') as mock_load, \
             patch('src.main.clean_dataframe') as mock_clean, \
             patch('src.main.validate_dataframe'), \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer:

            df = pd.DataFrame({
                'sepal_length': [5.1],
                'sepal_width': [3.5],
                'petal_length': [1.4],
                'petal_width': [0.2],
                'species': ['setosa']
            })
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['setosa']})

            main()

            mock_infer.assert_called_once()

    def test_main_saves_model_and_predictions(self):
        """Test that main saves model and predictions"""
        with patch('src.main.load_raw_data') as mock_load, \
             patch('src.main.clean_dataframe') as mock_clean, \
             patch('src.main.validate_dataframe'), \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer, \
             patch('src.main.save_model') as mock_save_model, \
             patch('src.main.save_csv') as mock_save_csv:

            df = pd.DataFrame({
                'sepal_length': [5.1],
                'sepal_width': [3.5],
                'petal_length': [1.4],
                'petal_width': [0.2],
                'species': ['setosa']
            })
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['setosa']})

            main()

            # Should save model and predictions
            assert mock_save_model.called
            assert mock_save_csv.called
