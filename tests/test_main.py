"""
Unit Tests for main module
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from src.main import main, SETTINGS


class TestMain:
    """Test suite for main function"""

    @pytest.fixture
    def temp_workspace(self):
        temp_dir = tempfile.mkdtemp()
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_dir)
        yield Path(temp_dir)
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)

    def _dummy_df(self, n=10):
        # create a tiny multi-row DataFrame with iris column names for splitting
        data = {
            'sepal_length': range(n),
            'sepal_width': range(n, 2*n),
            'petal_length': range(2*n, 3*n),
            'petal_width': range(3*n, 4*n),
            'species': ['A' if i % 2 == 0 else 'B' for i in range(n)]
        }
        return pd.DataFrame(data)

    def test_main_creates_directories(self, temp_workspace):
        with patch('src.main.load_data') as mock_load, \
             patch('src.main.clean_data') as mock_clean, \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer, \
             patch('src.main.save_model'), \
             patch('src.main.save_csv'):

            df = self._dummy_df(10)
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_model = MagicMock()
            mock_train.return_value = mock_model
            mock_evaluate.return_value = 0.95
            mock_infer.return_value = pd.DataFrame({'prediction': ['A']*10})

            main()

            assert (Path(temp_workspace) / "data" / "raw").exists()
            assert (Path(temp_workspace) / "data" / "processed").exists()
            assert (Path(temp_workspace) / "models").exists()
            assert (Path(temp_workspace) / "reports").exists()

    def test_settings_keys(self):
        required = ['problem_type', 'random_seed', 'test_size', 'target_column', 'data', 'paths', 'features']
        for k in required:
            assert k in SETTINGS

    def test_load_and_clean_called(self):
        with patch('src.main.load_data') as mock_load, \
             patch('src.main.clean_data') as mock_clean, \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer, \
             patch('src.main.save_model'), \
             patch('src.main.save_csv'):

            df = self._dummy_df(10)
            mock_load.return_value = df
            mock_clean.return_value = df
            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.1
            mock_infer.return_value = pd.DataFrame({'prediction': ['A']*10})

            main()
            mock_load.assert_called_once()
            mock_clean.assert_called_once()

    def test_train_evaluate_inference_chain(self):
        with patch('src.main.load_data', return_value=self._dummy_df(10)), \
             patch('src.main.clean_data', return_value=self._dummy_df(10)), \
             patch('src.main.train_model') as mock_train, \
             patch('src.main.evaluate_model') as mock_evaluate, \
             patch('src.main.run_inference') as mock_infer, \
             patch('src.main.save_model'), \
             patch('src.main.save_csv'):

            mock_train.return_value = MagicMock()
            mock_evaluate.return_value = 0.5
            mock_infer.return_value = pd.DataFrame({'prediction': ['A']*10})

            main()
            mock_train.assert_called_once()
            mock_evaluate.assert_called_once()
            mock_infer.assert_called_once()

    def test_main_saves_model_and_predictions(self):
        with patch('src.main.load_data', return_value=self._dummy_df(10)), \
             patch('src.main.clean_data', return_value=self._dummy_df(10)), \
             patch('src.main.train_model', return_value=MagicMock()), \
             patch('src.main.evaluate_model', return_value=0.5), \
             patch('src.main.run_inference', return_value=pd.DataFrame({'prediction':['A']*10})), \
             patch('src.main.save_model') as mock_save_model, \
             patch('src.main.save_csv') as mock_save_csv:

            main()
            assert mock_save_model.called
            assert mock_save_csv.called
