import numpy as np
import pandas as pd
from src.infer import run_inference


def make_dummy_model():
    """Helper that returns a dummy predictor object with predict method"""
    class Dummy:
        def predict(self, X):
            # simply return zeros
            import numpy as _np
            if isinstance(X, pd.DataFrame):
                length = len(X)
            else:
                length = _np.array(X).shape[0]
            return _np.zeros(length)
    return Dummy()


def test_run_inference_returns_prediction():
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=['a','b','c','d'])
    model = make_dummy_model()
    result = run_inference(model, sample)
    assert 'prediction' in result.columns


def test_run_inference_output_length():
    sample = pd.DataFrame([
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3]
    ], columns=['a','b','c','d'])
    model = make_dummy_model()
    result = run_inference(model, sample)
    assert len(result) == 2
    