import numpy as np
from src.infer import predict


def test_predict_returns_prediction():
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = predict(sample)
    assert prediction is not None


def test_predict_output_length():
    sample = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3]
    ])
    prediction = predict(sample)
    assert len(prediction) == 2
    