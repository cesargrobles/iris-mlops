import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.features import get_feature_preprocessor
from src.train import train_model

IRIS_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_train = pd.DataFrame({
    "sepal_length": np.random.uniform(4.3, 7.9, 10),
    "sepal_width":  np.random.uniform(2.0, 4.4, 10),
    "petal_length": np.random.uniform(1.0, 6.9, 10),
    "petal_width":  np.random.uniform(0.1, 2.5, 10),
})

y_train = pd.Series(np.random.choice(["setosa", "versicolor", "virginica"], 10))


def test_train_model_returns_pipeline():
    preprocessor = get_feature_preprocessor(numeric_passthrough_cols=IRIS_COLS)
    model = train_model(X_train, y_train, preprocessor, "classification")
    assert isinstance(model, Pipeline)


def test_train_model_uses_random_forest():
    preprocessor = get_feature_preprocessor(numeric_passthrough_cols=IRIS_COLS)
    model = train_model(X_train, y_train, preprocessor, "classification")
    assert isinstance(model.named_steps["model"], RandomForestClassifier)