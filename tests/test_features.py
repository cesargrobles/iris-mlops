from sklearn.compose import ColumnTransformer
from src.features import get_feature_preprocessor

IRIS_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def test_returns_column_transformer():
    result = get_feature_preprocessor(numeric_passthrough_cols=IRIS_COLS)
    assert isinstance(result, ColumnTransformer)

def test_has_one_transformer():
    result = get_feature_preprocessor(numeric_passthrough_cols=IRIS_COLS)
    assert len(result.transformers) == 1

def test_includes_iris_columns():
    result = get_feature_preprocessor(numeric_passthrough_cols=IRIS_COLS)
    columns = result.transformers[0][2]
    assert list(columns) == IRIS_COLS