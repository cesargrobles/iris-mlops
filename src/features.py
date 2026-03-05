"""
Educational Goal:
- Why this module exists in an MLOps system: Feature engineering transforms
  raw columns into the numeric representations a model can learn from.
  Encapsulating this in a ColumnTransformer RECIPE (not fitted yet) prevents
  data leakage because the recipe is fitted ONLY on the training split.
- Responsibility (separation of concerns): Define the feature transformation
  recipe.  Do NOT fit it here — fitting happens inside the training Pipeline.
- Pipeline contract (inputs and outputs):
  Input  -> lists of column names + hyperparameters
  Output -> an unfitted ColumnTransformer object (the recipe)

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols: Numeric columns to discretise into quantile bins.
    - categorical_onehot_cols: Categorical columns to one-hot encode.
    - numeric_passthrough_cols: Numeric columns to pass through unchanged.
    - n_bins: Number of bins for KBinsDiscretizer.
    Outputs:
    - An UNFITTED ColumnTransformer (the feature recipe).
    Why this contract matters for reliable ML delivery:
    - Returning an unfitted transformer means the caller controls WHEN fitting
      happens.  This is the key mechanism that prevents train/test leakage:
      the recipe is fitted only on X_train inside the Pipeline.
    """
    print("[features] Building feature preprocessor recipe (unfitted).")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    # ------------------------------------------------------------------
    # OneHotEncoder compatibility shim (scikit-learn version handling)
    # ------------------------------------------------------------------
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        # Older scikit-learn versions use sparse= instead of sparse_output=
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    # ------------------------------------------------------------------
    # Build the transformer list
    # ------------------------------------------------------------------
    transformers = []

    if quantile_bin_cols:
        transformers.append((
            "quantile_bin",
            KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile"),
            quantile_bin_cols,
        ))

    if categorical_onehot_cols:
        transformers.append((
            "onehot",
            ohe,
            categorical_onehot_cols,
        ))

    if numeric_passthrough_cols:
        transformers.append((
            "passthrough_num",
            "passthrough",
            numeric_passthrough_cols,
        ))

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
   
    # Iris dataset ==> The four columns are numeric - we should use passthrough as they
    #   need no change.

    # Random Forest needs no sclaing, no StandardScaler needed.

    iris_numeric_features = ['sepal_length','sepal_width','petal_length','petal_width']
    
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    print(f"[features] Recipe built with {len(transformers)} transformer group(s).")
    return preprocessor