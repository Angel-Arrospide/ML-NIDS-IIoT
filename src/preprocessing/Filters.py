import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .constants import TARGET_BIN, TARGET_15C, VALID_TIME_COL

# ---------------------------------------------------------
# Filters
# ---------------------------------------------------------

class ValidTimeFilter(BaseEstimator, TransformerMixin):
    """
    Drop rows where `col` == 0.
    Expects and returns a DataFrame (X contains all columns including targets).
    """

    def __init__(self, col: str = VALID_TIME_COL):
        self.col = col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.col not in X.columns:
            print(f"  WARNING: column '{self.col}' not found – skipping filter.")
            return X
        n_before = len(X)
        X = X[X[self.col] != 0].reset_index(drop=True)
        print(f"  [{self.__class__.__name__}] Dropped {n_before - len(X):,} rows  →  {X.shape}")
        return X


class CategoricalDropper(BaseEstimator, TransformerMixin):
    """
    Drop all columns with dtype object or category, except the target columns
    which must be preserved for downstream splitting and labelling.
    Expects and returns a DataFrame.
    """

    def __init__(self, keep_cols: list = None):
        self.keep_cols = keep_cols or [TARGET_BIN, TARGET_15C]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        cat_cols = [
            c for c in X.select_dtypes(include=["object", "category", "str"]).columns
            if c not in self.keep_cols
        ]
        X = X.drop(columns=cat_cols)
        print(
            f"  [{self.__class__.__name__}] "
            f"Dropped {len(cat_cols)} categorical columns: {cat_cols}  →  {X.shape}"
        )
        return X