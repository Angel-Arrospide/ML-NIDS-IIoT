import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Scalers
# ---------------------------------------------------------

class FeatureScaler:
    """
    Fit StandardScaler on X_train and apply to both X_train and X_test.
    Expects a tuple: (X_train, X_test, y_train, y_test).
    Excludes any remaining categorical columns from scaling.
    """

    def __init__(self):
        self.scaler_   = None
        self.num_cols_ = None

    def fit_transform(self, splits: tuple) -> tuple:
        X_train, X_test, y_train, y_test = splits

        X_train = X_train.copy()
        X_test  = X_test.copy()

        self.num_cols_ = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
        self.scaler_   = StandardScaler()

        X_train[self.num_cols_] = self.scaler_.fit_transform(X_train[self.num_cols_])
        X_test[self.num_cols_]  = self.scaler_.transform(X_test[self.num_cols_])

        print(
            f"  [{self.__class__.__name__}] "
            f"Scaled {len(self.num_cols_)} numeric columns  |  "
            f"X_train: {X_train.shape}  |  X_test: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    def transform(self, splits: tuple) -> tuple:
        """Apply already-fitted scaler — use on new data at inference time."""
        X, y = splits if len(splits) == 2 else (splits[0], None)
        X = X.copy()
        X[self.num_cols_] = self.scaler_.transform(X[self.num_cols_])
        return X