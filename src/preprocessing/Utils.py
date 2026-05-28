import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import pandas as pd

from src.config import SEED
from .constants import TARGET_BIN, TARGET_15C, TEST_SIZE, VAL_SIZE

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

class AllTest(BaseEstimator, TransformerMixin):
    """
    Return the entire dataset as the test set.
    Expects and returns a dict with keys 'train' and 'test'.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> dict:
        print(
            f"  [{self.__class__.__name__}] "
            f"Train: {0:,}  |  Test: {X.shape[0]:,}"
        )
        return {"train": pd.DataFrame(), "test": X}


class TrainTestSplitter(BaseEstimator, TransformerMixin):
    """
    Perform a stratified train/test split.
    Expects a full DataFrame with feature and target columns.
    Returns a dict with keys 'train' and 'test', each a DataFrame
    with all feature columns plus both target columns.
    """

    def __init__(
        self,
        target_bin: str   = TARGET_BIN,
        target_15c: str   = TARGET_15C,
        test_size: float  = TEST_SIZE,
        random_state: int = SEED,
    ):
        self.target_bin   = target_bin
        self.target_15c   = target_15c
        self.test_size    = test_size
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> dict:
        target_cols = [self.target_bin, self.target_15c]
        features    = X.drop(columns=target_cols)
        y_bin       = X[self.target_bin]
        y_15c       = X[self.target_15c]

        (
            X_train, X_test,
            y_train_bin, y_test_bin,
            y_train_15,  y_test_15,
        ) = train_test_split(
            features, y_bin, y_15c,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_15c,
        )

        df_train = X_train.copy()
        df_train[self.target_bin] = y_train_bin.values
        df_train[self.target_15c] = y_train_15.values

        df_test = X_test.copy()
        df_test[self.target_bin]  = y_test_bin.values
        df_test[self.target_15c]  = y_test_15.values

        print(
            f"  [{self.__class__.__name__}] "
            f"Train: {df_train.shape[0]:,}  |  Test: {df_test.shape[0]:,}"
        )
        return {"train": df_train, "test": df_test}

class BlockStratifiedSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, n_blocks, test_size=TEST_SIZE, random_state=SEED, stratify_col=TARGET_15C):
        self.n_blocks = n_blocks
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_col = stratify_col

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        """
        Splits the dataframe into train and test sets using block-based stratification.
        Returns: {"train": df_train, "test": df_test}
        """
        df_len = len(df)
        
        block_size = df_len // self.n_blocks
        block_ids = np.repeat(np.arange(self.n_blocks), block_size)
        
        remainder = df_len - len(block_ids)
        if remainder > 0:
            block_ids = np.append(block_ids, np.full(remainder, self.n_blocks - 1))

        block_labels = np.array([
            df[block_ids == b][self.stratify_col].mode()[0]
            for b in range(self.n_blocks)
        ])

        train_blocks, test_blocks = train_test_split(
            np.arange(self.n_blocks),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=block_labels
        )

        idx_train = np.sort(np.where(np.isin(block_ids, train_blocks))[0])
        idx_test  = np.sort(np.where(np.isin(block_ids, test_blocks))[0])

        return {
            "train": df.iloc[idx_train].copy(),
            "test":  df.iloc[idx_test].copy()
        }

class AnomalyTrainTestSplitter(BaseEstimator, TransformerMixin):
    """
    Train/val/test split for anomaly detection (one-class / semi-supervised) setups.

    Train set:  Normal samples ONLY.
    Val set:    A fraction of remaining Normals + a fraction of attack samples.
    Test set:   The rest of the Normals + the rest of the attack samples.
    """

    def __init__(
        self,
        target_bin: str   = TARGET_BIN,
        target_15c: str   = TARGET_15C,
        test_size: float  = TEST_SIZE,
        val_size: float   = VAL_SIZE,
        random_state: int = SEED,
    ):
        self.target_bin   = target_bin
        self.target_15c   = target_15c
        self.test_size    = test_size
        self.val_size     = val_size
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> dict:
        normal_mask = X[self.target_bin] == 0
        df_normal   = X.loc[normal_mask]
        df_attack   = X.loc[~normal_mask]

        pool_size_ratio = self.test_size + self.val_size
        
        normal_train, normal_pool = train_test_split(
            df_normal,
            test_size=pool_size_ratio,
            random_state=self.random_state,
            shuffle=True,
        )

        test_ratio_in_pool = self.test_size / pool_size_ratio
        normal_val, normal_test = train_test_split(
            normal_pool,
            test_size=test_ratio_in_pool,
            random_state=self.random_state,
            shuffle=True,
        )

        attack_val, attack_test = train_test_split(
            df_attack,
            test_size=test_ratio_in_pool,
            random_state=self.random_state,
            shuffle=True,
        )

        df_train = normal_train.copy().reset_index(drop=True)
        df_val = (
            pd.concat([normal_val, attack_val], axis=0)
              .sample(frac=1.0, random_state=self.random_state)
              .reset_index(drop=True)
        )
        df_test = (
            pd.concat([normal_test, attack_test], axis=0)
              .sample(frac=1.0, random_state=self.random_state)
              .reset_index(drop=True)
        )

        print(
            f"  [{self.__class__.__name__}]\n"
            f"    Train: {df_train.shape[0]:,} (Normal only)\n"
            f"    Val:   {df_val.shape[0]:,} ({normal_val.shape[0]:,} Normal + {attack_val.shape[0]:,} attacks)\n"
            f"    Test:  {df_test.shape[0]:,} ({normal_test.shape[0]:,} Normal + {attack_test.shape[0]:,} attacks)"
        )
        
        return {"train": df_train, "val": df_val, "test": df_test}

class Shuffler(BaseEstimator, TransformerMixin):
    """
    Shuffle both train and test splits.
    Expects and returns the dict produced by TrainTestSplitter.
    """

    def __init__(self, random_state: int = SEED):
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, splits: dict, y=None) -> dict:
        result = {
            split: df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            for split, df in splits.items()
        }
        for split, df in result.items():
            print(f"  [{self.__class__.__name__}] {split}: {df.shape}")
        return result


class SplitExtractor(BaseEstimator, TransformerMixin):
    """
    Extract X_train, X_test, y_train, y_test from the splits dict.
    Accepts a single target column name or a list for multi-output.
    Returns a tuple: (X_train, X_test, y_train, y_test).
    .squeeze() collapses a single target column into a Series automatically.
    """

    def __init__(
        self,
        target_bin: str   = TARGET_BIN,
        target_15c: str   = TARGET_15C,
    ):
        self.target_bin   = target_bin
        self.target_15c   = target_15c

    def fit(self, X, y=None):
        return self

    def transform(self, splits: dict, y=None) -> tuple:
        
        train = splits["train"]
        if train.empty:
            X_train = None
            y_bin_train = None
            y_15c_train = None
        else:
            X_train = train.drop(columns=[self.target_bin, self.target_15c])
            y_bin_train = train[self.target_bin]
            y_15c_train = train[self.target_15c]
            

        test  = splits["test"]
        if test.empty:
            X_test = None
            y_bin_test = None
            y_15c_test = None
        else:
            X_test  = test.drop(columns=[self.target_bin, self.target_15c])
            y_bin_test = test[self.target_bin]
            y_15c_test = test[self.target_15c]
        
        if "val" in splits.keys():
            val  = splits["val"]
            X_val = val.drop(columns=[self.target_bin, self.target_15c])
            y_bin_val = val[self.target_bin]
            y_15c_val = val[self.target_15c]
        else:
            X_val = None
            y_bin_val = None
            y_15c_val = None

        print(
            f"  [{self.__class__.__name__}] "
            f"X_train: {X_train.shape}  |  X_test: {X_test.shape}  |  "
            f"y_bin_train: {y_bin_train.shape} | y_15c_train: {y_15c_train.shape} | "
            f"y_bin_test: {y_bin_test.shape} | y_15c_test: {y_15c_test.shape}"
        )

        if X_val is not None:
            print(
                f"  [{self.__class__.__name__}] "
                f"X_val: {X_val.shape} | y_bin_val: {y_bin_val.shape} | y_15c_val: {y_15c_val.shape}"
            )
            return X_train, X_test, y_bin_train, y_bin_test, y_15c_train, y_15c_test, X_val, y_bin_val, y_15c_val
        else:
            return X_train, X_test, y_bin_train, y_bin_test, y_15c_train, y_15c_test