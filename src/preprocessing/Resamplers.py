import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from src.config import SEED
from .constants import TARGET_BIN, TARGET_15C, NORMAL_LABEL, SMOTE_RATIO, SMOTE_K, MAX_PRESENCE

# ---------------------------------------------------------
# Resamplers
# ---------------------------------------------------------

class SMOTEResampler(BaseEstimator, TransformerMixin):
    """
    Apply SMOTE to the training split only.
    Expects the dict produced by TrainTestSplitter (keys: 'train', 'test').
    Ordinal-encodes categorical columns before SMOTE and decodes afterwards.
    Returns the same dict with the resampled 'train' DataFrame.
    """

    def __init__(
        self,
        target_bin: str   = TARGET_BIN,
        target_15c: str   = TARGET_15C,
        normal_label: str = NORMAL_LABEL,
        ratio: float      = SMOTE_RATIO,
        k_neighbors: int  = SMOTE_K,
        random_state: int = SEED,
    ):
        self.target_bin   = target_bin
        self.target_15c   = target_15c
        self.normal_label = normal_label
        self.ratio        = ratio
        self.k_neighbors  = k_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, splits: dict, y=None) -> dict:
        df_train = splits["train"].copy()
        df_test  = splits["test"]

        target_cols = [self.target_bin, self.target_15c]
        X_train     = df_train.drop(columns=target_cols)
        y_train_15  = df_train[self.target_15c]

        # Ordinal-encode categorical columns (SMOTE requires numeric input)
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        encoder  = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if cat_cols:
            X_train = X_train.copy()
            X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
            print(f"  [{self.__class__.__name__}] Encoded {len(cat_cols)} categorical columns.")

        # Build per-class oversampling strategy
        counts       = Counter(y_train_15)
        majority_cls = max(counts, key=counts.get)
        target_size  = int(counts[majority_cls] * self.ratio)
        strategy = {
            cls: target_size
            for cls, cnt in counts.items()
            if cls != majority_cls and cnt < target_size
        }
        print(
            f"  [{self.__class__.__name__}] "
            f"Majority: '{majority_cls}' ({counts[majority_cls]:,})  "
            f"→ target minority size: {target_size:,}"
        )

        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )
        X_res, y_15_res = smote.fit_resample(X_train, y_train_15)

        # Rebuild DataFrame and decode categorical columns back to text
        df_res = pd.DataFrame(X_res, columns=X_train.columns)
        if cat_cols:
            df_res[cat_cols] = encoder.inverse_transform(df_res[cat_cols])

        # Synchronise binary label from 15-class label
        df_res[self.target_bin] = (y_15_res != self.normal_label).astype(int)
        df_res[self.target_15c] = y_15_res

        print(
            f"  [{self.__class__.__name__}] "
            f"Resampled train: {df_res.shape[0]:,}"
        )
        return {"train": df_res, "test": df_test}


class MaxPresenceUndersampler(BaseEstimator, TransformerMixin):
    """
    Undersample any class whose share of total rows exceeds `max_presence`.
    Classes already below the threshold are left untouched.

    Expects the dict produced by TrainTestSplitter (keys: 'train', 'test').
    Applies independently to each split so no data leakage occurs.
    Applies to the 15-class target so every dominant class is capped,
    regardless of whether it is an attack type or Normal traffic.
    """

    def __init__(
        self,
        target_15c: str     = TARGET_15C,
        max_presence: float = MAX_PRESENCE,
        random_state: int   = SEED,
    ):
        self.target_15c   = target_15c
        self.max_presence = max_presence
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def _resample_split(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        total_rows = len(df)
        counts     = df[self.target_15c].value_counts()

        sampling_strategy = {}
        for attack_type, count in counts.items():
            current_presence = count / total_rows
            if current_presence > self.max_presence:
                new_count = int(total_rows * self.max_presence)
            else:
                new_count = count
            sampling_strategy[attack_type] = new_count

        print(
            f"  [{self.__class__.__name__}] "
            f"Split: '{split_name}' | Old Count: {counts.sum():,} | New Count: {sum(sampling_strategy.values()):,}"
        )

        feature_cols = [c for c in df.columns if c != self.target_15c]
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
        )
        X_resampled, y_resampled = rus.fit_resample(df[feature_cols], df[self.target_15c])

        df_out = pd.DataFrame(X_resampled, columns=feature_cols)
        df_out[self.target_15c] = y_resampled.values
        return df_out.reset_index(drop=True)

    def transform(self, splits: dict, y=None) -> dict:
        return {
            split_name: self._resample_split(df, split_name)
            for split_name, df in splits.items()
        }