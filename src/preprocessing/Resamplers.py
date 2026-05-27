import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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
        if cat_cols:
            raise ValueError(f"  [{self.__class__.__name__}] Categorical columns found: {cat_cols}. SMOTE requires numeric input.")

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


class RandomUndersampler(BaseEstimator, TransformerMixin):
    """
    Randomly undersample the training split to a target size.
    """

    def __init__(
        self,
        n_samples: int | None = None,
        fraction: float | None = 0.1,
        target_15c: str       = TARGET_15C,
        random_state: int     = SEED,
    ):
        self.n_samples    = n_samples
        self.fraction     = fraction
        self.target_15c   = target_15c
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def _resolve_target_size(self, n_rows: int) -> int:
        if self.n_samples is not None:
            return min(int(self.n_samples), n_rows)
        if self.fraction is None:
            raise ValueError(
                f"  [{self.__class__.__name__}] "
                f"Either `n_samples` or `fraction` must be provided."
            )
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(
                f"  [{self.__class__.__name__}] "
                f"`fraction` must be in (0, 1], got {self.fraction}."
            )
        return max(1, int(n_rows * self.fraction))

    def transform(self, splits: dict, y=None) -> dict:
        df_train = splits["train"]
        n_in = len(df_train)

        if n_in == 0:
            print(
                f"  [{self.__class__.__name__}] "
                f"Empty train split — nothing to do."
            )
            return splits

        target_size = self._resolve_target_size(n_in)

        if target_size >= n_in:
            print(
                f"  [{self.__class__.__name__}] "
                f"Target ({target_size:,}) >= train size ({n_in:,}) — passing through."
            )
            return splits

        counts = df_train[self.target_15c].value_counts()

        if len(counts) <= 1:
            # Single-class train (e.g. AnomalyTrainTestSplitter output):
            # plain uniform random subsample.
            df_out = df_train.sample(
                n=target_size, random_state=self.random_state
            ).reset_index(drop=True)
        else:
            # Multi-class: keep per-class proportions, using imblearn's
            # RandomUnderSampler with a proportional sampling strategy.
            proportions = counts / counts.sum()
            sampling_strategy = {
                cls: max(1, int(round(target_size * proportions[cls])))
                for cls in counts.index
            }
            feature_cols = [c for c in df_train.columns if c != self.target_15c]
            rus = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
            )
            X_res, y_res = rus.fit_resample(
                df_train[feature_cols], df_train[self.target_15c]
            )
            df_out = pd.DataFrame(X_res, columns=feature_cols)
            df_out[self.target_15c] = y_res.values
            df_out = df_out.reset_index(drop=True)

        print(
            f"  [{self.__class__.__name__}] "
            f"Train: {n_in:,} -> {len(df_out):,} rows "
            f"({len(df_out) / n_in:.1%} kept)"
        )

        out = dict(splits)
        out["train"] = df_out
        return out


class MaxPresenceUndersampler(BaseEstimator, TransformerMixin):
    """
    Undersample any class whose share of total rows exceeds `max_presence`.
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


class TemporalUndersampler(BaseEstimator, TransformerMixin):
    """
    Undersample the attack class in the training split to a target attack rate.
    """
    def __init__(self, target_atk_rate, random_state=SEED, label_col=TARGET_BIN):
        self.target_atk_rate = target_atk_rate
        self.random_state = random_state
        self.label_col = label_col
        self.rng = np.random.RandomState(self.random_state)

    def fit(self, X, y=None):
        return self

    def transform(self, splits: dict, y=None):
        """
        Expects a dict: {"train": df_train, "test": df_test}
        Returns the same dict with an undersampled "train" set.
        """
        df_train = splits["train"].copy()
        
        y_train = df_train[self.label_col].values
        normal_idx = np.where(y_train == 0)[0]
        attack_idx = np.where(y_train == 1)[0]

        n_normal = len(normal_idx)
        
        n_atk_target = int(n_normal * self.target_atk_rate / (1 - self.target_atk_rate))
        
        n_atk_target = min(n_atk_target, len(attack_idx))

        attack_sub = self.rng.choice(attack_idx, n_atk_target, replace=False)
        balanced_idx = np.sort(np.concatenate([normal_idx, attack_sub]))

        splits["train"] = df_train.iloc[balanced_idx].reset_index(drop=True)
        
        new_rate = splits["train"][self.label_col].mean()
        print(f"Undersampling Complete | New Train Size: {len(splits['train'])} | Attack Rate: {new_rate:.3f}")
        
        return splits


class ClassPresenceUndersampler(BaseEstimator, TransformerMixin):
    """
    Undersampler where the reduction strength scales with class presence.
    """

    def __init__(
        self,
        target_15c: str       = TARGET_15C,
        ratio: float          = 0.5,
        min_presence: float   = 0.01,
        random_state: int     = SEED,
    ):
        self.target_15c   = target_15c
        self.ratio        = ratio
        self.min_presence = min_presence
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def _resample_split(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        total_rows = len(df)
        counts     = df[self.target_15c].value_counts()

        if len(counts) <= 1:
            cls       = counts.index[0]
            new_count = max(1, int(counts.iloc[0] * (1.0 - self.ratio)))
            df_out    = df.sample(n=new_count, random_state=self.random_state)

            print(
                f"  [{self.__class__.__name__}] "
                f"Split: '{split_name}' | Single class ({cls!r}) | "
                f"Old Count: {total_rows:,} | New Count: {new_count:,}"
            )
            return df_out.reset_index(drop=True)

        presences    = counts / total_rows
        max_presence = presences.max()

        sampling_strategy = {}
        for cls, count in counts.items():
            presence = presences[cls]
            if presence < self.min_presence:
                new_count = count
            else:
                reduction = self.ratio * (presence / max_presence)
                new_count = max(1, int(count * (1.0 - reduction)))
            sampling_strategy[cls] = new_count

        print(
            f"  [{self.__class__.__name__}] "
            f"Split: '{split_name}' | Old Count: {counts.sum():,} | "
            f"New Count: {sum(sampling_strategy.values()):,}"
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