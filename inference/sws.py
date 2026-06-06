import time
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import Filters

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCALER_PATH = Path("/home/aam/ML-NIDS-IIoT/inference/data/SWS_num_scaler.pkl")
BASELINE_PATH = Path("/home/aam/ML-NIDS-IIoT/inference/data/SWS_num_baseline.joblib")
INPUT_PKL = Path("/home/aam/ML-NIDS-IIoT/inference/data/DL-EdgeIIoT-dataset_clean.pkl")
OUTPUT_CSV = None # None = auto-name as <input>_scored.csv

BATCH_WINDOWS = 64
THRESHOLD = 0.5946188561810029 # None = derived at runtime from the 95th-pct of Normal scores
LABEL_COLS = ["Attack_type", "Attack_label"]

# ---------------------------------------------------------------------------
# Best config
# ---------------------------------------------------------------------------
BEST_W        = 520
BEST_STEP     = 520        # 'full' == W
BEST_AGG      = "mean"
BEST_NORM     = "L2"
BEST_BASELINE = "global"

SEED     = 42
N_BLOCKS = 10_000

TARGET_BIN = "Attack_label"
TARGET_15C = "Attack_type"

# ---------------------------------------------------------------------------
# SWS funcs
# ---------------------------------------------------------------------------

def make_windows(X: np.ndarray, W: int, step: int) -> np.ndarray:
    n = (len(X) - W) // step + 1
    idx = np.arange(W)[None, :] + step * np.arange(n)[:, None]
    return X[idx]


def window_labels(y: np.ndarray, W: int, step: int) -> np.ndarray:
    n = (len(y) - W) // step + 1
    idx = np.arange(W)[None, :] + step * np.arange(n)[:, None]
    return (y[idx].sum(axis=1) > 0).astype(np.uint8)


def estimate_baseline(X_normal: np.ndarray, method: str):
    if method == "global":
        mean = X_normal.mean(axis=0)
        std  = X_normal.std(axis=0) + 1e-9
    elif method == "percentile90":
        lo   = np.percentile(X_normal, 5, axis=0)
        hi   = np.percentile(X_normal, 95, axis=0)
        mask = ((X_normal >= lo) & (X_normal <= hi)).all(axis=1)
        clipped = X_normal[mask] if mask.sum() > 100 else X_normal
        mean = clipped.mean(axis=0)
        std  = clipped.std(axis=0) + 1e-9
    else:
        raise ValueError(method)
    return mean.astype(np.float32), std.astype(np.float32)


def score_batch(
    batch: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    agg_fn: str,
    norm: str,
) -> np.ndarray:
    if agg_fn == "mean":
        summary = batch.mean(axis=1)
    elif agg_fn == "max":
        summary = batch.max(axis=1)
    elif agg_fn == "median":
        summary = np.median(batch, axis=1)
    else:
        raise ValueError(agg_fn)

    z = (summary - mean) / std

    if norm == "L2":
        return np.linalg.norm(z, axis=1)
    if norm == "L1":
        return np.abs(z).sum(axis=1)
    if norm == "Linf":
        return np.abs(z).max(axis=1)
    raise ValueError(norm)


# ---------------------------------------------------------------------------
# Stratified-block split (all test)
# ---------------------------------------------------------------------------
def stratified_block_split(
    X: np.ndarray, y_bin: np.ndarray, y_15c: np.ndarray, n_blocks: int = N_BLOCKS
):
    idx_all = np.arange(len(X))
    return idx_all, idx_all


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
_COLS_TO_DROP = ["frame.time.delta", "frame.time.order"]


def clean_batch(df: pd.DataFrame) -> pd.DataFrame:
    drop = [
        *_COLS_TO_DROP,
        *[c for c in df.columns
          if c.startswith("ip.src_category") or c.startswith("ip.dst_category")],
    ]
    df = df.drop(columns=drop, errors="ignore")
    df = Filters.ValidTimeFilter().transform(df)
    df = Filters.CategoricalDropper(keep_cols=[]).transform(df)
    return df


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------
class SWSInference:

    def __init__(self):
        print(f"[init] Loading scaler   -> {SCALER_PATH}")
        self.scaler    = joblib.load(SCALER_PATH)
        if hasattr(self.scaler, "feature_names_in_"):
            self.feat_cols = list(self.scaler.feature_names_in_)
            print(f"[init] Scaler features  : {len(self.feat_cols)}")
        else:
            self.feat_cols = None
            print(f"[init] Scaler features  : None")
        

        print(f"[init] Loading baseline -> {BASELINE_PATH}")
        baseline       = joblib.load(BASELINE_PATH)
        self.mean      = baseline["mean"].astype(np.float32)
        self.std       = baseline["std"].astype(np.float32)

        self.threshold = THRESHOLD
        print(f"[init] Config           : W={BEST_W}, step={BEST_STEP}, "
              f"agg={BEST_AGG}, norm={BEST_NORM}, baseline={BEST_BASELINE}")
        if self.threshold is not None:
            print(f"[init] Threshold        : {self.threshold:.6f}")
        else:
            print(f"[init] Threshold        : derived at runtime (95th-pct Normal)")
        print(f"[init] Ready.")

    def score(self, x_scaled: np.ndarray, batch_windows: int = BATCH_WINDOWS) -> np.ndarray:
        n_windows = (len(x_scaled) - BEST_W) // BEST_STEP + 1
        scores    = np.empty(n_windows, dtype=np.float32)

        for b_start in range(0, n_windows, batch_windows):
            b_end   = min(b_start + batch_windows, n_windows)
            starts  = np.arange(b_start, b_end) * BEST_STEP
            idx     = starts[:, None] + np.arange(BEST_W)[None, :]
            batch   = x_scaled[idx]                       # (B, W, F)
            scores[b_start:b_end] = score_batch(
                batch, self.mean, self.std, BEST_AGG, BEST_NORM
            )
        return scores

    def run(self, input_pkl: Path, batch_windows: int = BATCH_WINDOWS) -> pd.DataFrame:
        print(f"\n[run] Reading {input_pkl}")
        df = pd.read_pickle(input_pkl)
        print(f"[run] Loaded {len(df):,} rows x {df.shape[1]} cols")

        # Labels kept but not used in scoring
        y_bin = df[TARGET_BIN].values.astype(np.uint8) if TARGET_BIN in df.columns else None
        y_15c = df[TARGET_15C].values                  if TARGET_15C in df.columns else None

        # Clean
        clean = clean_batch(df.drop(columns=[c for c in LABEL_COLS if c in df.columns]))

        # Scale
        if self.feat_cols is  None:
            self.feat_cols = list(clean.select_dtypes(include=np.number).columns)

        x_scaled = self.scaler.transform(clean[self.feat_cols]).astype(np.float32)
        print(f"[run] Scaled shape      : {x_scaled.shape}")

        threshold = self.threshold
        if threshold is None and y_bin is not None:
            normal_idx = np.where(y_bin == 0)[0]
            if len(normal_idx) >= BEST_W:
                normal_scores = self.score(x_scaled[normal_idx], batch_windows)
                threshold     = float(np.percentile(normal_scores, 95))
                print(f"[run] Derived threshold : {threshold:.6f} "
                      f"(95th-pct of {len(normal_scores):,} Normal windows)")
            else:
                threshold = 0.5
                print(f"[run] Warning: too few Normal samples; using fallback threshold 0.5")

        # Score
        t0      = time.perf_counter()
        scores  = self.score(x_scaled, batch_windows)
        elapsed = time.perf_counter() - t0

        n_windows   = len(scores)
        predictions = (scores >= threshold).astype(int)

        print(f"[run] {elapsed*1000:.0f} ms total  |  "
              f"{elapsed / n_windows * 1000:.3f} ms/window  |  "
              f"{(n_windows * BEST_W) / elapsed:,.0f} packets/s")
        print(f"[run] Score distribution — "
              f"p50={np.percentile(scores, 50):.4f}  "
              f"p95={np.percentile(scores, 95):.4f}  "
              f"p99={np.percentile(scores, 99):.4f}")
        print(f"[run] Anomalies: {predictions.sum():,} / {n_windows:,} windows "
              f"({predictions.mean()*100:.2f}%)")

        # Attach scores
        window_starts = np.arange(n_windows) * BEST_STEP
        results = pd.DataFrame({
            "window_start_idx": window_starts,
            "anomaly_score":    scores,
            "anomaly_pred":     predictions,
        })

        if y_bin is not None:
            results["attack_label_majority"] = window_labels(y_bin[clean.index], BEST_W, BEST_STEP)

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    engine  = SWSInference()
    results = engine.run(INPUT_PKL)

    out = OUTPUT_CSV or INPUT_PKL.parent / (INPUT_PKL.stem + "_scored.csv")
    results.to_csv(out, index=False)
    print(f"[out] Saved → {out}")


if __name__ == "__main__":
    main()