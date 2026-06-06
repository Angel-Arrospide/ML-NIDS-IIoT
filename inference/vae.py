import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import Filters

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCALER_PATH   = Path("/home/aam/ML-NIDS-IIoT/inference/data/VAE_num_latent4_leaky_relu_scaler.joblib")
MODEL_PATH    = Path("/home/aam/ML-NIDS-IIoT/inference/data/VAE_num_latent4_leaky_relu_weights.pth")
INPUT_PKL     = Path("/home/aam/ML-NIDS-IIoT/inference/data/DL-EdgeIIoT-dataset_clean.pkl")
OUTPUT_CSV    = None          # None = auto-name as <input>_scored.csv

BATCH_SIZE    = 256
THRESHOLD     = 0.035427      # From notebook
LABEL_COLS    = ["Attack_type", "Attack_label"]

# ---------------------------------------------------------------------------
# Arch (latent4_leaky_relu)
# ---------------------------------------------------------------------------
INPUT_DIM    = 43
HIDDEN_DIMS  = (32, 24, 16)
LATENT_DIM   = 4

# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------
class VAE(nn.Module):

    _EPS = 1e-8

    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU

        self.encoder = self._mlp([INPUT_DIM, *HIDDEN_DIMS], act, bn=True, final_act=True)
        self.to_latent = nn.Linear(HIDDEN_DIMS[-1], LATENT_DIM * 2)
        self.decoder = self._mlp([LATENT_DIM, *reversed(HIDDEN_DIMS), INPUT_DIM],
                                  act, bn=True, final_act=False)

    @staticmethod
    def _mlp(dims, activation, bn, final_act):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = (i == len(dims) - 2)
            if not is_last or final_act:
                if bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(activation())
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(self.to_latent(h), 2, dim=-1)
        return mu, log_var

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z   = mu + std * torch.randn_like(std)
        return torch.sigmoid(self.decoder(z)), mu, log_var

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        mu, _ = self.encode(x)
        recon  = torch.sigmoid(self.decoder(mu))
        return 1.0 - torch.cosine_similarity(x, recon, dim=1)


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
class VAEInference:

    def __init__(self):
        print(f"[init] Loading scaler  → {SCALER_PATH}")
        self.scaler    = joblib.load(SCALER_PATH)
        self.feat_cols = list(self.scaler.feature_names_in_)   # canonical column order
        print(f"[init] Scaler features : {len(self.feat_cols)}")

        print(f"[init] Building VAE (latent4_leaky_relu)")
        self.model = VAE()
        print(f"[init] Loading weights → {MODEL_PATH}")
        self.model.load_state_dict(
            torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
        )
        self.model.eval()
        print(f"[init] Threshold       : {THRESHOLD:.6f}")
        print(f"[init] Ready.")

    def score(self, x_scaled: np.ndarray) -> np.ndarray:
        t = torch.tensor(x_scaled, dtype=torch.float32)
        return self.model.reconstruction_error(t).numpy()

    def run(self, input_pkl: Path, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
        print(f"\n[run] Reading {input_pkl}")
        df = pd.read_pickle(input_pkl)
        print(f"[run] Loaded {len(df):,} rows × {df.shape[1]} cols")

        all_scores   = []
        kept_indices = []
        n_batches    = int(np.ceil(len(df) / batch_size))
        t0           = time.perf_counter()

        for i in range(n_batches):
            raw   = df.iloc[i * batch_size : (i + 1) * batch_size].copy()
            clean = clean_batch(raw)
            if clean.empty:
                continue

            # Recover original index (ValidTimeFilter resets index internally)
            if Filters.VALID_TIME_COL in raw.columns:
                surviving = raw[raw[Filters.VALID_TIME_COL] != 0].index
            else:
                surviving = raw.index

            x_scaled = self.scaler.transform(clean[self.feat_cols])
            scores   = self.score(x_scaled)

            all_scores.append(scores)
            kept_indices.extend(surviving.tolist())

        elapsed = time.perf_counter() - t0
        scores      = np.concatenate(all_scores)
        predictions = (scores >= THRESHOLD).astype(int)

        print(f"[run] {elapsed*1000:.0f} ms total  |  {elapsed/len(scores)*1000:.3f} ms/sample")
        print(f"[run] Score distribution — "
              f"p50={np.percentile(scores,50):.4f}  "
              f"p95={np.percentile(scores,95):.4f}  "
              f"p99={np.percentile(scores,99):.4f}")
        print(f"[run] Anomalies: {predictions.sum():,} / {len(predictions):,} "
              f"({predictions.mean()*100:.2f}%)")

        results = df.loc[kept_indices].copy()
        results["anomaly_score"] = scores
        results["anomaly_pred"]  = predictions
        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    engine  = VAEInference()
    results = engine.run(INPUT_PKL)

    out = OUTPUT_CSV or INPUT_PKL.parent / (INPUT_PKL.stem + "_scored.csv")
    results.to_csv(out, index=False)
    print(f"[out] Saved → {out}")


if __name__ == "__main__":
    main()