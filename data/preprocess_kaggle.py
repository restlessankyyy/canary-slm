"""
data/preprocess_kaggle.py — Kaggle Credit Card Fraud Dataset Adapter

Converts the Kaggle creditcard.csv format into our existing pipeline.

Kaggle columns:
    Time    — Seconds elapsed since first transaction in dataset
    V1–V28  — PCA-transformed anonymised features (continuous)
    Amount  — Transaction amount (USD)
    Class   — 0=Legitimate, 1=Fraud

Strategy:
  - Amount  → existing FinancialTokenizer amount bins
  - Time    → 6 time-of-day buckets (assuming 48h window, cycled mod 86400)
  - V1–V28  → each binned into 5 quantile-based tokens per feature
              (VERY_LOW / LOW / MID / HIGH / VERY_HIGH)
              Quantile boundaries are fit on the training set and saved.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ── Feature-level token names ─────────────────────────────────────────────────

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]   # V1 … V28
PCA_BINS = ["VLOW", "LOW", "MID", "HIGH", "VHIGH"]  # 5-quantile labels

# Token format:  V1:MID,  V14:VHIGH, …
def pca_token(feature: str, bin_label: str) -> str:
    return f"{feature}:{bin_label}"


def time_to_bucket(seconds: float) -> int:
    """
    Convert Kaggle "Time" (elapsed seconds from start of dataset) to
    an approximate hour-of-day (mod 86400) then time bucket index 0–5.
    """
    hour = int(seconds % 86400 / 3600)
    if hour < 6:   return 0   # EARLY_MORNING
    if hour < 11:  return 1   # MORNING
    if hour < 14:  return 2   # MIDDAY
    if hour < 18:  return 3   # AFTERNOON
    if hour < 22:  return 4   # EVENING
    return 5                  # LATE_NIGHT

TIME_BUCKET_NAMES = [
    "EARLY_MORNING", "MORNING", "MIDDAY", "AFTERNOON", "EVENING", "LATE_NIGHT"
]


# ── Quantile fitter ───────────────────────────────────────────────────────────

class KaggleQuantileFitter:
    """
    Fits per-feature quantile boundaries on training data and
    transforms continuous PCA values → bin index.
    """

    N_BINS = 5

    def __init__(self):
        self.boundaries: Dict[str, List[float]] = {}  # feature → [q20, q40, q60, q80]

    def fit(self, df: pd.DataFrame) -> "KaggleQuantileFitter":
        """Compute quantile boundaries from a training DataFrame."""
        for feat in PCA_FEATURES:
            if feat not in df.columns:
                continue
            vals = df[feat].values
            qs = np.percentile(vals, [20, 40, 60, 80])
            self.boundaries[feat] = qs.tolist()
        return self

    def transform_value(self, feature: str, value: float) -> str:
        """Map a continuous value to a bin label string."""
        boundaries = self.boundaries.get(feature)
        if boundaries is None:
            return "MID"
        for i, b in enumerate(boundaries):
            if value < b:
                return PCA_BINS[i]
        return PCA_BINS[-1]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.boundaries, f)
        print(f"  Quantile boundaries saved → {path}")

    @classmethod
    def load(cls, path: str) -> "KaggleQuantileFitter":
        with open(path) as f:
            boundaries = json.load(f)
        fitter = cls()
        fitter.boundaries = boundaries
        return fitter


# ── Row → transaction dict ────────────────────────────────────────────────────

def kaggle_row_to_txn(
    row: pd.Series,
    fitter: KaggleQuantileFitter,
) -> Dict:
    """
    Convert one Kaggle creditcard.csv row to a transaction dict
    compatible with FinancialTokenizer.

    The transaction dict uses a custom 'pca_tokens' key that holds
    the pre-computed PCA bin tokens (V1:MID, V2:VHIGH, …).
    The KaggleTokenizer (see below) reads these directly.
    """
    time_bucket = TIME_BUCKET_NAMES[time_to_bucket(float(row.get("Time", 0)))]
    amount = float(row.get("Amount", 0))

    # Bin each V1–V28 feature
    pca_tokens = []
    for feat in PCA_FEATURES:
        val = float(row.get(feat, 0))
        bin_label = fitter.transform_value(feat, val)
        pca_tokens.append(pca_token(feat, bin_label))

    return {
        "amount":      amount,
        "time_bucket": time_bucket,   # custom key used by KaggleTokenizer
        "pca_tokens":  pca_tokens,    # custom key: pre-binned V1–V28
        # Defaults (not in Kaggle dataset)
        "merchant_cat": "UNKNOWN",
        "country":      "DOMESTIC",
        "is_domestic":  True,
        "hour":         time_to_bucket(float(row.get("Time", 0))) * 4,
        "day_of_week":  0,
        "channel":      "POS_CHIP",
        "currency":     "USD",
        "velocity":     "NORMAL",
        "flags":        [],
    }


# ── Extended Kaggle Vocabulary ─────────────────────────────────────────────────

def build_kaggle_vocab_extension() -> List[str]:
    """
    Build the list of PCA-based tokens to add to the vocabulary.
    Format:  V1:VLOW, V1:LOW, V1:MID, V1:HIGH, V1:VHIGH, V2:VLOW, …
    = 28 features × 5 bins = 140 tokens
    """
    tokens = []
    for feat in PCA_FEATURES:
        for b in PCA_BINS:
            tokens.append(pca_token(feat, b))
    return tokens


# ── Preprocessing pipeline ─────────────────────────────────────────────────────

def preprocess_kaggle(
    csv_path: str,
    out_dir: str = "data",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    # test_frac implicitly = remaining
    seed: int = 42,
    quantile_path: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """
    Load Kaggle creditcard.csv, split into train/val/test,
    fit quantile boundaries on train, and save three CSV files
    in the project's data format.

    Returns:
        (train_path, val_path, test_path, quantile_path)
    """
    print(f"\n📂 Loading Kaggle dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   {len(df):,} rows | Fraud: {df['Class'].sum():,} ({100*df['Class'].mean():.3f}%)")

    # Rename Class → label for our pipeline
    df = df.rename(columns={"Class": "label"})

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split
    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"\n   Split: {len(train_df):,} train / {len(val_df):,} val / {len(test_df):,} test")

    # Fit quantiles on train
    print("\n⚙️  Fitting per-feature quantile boundaries on training set...")
    fitter = KaggleQuantileFitter()
    fitter.fit(train_df)

    q_path = quantile_path or os.path.join(out_dir, "kaggle_quantiles.json")
    fitter.save(q_path)

    # Convert each split to project format
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n🔄 Converting {split_name} split ({len(split_df):,} rows)...")
        rows = []
        for _, row in split_df.iterrows():
            txn = kaggle_row_to_txn(row, fitter)
            rows.append({
                "amount":       txn["amount"],
                "merchant_cat": "UNKNOWN",
                "country":      "DOMESTIC",
                "is_domestic":  1,
                "hour":         int(time_to_bucket(float(row.get("Time", 0))) * 4),
                "day_of_week":  0,
                "channel":      "POS_CHIP",
                "currency":     "USD",
                "velocity":     "NORMAL",
                "flags":        "",
                # Store the PCA bins as a pipe-separated string in a new column
                "pca_features": "|".join(txn["pca_tokens"]),
                "label":        int(row["label"]),
            })
        out_df = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, f"kaggle_{split_name}.csv")
        out_df.to_csv(out_path, index=False)
        fraud_n = out_df["label"].sum()
        print(f"   Saved {len(out_df):,} rows → {out_path}  (fraud: {fraud_n:,}, {100*fraud_n/len(out_df):.2f}%)")
        paths.append(out_path)

    print("\n✅ Kaggle preprocessing complete.")
    return tuple(paths) + (q_path,)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Kaggle creditcard.csv")
    parser.add_argument("--csv",      required=True, help="Path to creditcard.csv")
    parser.add_argument("--out-dir",  default="data",  help="Output directory")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    preprocess_kaggle(args.csv, out_dir=args.out_dir, seed=args.seed)
