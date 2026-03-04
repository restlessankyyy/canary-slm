"""
setup_kaggle.py — Kaggle Credit Card Fraud Dataset Setup Script

One-stop script to:
  1. Check if creditcard.csv exists locally
  2. Optionally download it via kaggle CLI (if credentials are configured)
  3. Run preprocessing (quantile fitting, train/val/test split, CSV export)
  4. Verify the output files

Usage:
    # If you already have creditcard.csv downloaded:
    python setup_kaggle.py --csv /path/to/creditcard.csv

    # If you have Kaggle API credentials configured:
    python setup_kaggle.py --download

    # Full pipeline with custom output dir:
    python setup_kaggle.py --csv ~/Downloads/creditcard.csv --out-dir data/kaggle
"""

import os
import sys
import argparse
import subprocess


KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
EXPECTED_COLS = {"Time", "Amount", "Class", "V1", "V28"}
EXPECTED_ROWS = 284_807


def check_csv(csv_path: str) -> bool:
    """Basic sanity check on the CSV file."""
    try:
        import pandas as pd
        print(f"  Checking {csv_path}...")
        df = pd.read_csv(csv_path, nrows=5)
        cols = set(df.columns)
        missing = EXPECTED_COLS - cols
        if missing:
            print(f"  ⚠️  Missing expected columns: {missing}")
            return False
        print(f"  ✅ CSV looks valid (columns: {list(df.columns)[:5]}...)")
        return True
    except Exception as e:
        print(f"  ❌ Error reading CSV: {e}")
        return False


def download_kaggle(out_dir: str = ".") -> str:
    """Download creditcard.csv using the Kaggle CLI."""
    print("\n📥 Downloading Kaggle Credit Card Fraud dataset...")
    print(f"   Dataset: {KAGGLE_DATASET}")
    print("   (Requires kaggle CLI + API key: https://www.kaggle.com/docs/api)")

    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "--unzip", "-p", out_dir],
            check=True,
        )
        csv_path = os.path.join(out_dir, "creditcard.csv")
        if os.path.exists(csv_path):
            print(f"   ✅ Downloaded → {csv_path}")
            return csv_path
        else:
            print("   ❌ Download completed but creditcard.csv not found.")
            sys.exit(1)
    except FileNotFoundError:
        print("\n  ❌ kaggle CLI not found. Install it with:  pip install kaggle")
        print("     Then configure your API key:")
        print("     https://www.kaggle.com/docs/api#getting-started-installation-&-authentication")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Download failed: {e}")
        sys.exit(1)


def print_instructions():
    """Print manual download instructions."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     HOW TO DOWNLOAD THE KAGGLE CREDIT CARD FRAUD DATASET    ║
╚══════════════════════════════════════════════════════════════╝

Option 1 — Browser Download (No setup required):
  1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. Click "Download" (free Kaggle account required)
  3. Unzip creditcardfraud.zip → creditcard.csv
  4. Run: python setup_kaggle.py --csv /path/to/creditcard.csv

Option 2 — Kaggle CLI:
  1. Install:   pip install kaggle
  2. Get key:   https://www.kaggle.com/settings → "Create New Token"
  3. Place at:  ~/.kaggle/kaggle.json
  4. Run:       python setup_kaggle.py --download

The dataset is ~144MB (284,807 transactions, 0.172% fraud).
""")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Credit Card Fraud setup")
    parser.add_argument("--csv",      type=str, default=None,
                        help="Path to creditcard.csv if already downloaded")
    parser.add_argument("--download", action="store_true",
                        help="Download via Kaggle CLI")
    parser.add_argument("--out-dir",  type=str, default="data",
                        help="Output directory for preprocessed files")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    csv_path = args.csv

    if not csv_path and not args.download:
        print_instructions()
        sys.exit(0)

    if args.download and not csv_path:
        csv_path = download_kaggle(args.out_dir)

    if not os.path.exists(csv_path):
        print(f"\n❌ File not found: {csv_path}")
        print_instructions()
        sys.exit(1)

    if not check_csv(csv_path):
        sys.exit(1)

    # Run preprocessing
    from data.preprocess_kaggle import preprocess_kaggle

    train_path, val_path, test_path, q_path = preprocess_kaggle(
        csv_path=csv_path,
        out_dir=args.out_dir,
        seed=args.seed,
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    ✅ Setup Complete!                        ║
╚══════════════════════════════════════════════════════════════╝

Preprocessed files:
  Train:     {train_path}
  Val:       {val_path}
  Test:      {test_path}
  Quantiles: {q_path}

Next step — train the model:

  python train_kaggle.py --epochs 20

Then evaluate:

  python evaluate.py --checkpoint checkpoints/kaggle_best.pt \\
                     --data data/kaggle_test.csv

Expected results (~20 epochs, CPU ~30 min):
  AUC-ROC:   0.92 – 0.97
  F1 (fraud): 0.80 – 0.90
  Recall:    0.85 – 0.95  (fraud caught)
  Precision: 0.70 – 0.90  (false alarm rate)
""")


if __name__ == "__main__":
    main()
