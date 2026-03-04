"""
download_kaggle_data.py — Download Credit Card Fraud dataset via kagglehub
No API credentials required for public datasets.
"""
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

print("📥 Downloading Kaggle Credit Card Fraud dataset via kagglehub...")
print("   (First run may take a moment to download ~144MB)")

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "mlg-ulb/creditcardfraud",
    "creditcard.csv",   # explicit filename inside the dataset
)

print(f"\n✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"   Columns: {list(df.columns)}")
print(f"   Fraud:   {df['Class'].sum():,} ({100*df['Class'].mean():.3f}%)")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

# Save to data/creditcard.csv
os.makedirs("data", exist_ok=True)
out_path = "data/creditcard.csv"
df.to_csv(out_path, index=False)
print(f"\n💾 Saved to {out_path}  ({os.path.getsize(out_path)//1_048_576} MB)")
