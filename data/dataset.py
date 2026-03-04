"""
data/dataset.py — PyTorch Dataset for Financial Transaction Classification

Loads CSV transaction data, tokenizes rows, and returns tensors
for use with DataLoader.
"""

import os
import ast
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, Tuple, Dict

from data.tokenizer import FinancialTokenizer, get_tokenizer


class FraudDataset(Dataset):
    """
    PyTorch Dataset for fraud transaction classification.

    Each sample:
        input_ids:      LongTensor (seq_len,)
        attention_mask: LongTensor (seq_len,)
        label:          LongTensor scalar  — 0=legit, 1=fraud
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: Optional[FinancialTokenizer] = None,
        max_seq_len: int = 64,
        vocab_path: Optional[str] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer or get_tokenizer(vocab_path)
        self.max_seq_len = max_seq_len

        # Pre-compute encoded samples for speed
        self._input_ids = []
        self._attention_masks = []
        self._labels = []

        print(f"  Tokenizing {len(self.df):,} transactions from {csv_path}...")
        for _, row in self.df.iterrows():
            txn = self._row_to_txn(row)
            ids, mask = self.tokenizer.encode(txn, max_length=max_seq_len, padding=True)
            self._input_ids.append(ids)
            self._attention_masks.append(mask)
            self._labels.append(int(row["label"]))

        self._input_ids = torch.tensor(self._input_ids, dtype=torch.long)
        self._attention_masks = torch.tensor(self._attention_masks, dtype=torch.long)
        self._labels = torch.tensor(self._labels, dtype=torch.long)

        n_fraud = self._labels.sum().item()
        n_legit = len(self._labels) - n_fraud
        print(f"  Loaded {len(self._labels):,} samples | Legit: {n_legit:,} | Fraud: {n_fraud:,}")

    @staticmethod
    def _row_to_txn(row: pd.Series) -> Dict:
        """Convert a CSV row back into a transaction dict for the tokenizer."""
        flags_raw = row.get("flags", "")
        flags = [f for f in str(flags_raw).split("|") if f] if pd.notna(flags_raw) and flags_raw else []

        return {
            "amount": float(row.get("amount", 0)),
            "merchant_cat": str(row.get("merchant_cat", "UNKNOWN")),
            "country": str(row.get("country", "US")),
            "is_domestic": bool(int(row.get("is_domestic", 1))),
            "hour": int(row.get("hour", 12)),
            "day_of_week": int(row.get("day_of_week", 0)),
            "channel": str(row.get("channel", "POS_CHIP")),
            "currency": str(row.get("currency", "USD")),
            "velocity": str(row.get("velocity", "NORMAL")),
            "flags": flags,
        }

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._input_ids[idx],
            self._attention_masks[idx],
            self._labels[idx],
        )

    @property
    def class_counts(self) -> Tuple[int, int]:
        n_fraud = self._labels.sum().item()
        return int(len(self._labels) - n_fraud), int(n_fraud)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler to oversample fraud transactions
        so each batch has a balanced class distribution.
        """
        n_legit, n_fraud = self.class_counts
        class_weights = torch.tensor([1.0 / n_legit, 1.0 / n_fraud], dtype=torch.float)
        sample_weights = class_weights[self._labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


def build_dataloaders(
    data_dir: str = "data",
    batch_size: int = 256,
    max_seq_len: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_weighted_sampler: bool = True,
    vocab_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    tokenizer = get_tokenizer(vocab_path)

    train_ds = FraudDataset(
        os.path.join(data_dir, "train.csv"),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    val_ds = FraudDataset(
        os.path.join(data_dir, "val.csv"),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    test_ds = FraudDataset(
        os.path.join(data_dir, "test.csv"),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    sampler = train_ds.get_weighted_sampler() if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check
    from data.generate_synthetic import generate_dataset
    import tempfile, os

    print("Generating mini dataset for sanity check...")
    with tempfile.TemporaryDirectory() as tmpdir:
        for split, n in [("train", 500), ("val", 100), ("test", 100)]:
            df = generate_dataset(n, fraud_ratio=0.1, seed=42)
            df.to_csv(os.path.join(tmpdir, f"{split}.csv"), index=False)

        train_loader, val_loader, test_loader = build_dataloaders(
            data_dir=tmpdir, batch_size=32, num_workers=0
        )
        batch = next(iter(train_loader))
        ids, mask, labels = batch
        print(f"\nBatch shapes: ids={ids.shape}, mask={mask.shape}, labels={labels.shape}")
        print(f"Fraud in batch: {labels.sum().item()}/{len(labels)}")
