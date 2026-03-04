"""
data/kaggle_dataset.py — PyTorch Dataset for the real Kaggle Credit Card Fraud data

Loads the preprocessed Kaggle CSV (produced by preprocess_kaggle.py).
Tokenizes rows using an extended vocabulary that includes PCA-bin tokens
(V1:MID, V14:VHIGH, …) in addition to the standard financial tokens.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, Tuple, List

from data.tokenizer import FinancialTokenizer, _build_vocab, get_tokenizer
from data.preprocess_kaggle import build_kaggle_vocab_extension, PCA_FEATURES, PCA_BINS


# ── Extended tokenizer ────────────────────────────────────────────────────────

def build_kaggle_tokenizer() -> FinancialTokenizer:
    """
    Build a FinancialTokenizer extended with PCA-bin tokens.
    Standard vocab (≤512) + 140 PCA tokens = up to 652 tokens.
    """
    vocab = _build_vocab()
    idx = max(vocab.values()) + 1
    for token in build_kaggle_vocab_extension():
        if token not in vocab:
            vocab[token] = idx
            idx += 1
    return FinancialTokenizer(vocab=vocab)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KaggleFraudDataset(Dataset):
    """
    PyTorch Dataset for the preprocessed Kaggle credit card fraud CSV.

    CSV columns expected:
        amount, hour, label
        pca_features  — pipe-separated string of pre-computed PCA tokens
                         e.g. "V1:MID|V2:VHIGH|V3:LOW|..."

    Each sample:
        input_ids:       LongTensor (max_seq_len,)
        attention_mask:  LongTensor (max_seq_len,)
        label:           LongTensor scalar
    """

    CLS_ID = 1
    PAD_ID = 0

    def __init__(
        self,
        csv_path: str,
        tokenizer: Optional[FinancialTokenizer] = None,
        max_seq_len: int = 64,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer or build_kaggle_tokenizer()
        self.max_seq_len = max_seq_len

        print(f"  Tokenizing {len(self.df):,} Kaggle transactions from {csv_path}...")

        self._input_ids      = []
        self._attention_masks = []
        self._labels          = []

        for _, row in self.df.iterrows():
            ids, mask = self._encode_row(row)
            self._input_ids.append(ids)
            self._attention_masks.append(mask)
            self._labels.append(int(row["label"]))

        self._input_ids       = torch.tensor(self._input_ids,       dtype=torch.long)
        self._attention_masks = torch.tensor(self._attention_masks, dtype=torch.long)
        self._labels          = torch.tensor(self._labels,          dtype=torch.long)

        n_fraud = self._labels.sum().item()
        n_legit = len(self._labels) - n_fraud
        print(f"  Loaded {len(self._labels):,} | Legit: {n_legit:,} | Fraud: {n_fraud:,} ({100*n_fraud/len(self._labels):.2f}%)")

    def _encode_row(self, row: pd.Series) -> Tuple[List[int], List[int]]:
        """
        Convert a preprocessed Kaggle row to (input_ids, attention_mask).

        Token sequence:
          [CLS] [AMT:xxx] [TIME:xxx] [V1:xxx] [V2:xxx] … [V28:xxx]
        """
        tok = self.tokenizer
        tokens = [self.CLS_ID]

        # Amount
        amount = float(row.get("amount", 0))
        amt_tok = tok._amount_token(amount)
        tokens.append(tok._tok(amt_tok))

        # Time of day
        hour = int(row.get("hour", 12))
        time_tok = tok._time_token(hour)
        tokens.append(tok._tok(time_tok))

        # PCA feature tokens (pipe-separated string)
        pca_raw = str(row.get("pca_features", ""))
        if pca_raw:
            for pca_tok in pca_raw.split("|"):
                tok_id = tok._tok(pca_tok.strip())
                tokens.append(tok_id)

        # Truncate & pad
        tokens = tokens[:self.max_seq_len]
        seq_len = len(tokens)
        attention_mask = [1] * seq_len

        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            tokens         += [self.PAD_ID] * pad_len
            attention_mask += [0] * pad_len

        return tokens, attention_mask

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
        n_fraud = int(self._labels.sum().item())
        return len(self._labels) - n_fraud, n_fraud

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        n_legit, n_fraud = self.class_counts
        class_weights = torch.tensor(
            [1.0 / max(n_legit, 1), 1.0 / max(n_fraud, 1)], dtype=torch.float
        )
        sample_weights = class_weights[self._labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


def build_kaggle_dataloaders(
    data_dir: str = "data",
    batch_size: int = 256,
    max_seq_len: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders for the Kaggle dataset.
    Expects preprocessed files: kaggle_train.csv, kaggle_val.csv, kaggle_test.csv
    """
    tokenizer = build_kaggle_tokenizer()
    print(f"\n  Kaggle vocab size: {tokenizer.vocab_size}")

    train_ds = KaggleFraudDataset(
        os.path.join(data_dir, "kaggle_train.csv"), tokenizer=tokenizer, max_seq_len=max_seq_len
    )
    val_ds = KaggleFraudDataset(
        os.path.join(data_dir, "kaggle_val.csv"), tokenizer=tokenizer, max_seq_len=max_seq_len
    )
    test_ds = KaggleFraudDataset(
        os.path.join(data_dir, "kaggle_test.csv"), tokenizer=tokenizer, max_seq_len=max_seq_len
    )

    sampler = train_ds.get_weighted_sampler() if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
