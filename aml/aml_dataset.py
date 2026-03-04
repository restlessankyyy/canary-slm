"""
aml/aml_dataset.py — PyTorch Dataset for account-level AML sequences

Input CSV columns:
    account_id, label, scheme,
    t0_amount, t0_direction, t0_country, t0_hour, t0_gap_hours,
    t1_amount, ..., t29_gap_hours

Each account → fixed 160-token tensor:
  [CLS] + 30 × [AMT][DIR][CTRY][TIME_BUCKET][GAP] = 1 + 150 = 151 tokens
  Padded to max_seq_len=160 with [PAD]=0
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Tuple

SEQ_LEN = 30   # transactions per account

# ── Minimal vocab for AML ─────────────────────────────────────────────────────

PAD_ID = 0
CLS_ID = 1

# Amount bins (same logic as FinancialTokenizer)
AMOUNT_BINS = [
    (1,       "AMT:MICRO"),       # < $1
    (10,      "AMT:TINY"),        # $1-10
    (50,      "AMT:SMALL"),       # $10-50
    (200,     "AMT:LOW"),         # $50-200
    (500,     "AMT:MED"),         # $200-500
    (1_000,   "AMT:1k"),
    (5_000,   "AMT:5k"),
    (9_800,   "AMT:9k"),          # structuring range
    (10_000,  "AMT:10k"),         # just over threshold
    (20_000,  "AMT:20k"),
    (50_000,  "AMT:50k"),
    (100_000, "AMT:100k"),
    (float("inf"), "AMT:WHALE"),
]

DIRECTION_TOKENS = {"IN": 20, "OUT": 21}

COUNTRY_TOKENS = {
    "US": 30, "CA": 31, "GB": 32, "DE": 33, "FR": 34,
    "AU": 35, "JP": 36, "SG": 37, "CH": 38, "NL": 39,
    "DOMESTIC": 30,
    # High-risk
    "NG": 50, "KP": 51, "IR": 52, "RU": 53, "BY": 54,
    "VE": 55, "CU": 56, "SY": 57,
    # Neutral foreign
    "OTHER": 45,
}

TIME_TOKENS = {  # hour → token
    "LATE_NIGHT":    60,  # 0-5
    "EARLY_MORNING": 61,  # 6-8
    "MORNING":       62,  # 9-11
    "MIDDAY":        63,  # 12-13
    "AFTERNOON":     64,  # 14-17
    "EVENING":       65,  # 18-21
    "NIGHT":         66,  # 22-23
}

GAP_TOKENS = {   # gap between transactions
    "GAP:RAPID":  70,   # < 2h
    "GAP:SAME_DAY": 71, # 2-24h
    "GAP:SHORT":  72,   # 1-3 days
    "GAP:NORMAL": 73,   # 3-14 days
    "GAP:LONG":   74,   # > 14 days
    "GAP:DORMANT":75,   # > 90 days
}

AML_VOCAB_SIZE = 100   # well within 600


def _amount_id(amount: float) -> int:
    for i, (threshold, _) in enumerate(AMOUNT_BINS):
        if amount < threshold:
            return 100 + i   # IDs 100–112
    return 112


def _country_id(country: str) -> int:
    return COUNTRY_TOKENS.get(str(country).upper(), 45)


def _time_id(hour: int) -> int:
    h = int(hour)
    if h < 6:
        return TIME_TOKENS["LATE_NIGHT"]
    if h < 9:
        return TIME_TOKENS["EARLY_MORNING"]
    if h < 12:
        return TIME_TOKENS["MORNING"]
    if h < 14:
        return TIME_TOKENS["MIDDAY"]
    if h < 18:
        return TIME_TOKENS["AFTERNOON"]
    if h < 22:
        return TIME_TOKENS["EVENING"]
    return TIME_TOKENS["NIGHT"]


def _gap_id(gap_hours: float) -> int:
    g = float(gap_hours)
    if g < 2:
        return GAP_TOKENS["GAP:RAPID"]
    if g < 24:
        return GAP_TOKENS["GAP:SAME_DAY"]
    if g < 72:
        return GAP_TOKENS["GAP:SHORT"]
    if g < 336:
        return GAP_TOKENS["GAP:NORMAL"]
    if g < 2160:
        return GAP_TOKENS["GAP:LONG"]
    return GAP_TOKENS["GAP:DORMANT"]


def encode_account(row: pd.Series, max_seq_len: int = 160) -> Tuple[List[int], List[int]]:
    """
    Convert one account row → (input_ids, attention_mask).

    Token layout per transaction (5 tokens):
      [AMOUNT_BIN] [DIRECTION] [COUNTRY] [TIME_OF_DAY] [GAP_BIN]
    """
    tokens = [CLS_ID]
    for i in range(SEQ_LEN):
        amt   = row.get(f"t{i}_amount",    0)
        direc = row.get(f"t{i}_direction", "OUT")
        cntry = row.get(f"t{i}_country",   "US")
        hour  = row.get(f"t{i}_hour",      12)
        gap   = row.get(f"t{i}_gap_hours", 24)

        tokens.append(_amount_id(float(amt)))
        tokens.append(DIRECTION_TOKENS.get(str(direc).upper(), 21))
        tokens.append(_country_id(cntry))
        tokens.append(_time_id(hour))
        tokens.append(_gap_id(gap))

    tokens = tokens[:max_seq_len]
    seq_len = len(tokens)
    attn = [1] * seq_len

    if seq_len < max_seq_len:
        pad = max_seq_len - seq_len
        tokens += [PAD_ID] * pad
        attn   += [0]      * pad

    return tokens, attn


# ── Dataset ───────────────────────────────────────────────────────────────────

class AMLDataset(Dataset):
    """Per-account AML sequence dataset."""

    def __init__(self, csv_path: str, max_seq_len: int = 160):
        self.df = pd.read_csv(csv_path)
        self.max_seq_len = max_seq_len

        print(f"  Encoding {len(self.df):,} accounts from {csv_path}...")
        ids_list, masks_list, labels_list = [], [], []
        for _, row in self.df.iterrows():
            ids, mask = encode_account(row, max_seq_len)
            ids_list.append(ids)
            masks_list.append(mask)
            labels_list.append(int(row["label"]))

        self._ids    = torch.tensor(ids_list,    dtype=torch.long)
        self._masks  = torch.tensor(masks_list,  dtype=torch.long)
        self._labels = torch.tensor(labels_list, dtype=torch.long)

        n_susp = self._labels.sum().item()
        n_clean = len(self._labels) - n_susp
        print(f"  Loaded {len(self._labels):,} | Clean: {n_clean:,} | Suspicious: {n_susp:,} ({100*n_susp/len(self._labels):.1f}%)")

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return self._ids[idx], self._masks[idx], self._labels[idx]

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        n_susp  = int(self._labels.sum().item())
        n_clean = len(self._labels) - n_susp
        w = torch.tensor([1.0 / max(n_clean, 1), 1.0 / max(n_susp, 1)], dtype=torch.float)
        return WeightedRandomSampler(w[self._labels], len(self._labels), replacement=True)


def build_aml_dataloaders(
    data_dir:    str = "aml/data",
    batch_size:  int = 128,
    max_seq_len: int = 160,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders for the AML dataset."""

    train_ds = AMLDataset(os.path.join(data_dir, "train.csv"), max_seq_len)
    val_ds   = AMLDataset(os.path.join(data_dir, "val.csv"),   max_seq_len)
    test_ds  = AMLDataset(os.path.join(data_dir, "test.csv"),  max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_ds.get_weighted_sampler(),
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size * 2,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
