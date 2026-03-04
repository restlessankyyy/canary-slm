"""
aml/generate_aml_data.py — Synthetic AML Dataset Generator

Generates synthetic account-level transaction histories labelled as:
  0 = Clean account (normal spending behaviour)
  1 = Suspicious account (one of 5 AML schemes)

Each row in the output CSV represents ONE ACCOUNT with 30 transactions
encoded as flat feature columns:
  t0_amount, t0_direction, t0_country, t0_hour, t0_gap_hours,
  t1_amount, t1_direction, t1_country, ...
  label

AML Schemes:
  1. Structuring   — repeated deposits $8k-$9.5k (just under reporting $10k)
  2. Layering      — rapid in→out chains, multiple countries
  3. Smurfing      — same total split into many small transfers to different people
  4. Dormant burst — 90+ day quiet period then sudden large burst
  5. Round-tripping — funds leave then return via different route within 7 days
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

random.seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN = 30          # transactions per account
COUNTRIES_DOMESTIC = ["US", "US", "US", "CA", "GB"]
COUNTRIES_FOREIGN  = ["NG", "KP", "IR", "RU", "BY", "VE", "CU", "SY"]
COUNTRIES_NEUTRAL  = ["DE", "FR", "AU", "JP", "SG", "CH", "NL"]

MCC_NORMAL  = ["GROCERY", "RESTAURANT", "FUEL", "RETAIL", "PHARMACY", "UTILITIES"]
MCC_RISKY   = ["CRYPTO", "MONEY_TRANSFER", "CASINO", "FOREX", "SHELL_CO"]


# ── Transaction builders ──────────────────────────────────────────────────────

def _txn(amount: float, direction: str, country: str, hour: int, gap_hours: float) -> Dict:
    return {
        "amount":    round(amount, 2),
        "direction": direction,   # IN / OUT
        "country":   country,
        "hour":      hour % 24,
        "gap_hours": round(gap_hours, 1),
    }


def _rand_clean_txn(prev_gap: float = None) -> Dict:
    gap = prev_gap if prev_gap else random.uniform(12, 72)
    return _txn(
        amount    = round(np.random.lognormal(4, 0.8), 2),
        direction = random.choice(["IN", "IN", "OUT", "OUT", "OUT"]),
        country   = random.choice(COUNTRIES_DOMESTIC),
        hour      = random.randint(8, 22),
        gap_hours = gap,
    )


# ── Clean account ─────────────────────────────────────────────────────────────

def generate_clean_account() -> List[Dict]:
    """Normal consumer spending. Variable amounts, domestic, business hours."""
    txns = []
    for _ in range(SEQ_LEN):
        txns.append(_rand_clean_txn())
    return txns


# ── AML Scheme 1: Structuring ─────────────────────────────────────────────────

def generate_structuring_account() -> List[Dict]:
    """
    Multiple cash deposits just under $10,000 (US BSA reporting threshold).
    Interspersed with some normal transactions to hide the pattern.
    """
    txns = []
    n_structured = random.randint(8, 14)   # 8-14 structured deposits in window
    structured_indices = sorted(random.sample(range(SEQ_LEN), n_structured))

    for i in range(SEQ_LEN):
        if i in structured_indices:
            txns.append(_txn(
                amount    = random.uniform(8_500, 9_800),  # just under $10k
                direction = "IN",
                country   = random.choice(COUNTRIES_DOMESTIC),
                hour      = random.randint(9, 17),          # business hours
                gap_hours = random.uniform(2, 36),          # multiple per day
            ))
        else:
            txns.append(_rand_clean_txn())
    return txns


# ── AML Scheme 2: Layering ────────────────────────────────────────────────────

def generate_layering_account() -> List[Dict]:
    """
    Funds move rapidly: large in, then immediately out to foreign accounts,
    multiple hops across countries within hours.
    """
    txns = []
    i = 0
    while i < SEQ_LEN:
        # Inject a rapid layering burst
        if i < SEQ_LEN - 4 and random.random() < 0.4:
            amount = random.uniform(20_000, 200_000)
            txns.append(_txn(amount, "IN",  random.choice(COUNTRIES_NEUTRAL), random.randint(0, 6),   0.5))
            txns.append(_txn(amount * 0.99, "OUT", random.choice(COUNTRIES_FOREIGN), random.randint(0, 6), 1.0))
            txns.append(_txn(amount * 0.97, "IN",  random.choice(COUNTRIES_FOREIGN), random.randint(0, 6), 2.0))
            txns.append(_txn(amount * 0.95, "OUT", random.choice(COUNTRIES_FOREIGN), random.randint(0, 6), 1.5))
            i += 4
        else:
            txns.append(_rand_clean_txn())
            i += 1
    return txns[:SEQ_LEN]


# ── AML Scheme 3: Smurfing ────────────────────────────────────────────────────

def generate_smurfing_account() -> List[Dict]:
    """
    One large source amount split into many small outgoing transfers
    (to different recipients, same period). Avoids threshold detection.
    """
    txns = []
    # One large incoming deposit
    total = random.uniform(50_000, 500_000)
    txns.append(_txn(total, "IN", random.choice(COUNTRIES_DOMESTIC), random.randint(9, 12), 48.0))

    # Rapid small outgoing transfers
    n_out = random.randint(12, SEQ_LEN - 4)
    per_transfer = total / n_out
    for _ in range(n_out):
        txns.append(_txn(
            amount    = per_transfer * random.uniform(0.85, 1.15),
            direction = "OUT",
            country   = random.choice(COUNTRIES_DOMESTIC + COUNTRIES_NEUTRAL),
            hour      = random.randint(8, 20),
            gap_hours = random.uniform(0.5, 6),
        ))

    # Pad with clean transactions
    while len(txns) < SEQ_LEN:
        txns.append(_rand_clean_txn())
    return txns[:SEQ_LEN]


# ── AML Scheme 4: Dormant + Burst ─────────────────────────────────────────────

def generate_dormant_burst_account() -> List[Dict]:
    """
    Account is quiet for months, then suddenly activates with large transactions.
    """
    txns = []
    # First ~20 transactions: very infrequent, small amounts (dormant)
    for _ in range(20):
        txns.append(_txn(
            amount    = random.uniform(5, 100),
            direction = random.choice(["IN", "OUT"]),
            country   = random.choice(COUNTRIES_DOMESTIC),
            hour      = random.randint(8, 18),
            gap_hours = random.uniform(200, 800),  # Very long gaps
        ))
    # Last 10 transactions: sudden large burst
    for _ in range(10):
        txns.append(_txn(
            amount    = random.uniform(10_000, 250_000),
            direction = random.choice(["IN", "OUT"]),
            country   = random.choice(COUNTRIES_FOREIGN + COUNTRIES_NEUTRAL),
            hour      = random.randint(0, 6),   # Late night
            gap_hours = random.uniform(0.5, 4),  # Rapid fire
        ))
    return txns[:SEQ_LEN]


# ── AML Scheme 5: Round-tripping ──────────────────────────────────────────────

def generate_round_trip_account() -> List[Dict]:
    """
    Funds leave (OUT) and return (IN) via a different route within days.
    Pattern: big OUT to foreign, then big IN from different foreign country.
    """
    txns = []
    i = 0
    while i < SEQ_LEN:
        if i < SEQ_LEN - 2 and random.random() < 0.35:
            amount = random.uniform(15_000, 300_000)
            out_country = random.choice(COUNTRIES_FOREIGN)
            in_country  = random.choice([c for c in COUNTRIES_FOREIGN if c != out_country])
            txns.append(_txn(amount,            "OUT", out_country, random.randint(0, 23), random.uniform(24, 120)))
            txns.append(_txn(amount * random.uniform(0.93, 0.99), "IN", in_country, random.randint(0, 23), random.uniform(12, 96)))
            i += 2
        else:
            txns.append(_rand_clean_txn())
            i += 1
    return txns[:SEQ_LEN]


# ── Dataset assembly ──────────────────────────────────────────────────────────

SCHEME_GENERATORS = [
    generate_structuring_account,
    generate_layering_account,
    generate_smurfing_account,
    generate_dormant_burst_account,
    generate_round_trip_account,
]

SCHEME_NAMES = [
    "structuring", "layering", "smurfing", "dormant_burst", "round_trip"
]


def account_to_row(account_id: int, txns: List[Dict], label: int, scheme: str = "") -> Dict:
    """Flatten 30-transaction account into a single CSV row."""
    row = {"account_id": account_id, "label": label, "scheme": scheme}
    for i, t in enumerate(txns):
        row[f"t{i}_amount"]    = t["amount"]
        row[f"t{i}_direction"] = t["direction"]
        row[f"t{i}_country"]   = t["country"]
        row[f"t{i}_hour"]      = t["hour"]
        row[f"t{i}_gap_hours"] = t["gap_hours"]
    return row


def generate_aml_dataset(
    n_accounts:      int   = 10_000,
    suspicious_ratio: float = 0.05,
    seed:            int   = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic AML dataset.
    Returns a DataFrame where each row = one account's 30-transaction history.
    """
    random.seed(seed)
    np.random.seed(seed)

    n_suspicious = int(n_accounts * suspicious_ratio)
    n_clean      = n_accounts - n_suspicious

    rows = []

    # Clean accounts
    for i in range(n_clean):
        txns = generate_clean_account()
        rows.append(account_to_row(i, txns, label=0, scheme="clean"))

    # Suspicious accounts (balanced across schemes)
    for i in range(n_suspicious):
        gen   = SCHEME_GENERATORS[i % len(SCHEME_GENERATORS)]
        scheme = SCHEME_NAMES[i % len(SCHEME_NAMES)]
        txns  = gen()
        rows.append(account_to_row(n_clean + i, txns, label=1, scheme=scheme))

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_and_save(
    out_dir: str = "aml/data",
    n_total: int = 10_000,
    suspicious_ratio: float = 0.05,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🏦 Generating {n_total:,} account histories (suspicious: {100*suspicious_ratio:.0f}%)...")
    df = generate_aml_dataset(n_total, suspicious_ratio, seed)

    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    for split_name, split_df in [
        ("train", df.iloc[:n_train]),
        ("val",   df.iloc[n_train:n_train + n_val]),
        ("test",  df.iloc[n_train + n_val:]),
    ]:
        path = os.path.join(out_dir, f"{split_name}.csv")
        split_df.to_csv(path, index=False)
        n_susp = split_df["label"].sum()
        print(f"  {split_name:5s}: {len(split_df):,} accounts | suspicious: {n_susp:,} ({100*n_susp/len(split_df):.1f}%)")

    print(f"✅ Saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-accounts",       type=int,   default=10_000)
    p.add_argument("--suspicious-ratio", type=float, default=0.05)
    p.add_argument("--out-dir",          default="aml/data")
    p.add_argument("--seed",             type=int,   default=42)
    args = p.parse_args()
    generate_and_save(args.out_dir, args.n_accounts, args.suspicious_ratio, args.seed)
