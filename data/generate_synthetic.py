"""
data/generate_synthetic.py — Synthetic Financial Transaction Dataset Generator

Generates statistically realistic fraud/legitimate transaction data.

Legitimate transaction profile:
 - Normal amounts ($10–$500), known domestic merchants
 - Business hours, home country, regular velocity

Fraud transaction profiles (multiple):
 1. Card Testing   — micro-amounts, online, rapid succession
 2. Account Takeover — large amounts, foreign country, night
 3. CNP Fraud      — online, new device, foreign IP
 4. Skimming       — POS, cloned card signals
 5. Money Mule     — crypto/money transfer, high amount
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# ── Fraud profile definitions ────────────────────────────────────────────────

LEGIT_MERCHANTS = [
    "GROCERY", "GAS", "RESTAURANT", "RETAIL", "PHARMACY",
    "UTILITIES", "TELECOM", "INSURANCE", "SUBSCRIPTION",
    "ENTERTAINMENT", "HEALTHCARE", "AUTO",
]

FRAUD_HIGH_RISK_MERCHANTS = [
    "CRYPTO", "MONEY_TRANSFER", "GAMBLING", "JEWELRY",
    "ELECTRONICS", "LUXURY", "DIGITAL_GOODS",
]

DOMESTIC_COUNTRIES = ["US", "US", "US", "GB", "DE", "FR", "CA", "AU"]

HIGH_RISK_COUNTRIES = [
    "NG", "GH", "KE", "PH", "ID", "RU", "UA", "RO",
    "CN", "BR", "MX", "VN", "TR", "PK", "BD",
]

CURRENCIES = ["USD"] * 8 + ["EUR", "GBP", "CAD", "AUD", "JPY", "CNY", "INR", "BRL", "NGN", "GHS"]

CHANNELS_LEGIT = ["POS_CHIP", "POS_TAP", "ONLINE", "MOBILE_APP", "RECURRING"]
CHANNELS_FRAUD = ["ONLINE", "PHONE", "ATM", "POS_SWIPE"]

VELOCITIES_LEGIT = ["NORMAL", "NORMAL", "NORMAL", "ELEVATED"]
VELOCITIES_FRAUD = ["HIGH", "VERY_HIGH", "EXTREME", "RAPID_SUCCESSION", "DORMANT_REUSE"]

FLAGS_LEGIT = [
    ["VERIFIED"],
    ["VERIFIED", "3DS_PASSED"],
    ["VERIFIED", "ADDRESS_VERIFIED"],
    ["3DS_PASSED", "ADDRESS_VERIFIED"],
    [],
]

FLAGS_FRAUD = [
    ["NEW_DEVICE", "TOR_VPN", "FOREIGN_IP"],
    ["NEW_DEVICE", "FOREIGN_IP"],
    ["BILLING_MISMATCH", "FOREIGN_IP"],
    ["CVV_FAIL", "NEW_DEVICE"],
    ["HIGH_DECLINE_RATE", "NEW_DEVICE"],
    ["TOR_VPN", "BILLING_MISMATCH"],
    ["RECENTLY_COMPROMISED"],
    ["GEO_IMPOSSIBLE", "FOREIGN_IP"],
    ["NO_HISTORY", "NEW_DEVICE", "TOR_VPN"],
    ["CVV_FAIL", "BILLING_MISMATCH"],
]


def _rand_legit(rng: np.random.Generator) -> Dict:
    """Generate one legitimate transaction."""
    hour = int(rng.integers(7, 22))        # Business hours skewed
    day = int(rng.integers(0, 5))          # Weekdays more common

    # Amount: log-normal, most transactions $5–$500
    amount = round(float(np.exp(rng.normal(4.0, 1.2))), 2)
    amount = max(1.0, min(amount, 5_000.0))

    is_domestic = rng.random() < 0.92
    country = random.choice(DOMESTIC_COUNTRIES) if is_domestic else random.choice(HIGH_RISK_COUNTRIES[:5])

    return {
        "amount": amount,
        "merchant_cat": random.choice(LEGIT_MERCHANTS),
        "country": country,
        "is_domestic": is_domestic,
        "hour": hour,
        "day_of_week": day,
        "channel": random.choice(CHANNELS_LEGIT),
        "currency": "USD" if is_domestic else random.choice(CURRENCIES),
        "velocity": random.choice(VELOCITIES_LEGIT),
        "flags": random.choice(FLAGS_LEGIT),
        "label": 0,
    }


def _rand_fraud(rng: np.random.Generator) -> Dict:
    """Generate one fraudulent transaction using one of several fraud profiles."""
    profile = rng.integers(0, 5)

    if profile == 0:
        # Card Testing: micro amounts, online, rapid succession
        return {
            "amount": round(float(rng.uniform(0.01, 1.50)), 2),
            "merchant_cat": random.choice(["DIGITAL_GOODS", "GAMBLING", "CRYPTO"]),
            "country": random.choice(HIGH_RISK_COUNTRIES),
            "is_domestic": False,
            "hour": int(rng.integers(0, 6)),
            "day_of_week": int(rng.integers(0, 7)),
            "channel": "ONLINE",
            "currency": random.choice(["USD", "EUR", "OTHER"]),
            "velocity": "RAPID_SUCCESSION",
            "flags": random.choice([
                ["NEW_DEVICE", "TOR_VPN"],
                ["NEW_DEVICE", "FOREIGN_IP", "HIGH_DECLINE_RATE"],
                ["TOR_VPN", "NO_HISTORY"],
            ]),
            "label": 1,
        }

    elif profile == 1:
        # Account Takeover: large amount, foreign, night, new device
        amount = round(float(rng.uniform(1_500, 15_000)), 2)
        # Make some just-under round numbers ($999.99 style)
        if rng.random() < 0.3:
            amount = round(random.choice([999.99, 4999.99, 9999.99, 2999.99]), 2)
        return {
            "amount": amount,
            "merchant_cat": random.choice(FRAUD_HIGH_RISK_MERCHANTS),
            "country": random.choice(HIGH_RISK_COUNTRIES),
            "is_domestic": False,
            "hour": int(rng.integers(0, 5)),
            "day_of_week": int(rng.integers(0, 7)),
            "channel": random.choice(["ONLINE", "PHONE"]),
            "currency": random.choice(["NGN", "GHS", "CNY", "BRL", "OTHER"]),
            "velocity": random.choice(["EXTREME", "VERY_HIGH", "DORMANT_REUSE"]),
            "flags": random.choice(FLAGS_FRAUD),
            "label": 1,
        }

    elif profile == 2:
        # CNP (Card Not Present) Fraud: online, foreign IP, billing mismatch
        return {
            "amount": round(float(rng.uniform(100, 3_000)), 2),
            "merchant_cat": random.choice(["ECOMMERCE", "ELECTRONICS", "LUXURY", "JEWELRY"]),
            "country": random.choice(HIGH_RISK_COUNTRIES),
            "is_domestic": rng.random() < 0.3,
            "hour": int(rng.integers(18, 24)),
            "day_of_week": int(rng.integers(5, 7)),   # Weekend
            "channel": "ONLINE",
            "currency": random.choice(CURRENCIES),
            "velocity": random.choice(["HIGH", "ELEVATED"]),
            "flags": random.choice([
                ["BILLING_MISMATCH", "FOREIGN_IP"],
                ["BILLING_MISMATCH", "NEW_DEVICE", "FOREIGN_IP"],
                ["CVV_FAIL", "BILLING_MISMATCH"],
            ]),
            "label": 1,
        }

    elif profile == 3:
        # ATM/Skimming: cloned card, foreign ATM
        amount = round(float(rng.choice([200, 300, 400, 500, 1000])), 2)
        return {
            "amount": amount,
            "merchant_cat": "ATM",
            "country": random.choice(HIGH_RISK_COUNTRIES),
            "is_domestic": False,
            "hour": int(rng.integers(22, 24)),
            "day_of_week": int(rng.integers(0, 7)),
            "channel": "ATM",
            "currency": random.choice(["NGN", "GHS", "BRL", "MXN"]),
            "velocity": random.choice(["HIGH", "RAPID_SUCCESSION"]),
            "flags": random.choice([
                ["GEO_IMPOSSIBLE"],
                ["RECENTLY_COMPROMISED", "GEO_IMPOSSIBLE"],
                ["PIN_FAIL", "GEO_IMPOSSIBLE"],
            ]),
            "label": 1,
        }

    else:
        # Money Mule / Crypto: large transfer, crypto, high velocity
        amount = round(float(rng.uniform(5_000, 100_000)), 2)
        return {
            "amount": amount,
            "merchant_cat": random.choice(["CRYPTO", "MONEY_TRANSFER"]),
            "country": random.choice(HIGH_RISK_COUNTRIES),
            "is_domestic": False,
            "hour": int(rng.integers(0, 24)),
            "day_of_week": int(rng.integers(0, 7)),
            "channel": random.choice(["ONLINE", "MOBILE_APP"]),
            "currency": random.choice(["CRYPTO_BTC", "CRYPTO_ETH", "OTHER"]),
            "velocity": "EXTREME",
            "flags": random.choice([
                ["TOR_VPN", "NO_HISTORY"],
                ["NEW_DEVICE", "TOR_VPN", "HIGH_DECLINE_RATE"],
            ]),
            "label": 1,
        }


def _txn_to_row(txn: Dict) -> Dict:
    """Flatten transaction dict into a CSV-friendly row."""
    return {
        "amount": txn["amount"],
        "merchant_cat": txn["merchant_cat"],
        "country": txn["country"],
        "is_domestic": int(txn["is_domestic"]),
        "hour": txn["hour"],
        "day_of_week": txn["day_of_week"],
        "channel": txn["channel"],
        "currency": txn["currency"],
        "velocity": txn["velocity"],
        "flags": "|".join(txn["flags"]) if txn["flags"] else "",
        "label": txn["label"],
    }


def generate_dataset(
    n_total: int,
    fraud_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n_fraud = int(n_total * fraud_ratio)
    n_legit = n_total - n_fraud

    print(f"  Generating {n_legit:,} legitimate + {n_fraud:,} fraud = {n_total:,} total")

    rows = []
    for _ in range(n_legit):
        rows.append(_txn_to_row(_rand_legit(rng)))
    for _ in range(n_fraud):
        rows.append(_txn_to_row(_rand_fraud(rng)))

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud transaction data")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val",   type=int, default=10_000)
    parser.add_argument("--n-test",  type=int, default=10_000)
    parser.add_argument("--fraud-ratio", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = [
        ("train", args.n_train, args.seed),
        ("val",   args.n_val,   args.seed + 1),
        ("test",  args.n_test,  args.seed + 2),
    ]

    for split, n, seed in splits:
        print(f"\n[{split.upper()}]")
        df = generate_dataset(n, fraud_ratio=args.fraud_ratio, seed=seed)
        path = os.path.join(args.out_dir, f"{split}.csv")
        df.to_csv(path, index=False)
        fraud_count = df["label"].sum()
        print(f"  Saved {len(df):,} rows → {path}")
        print(f"  Fraud: {fraud_count:,} ({100*fraud_count/len(df):.1f}%)")

    print("\n✅ Synthetic data generation complete.")


if __name__ == "__main__":
    main()
