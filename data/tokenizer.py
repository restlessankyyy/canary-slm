"""
data/tokenizer.py — Domain-Specific Financial Transaction Tokenizer

Maps structured transaction features to integer token IDs.
Vocabulary of ~512 tokens covering:
  - Special tokens: [PAD], [CLS], [UNK], [SEP]
  - Amount bins (20 bins)
  - Merchant categories (30 types)
  - Country/region (80 regions)
  - Time of day (6 buckets)
  - Day of week (7)
  - Transaction frequency flags (5)
  - Card-not-present flag (2)
  - Currency (20)
  - Channel (online/POS/ATM/phone)
  - Velocity flags (unusual spend rate, etc.)
  - Device flags
"""

import json
import os
from typing import Dict, List, Optional, Tuple


# ── Vocabulary Definition ────────────────────────────────────────────────────

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[UNK]": 2,
    "[SEP]": 3,
}

AMOUNT_TOKENS = [f"AMT:{b}" for b in [
    "0-1", "1-5", "5-10", "10-20", "20-50", "50-100",
    "100-200", "200-500", "500-1k", "1k-2k", "2k-5k", "5k-10k",
    "10k-20k", "20k-50k", "50k-100k", "100k+",
    "ROUND",          # Round number (e.g. $500.00) — fraud signal
    "MICRO",          # Very small amount (<$1) — test transaction fraud signal
    "JUST_UNDER",     # Just under reporting threshold (e.g. $999.99)
    "SPLIT",          # Small split of larger amount
]]

MERCHANT_TOKENS = [f"MCC:{c}" for c in [
    "GROCERY", "GAS", "RESTAURANT", "RETAIL", "TRAVEL", "HOTEL",
    "AIRLINE", "ECOMMERCE", "DIGITAL_GOODS", "GAMBLING", "CRYPTO",
    "PHARMACY", "HEALTHCARE", "EDUCATION", "UTILITIES", "TELECOM",
    "AUTO", "ENTERTAINMENT", "LUXURY", "JEWELRY", "ELECTRONICS",
    "MONEY_TRANSFER", "ATM", "UNKNOWN", "CHARITY", "GOVERNMENT",
    "INSURANCE", "REAL_ESTATE", "WHOLESALE", "SUBSCRIPTION",
]]

COUNTRY_TOKENS = [f"CTRY:{c}" for c in [
    "US", "GB", "DE", "FR", "CA", "AU", "JP", "CN", "IN", "BR",
    "MX", "RU", "NG", "GH", "KE", "ZA", "PH", "ID", "TH", "VN",
    "UA", "PL", "RO", "BG", "TR", "EG", "PK", "BD", "IR", "IQ",
    "SY", "LY", "MM", "KP", "VE", "BY", "CU", "SD", "SO", "YE",
    "NL", "BE", "CH", "SE", "NO", "DK", "FI", "AT", "ES", "IT",
    "PT", "GR", "CZ", "HU", "SK", "HR", "RS", "AL", "MK", "KZ",
    "UZ", "AZ", "GE", "AM", "TM", "TJ", "KG", "MN", "NP", "LK",
    "HK", "SG", "MY", "TW", "KR", "NZ", "IL", "AE", "SA", "QA",
    "DOMESTIC",    # Same country as card issuer
    "FOREIGN",     # Different country from card issuer
    "HIGH_RISK",   # High-risk country flag
    "EMBARGOED",   # Sanctioned/embargoed country
]]

TIME_TOKENS = [f"TIME:{t}" for t in [
    "EARLY_MORNING",   # 00:00–05:59
    "MORNING",         # 06:00–10:59
    "MIDDAY",          # 11:00–13:59
    "AFTERNOON",       # 14:00–17:59
    "EVENING",         # 18:00–21:59
    "LATE_NIGHT",      # 22:00–23:59
]]

DAY_TOKENS = [f"DAY:{d}" for d in [
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
]]

VELOCITY_TOKENS = [f"VEL:{v}" for v in [
    "NORMAL",           # Normal spend velocity
    "ELEVATED",         # 2–3× normal rate
    "HIGH",             # 3–5× normal rate
    "VERY_HIGH",        # 5–10× normal rate
    "EXTREME",          # 10×+ normal rate — strong fraud signal
    "FIRST_USE",        # First transaction on card
    "DORMANT_REUSE",    # Card dormant >90d, suddenly active
    "RAPID_SUCCESSION", # Multiple txns within seconds/minutes
    "GEO_IMPOSSIBLE",   # Transaction impossible given travel time
]]

CHANNEL_TOKENS = [f"CH:{c}" for c in [
    "POS_CHIP",         # Card present, chip
    "POS_SWIPE",        # Card present, magstripe
    "POS_TAP",          # Contactless
    "ATM",              # ATM withdrawal
    "ONLINE",           # Card not present, online
    "PHONE",            # Card not present, phone order
    "RECURRING",        # Recurring charge
    "MOBILE_APP",       # Bank's own mobile app
]]

CURRENCY_TOKENS = [f"CUR:{c}" for c in [
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY",
    "INR", "BRL", "MXN", "RUB", "NGN", "GHS", "KES", "ZAR",
    "CRYPTO_BTC", "CRYPTO_ETH", "CRYPTO_OTHER", "OTHER",
]]

FLAG_TOKENS = [f"FLAG:{f}" for f in [
    "NEW_DEVICE",           # Transaction from new/unknown device
    "NEW_MERCHANT",         # First time at this merchant
    "BILLING_MISMATCH",     # Billing address doesn't match card
    "CVV_FAIL",             # CVV verification failed
    "PIN_FAIL",             # PIN entered incorrectly
    "FOREIGN_IP",           # IP in different country than card
    "TOR_VPN",              # Traffic from Tor/VPN
    "HIGH_DECLINE_RATE",    # Account has recent declines
    "RECENTLY_COMPROMISED", # Card recently reported lost/stolen
    "VERIFIED",             # All checks pass
    "3DS_PASSED",           # 3D Secure authentication passed
    "3DS_FAILED",           # 3D Secure failed
    "ADDRESS_VERIFIED",     # AVS match
    "NO_HISTORY",           # No transaction history for this pattern
]]


def _build_vocab() -> Dict[str, int]:
    """Construct full vocabulary mapping token → ID."""
    vocab: Dict[str, int] = {}
    vocab.update(SPECIAL_TOKENS)
    idx = len(SPECIAL_TOKENS)

    for token_group in [
        AMOUNT_TOKENS,
        MERCHANT_TOKENS,
        COUNTRY_TOKENS,
        TIME_TOKENS,
        DAY_TOKENS,
        VELOCITY_TOKENS,
        CHANNEL_TOKENS,
        CURRENCY_TOKENS,
        FLAG_TOKENS,
    ]:
        for tok in token_group:
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1

    return vocab


class FinancialTokenizer:
    """
    Converts structured transaction feature dicts into token ID sequences.

    Transaction dict keys (all optional — missing fields → [UNK]):
        amount          float   — transaction amount in USD
        merchant_cat    str     — merchant category code name
        country         str     — ISO country code or special value
        is_domestic     bool    — is the txn in card's home country
        hour            int     — hour of day (0–23)
        day_of_week     int     — day (0=Mon … 6=Sun)
        currency        str     — currency code
        channel         str     — transaction channel
        velocity        str     — velocity flag key
        flags           list    — list of flag names (e.g. ["NEW_DEVICE", "TOR_VPN"])
    """

    PAD_ID = 0
    CLS_ID = 1
    UNK_ID = 2
    SEP_ID = 3

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab if vocab is not None else _build_vocab()
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    # ── Serialization ────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FinancialTokenizer":
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab={k: int(v) for k, v in vocab.items()})

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _tok(self, token: str) -> int:
        return self.vocab.get(token, self.UNK_ID)

    def _amount_token(self, amount: float) -> str:
        """Bin a dollar amount into an amount token."""
        # Check structural properties first
        if amount < 1.0:
            return "AMT:MICRO"
        if amount > 0 and round(amount) == amount and amount % 100 == 0:
            return "AMT:ROUND"
        thresholds = [
            (1, "AMT:0-1"), (5, "AMT:1-5"), (10, "AMT:5-10"),
            (20, "AMT:10-20"), (50, "AMT:20-50"), (100, "AMT:50-100"),
            (200, "AMT:100-200"), (500, "AMT:200-500"), (1_000, "AMT:500-1k"),
            (2_000, "AMT:1k-2k"), (5_000, "AMT:2k-5k"), (10_000, "AMT:5k-10k"),
            (20_000, "AMT:10k-20k"), (50_000, "AMT:20k-50k"), (100_000, "AMT:50k-100k"),
        ]
        for threshold, token in thresholds:
            if amount < threshold:
                return token
        return "AMT:100k+"

    def _time_token(self, hour: int) -> str:
        if hour < 6:
            return "TIME:EARLY_MORNING"
        if hour < 11:
            return "TIME:MORNING"
        if hour < 14:
            return "TIME:MIDDAY"
        if hour < 18:
            return "TIME:AFTERNOON"
        if hour < 22:
            return "TIME:EVENING"
        return "TIME:LATE_NIGHT"

    def _country_token(self, country: str, is_domestic: Optional[bool]) -> str:
        if is_domestic is not None:
            return "CTRY:DOMESTIC" if is_domestic else "CTRY:FOREIGN"
        country = country.upper()
        tok = f"CTRY:{country}"
        return tok if tok in self.vocab else "CTRY:FOREIGN"

    # ── Core Tokenization ────────────────────────────────────────────────────

    def tokenize(self, transaction: Dict) -> List[int]:
        """
        Convert a transaction dict to a list of token IDs.
        Always starts with [CLS].
        """
        tokens = [self.CLS_ID]

        # Amount
        amount = transaction.get("amount")
        if amount is not None:
            tokens.append(self._tok(self._amount_token(float(amount))))

        # Merchant category
        mcc = transaction.get("merchant_cat", "").upper()
        tokens.append(self._tok(f"MCC:{mcc}") if f"MCC:{mcc}" in self.vocab else self.UNK_ID)

        # Country / domestic flag
        country = transaction.get("country", "")
        is_domestic = transaction.get("is_domestic")
        tokens.append(self._tok(self._country_token(country, is_domestic)))

        # Time of day
        hour = transaction.get("hour")
        if hour is not None:
            tokens.append(self._tok(self._time_token(int(hour))))

        # Day of week
        dow = transaction.get("day_of_week")
        if dow is not None:
            days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
            tokens.append(self._tok(f"DAY:{days[int(dow) % 7]}"))

        # Channel
        channel = transaction.get("channel", "").upper()
        tokens.append(self._tok(f"CH:{channel}") if f"CH:{channel}" in self.vocab else self.UNK_ID)

        # Currency
        currency = transaction.get("currency", "USD").upper()
        tokens.append(self._tok(f"CUR:{currency}") if f"CUR:{currency}" in self.vocab else self.UNK_ID)

        # Velocity flag
        velocity = transaction.get("velocity", "NORMAL").upper()
        tokens.append(self._tok(f"VEL:{velocity}") if f"VEL:{velocity}" in self.vocab else self.UNK_ID)

        # Boolean/flag list
        flags = transaction.get("flags", [])
        for flag in flags:
            tok = f"FLAG:{flag.upper()}"
            if tok in self.vocab:
                tokens.append(self._tok(tok))

        return tokens

    def encode(
        self,
        transaction: Dict,
        max_length: int = 64,
        padding: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Returns (input_ids, attention_mask) both of length max_length.
        Truncates if sequence is too long; pads with [PAD] if too short.
        """
        token_ids = self.tokenize(transaction)

        # Truncate
        token_ids = token_ids[:max_length]
        seq_len = len(token_ids)

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1] * seq_len

        if padding and seq_len < max_length:
            pad_len = max_length - seq_len
            token_ids += [self.PAD_ID] * pad_len
            attention_mask += [0] * pad_len

        return token_ids, attention_mask

    def decode(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs back to token strings."""
        return [self.id2token.get(tid, "[UNK]") for tid in token_ids if tid != self.PAD_ID]

    def __repr__(self):
        return f"FinancialTokenizer(vocab_size={self.vocab_size})"


# ── Default tokenizer instance ───────────────────────────────────────────────

_DEFAULT_TOKENIZER: Optional[FinancialTokenizer] = None


def get_tokenizer(vocab_path: Optional[str] = None) -> FinancialTokenizer:
    global _DEFAULT_TOKENIZER
    if vocab_path and os.path.exists(vocab_path):
        return FinancialTokenizer.load(vocab_path)
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = FinancialTokenizer()
    return _DEFAULT_TOKENIZER


if __name__ == "__main__":
    tok = FinancialTokenizer()
    print(f"Vocab size: {tok.vocab_size}")

    sample = {
        "amount": 2500.00,
        "merchant_cat": "CRYPTO",
        "country": "NG",
        "is_domestic": False,
        "hour": 2,
        "day_of_week": 5,  # Saturday
        "channel": "ONLINE",
        "currency": "USD",
        "velocity": "EXTREME",
        "flags": ["NEW_DEVICE", "TOR_VPN", "FOREIGN_IP"],
    }

    ids, mask = tok.encode(sample, max_length=64)
    print(f"\nSuspicious transaction tokens ({sum(mask)} real tokens):")
    print(tok.decode(ids))
    print(f"IDs: {[i for i, m in zip(ids, mask) if m]}")
