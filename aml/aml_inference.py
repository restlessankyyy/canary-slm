"""
aml/aml_inference.py — AML Detector: score an account's recent transaction history

Usage:
    from aml.aml_inference import AMLDetector
    detector = AMLDetector.from_checkpoint("checkpoints/aml_best.pt")
    result = detector.score_account(transactions)  # list of 30 dicts
    print(result["risk_label"])   # 🚨 HIGH AML RISK
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional

from model import FraudTransformer
from aml.aml_config import get_aml_config
from aml.aml_dataset import encode_account
import pandas as pd


AML_RISK_LEVELS = [
    (0.85, "🚨 HIGH AML RISK",    "File SAR immediately",          "RED"),
    (0.60, "🟠 MEDIUM AML RISK",  "Enhanced due diligence required", "AMBER"),
    (0.35, "🟡 LOW AML RISK",     "Monitor closely",               "YELLOW"),
    (0.00, "🟢 CLEAN",            "No action required",            "GREEN"),
]

SCHEME_DESCRIPTIONS = {
    "structuring":    "Repeated deposits just under the $10K reporting threshold",
    "layering":       "Rapid movement of funds through multiple accounts/countries",
    "smurfing":       "Large sum split into many small transfers to evade detection",
    "dormant_burst":  "Dormant account suddenly activated with large transactions",
    "round_trip":     "Funds exported and returned via a different route",
}


class AMLDetector:
    """Scores an account's recent transaction history for AML risk."""

    def __init__(self, model: FraudTransformer, max_seq_len: int = 160,
                 device: torch.device = None, threshold: float = 0.5):
        self.model = model
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        self.threshold = threshold
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = "checkpoints/aml_best.pt",
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> "AMLDetector":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"AML checkpoint not found: {checkpoint_path}\n"
                "  Run: python aml/train_aml.py"
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = get_aml_config()
        if "config" in ckpt and "model" in ckpt["config"]:
            for k, v in ckpt["config"]["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        model = FraudTransformer(cfg.model)
        model.load_state_dict(ckpt["model_state"])

        if device is None:
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)

        return cls(model, cfg.model.max_seq_len, dev, threshold)

    def _txns_to_row(self, transactions: List[Dict]) -> pd.Series:
        """Convert a list of transaction dicts to a flat pandas Series."""
        row = {}
        for i, t in enumerate(transactions[:30]):
            row[f"t{i}_amount"]    = t.get("amount",    0)
            row[f"t{i}_direction"] = t.get("direction", "OUT")
            row[f"t{i}_country"]   = t.get("country",   "US")
            row[f"t{i}_hour"]      = t.get("hour",      12)
            row[f"t{i}_gap_hours"] = t.get("gap_hours", 24)
        return pd.Series(row)

    def score_account(self, transactions: List[Dict]) -> Dict:
        """
        Score an account's transaction history.

        Args:
            transactions: List of up to 30 transaction dicts, each with:
                amount (float), direction ("IN"/"OUT"), country (str),
                hour (int 0-23), gap_hours (float — time since previous txn)

        Returns:
            dict with risk_label, aml_probability, is_suspicious, signals
        """
        # Pad to 30 if shorter
        while len(transactions) < 30:
            transactions.append({"amount": 50, "direction": "OUT",
                                  "country": "US", "hour": 12, "gap_hours": 48})

        row = self._txns_to_row(transactions)
        ids, mask = encode_account(row, self.max_seq_len)

        input_ids = torch.tensor([ids],  dtype=torch.long,  device=self.device)
        attn_mask = torch.tensor([mask], dtype=torch.long,  device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attn_mask)
            probs  = F.softmax(logits, dim=-1)

        aml_prob = float(probs[0, 1].item())
        is_susp  = aml_prob >= self.threshold

        # Risk label
        risk_label, action, colour = "🟢 CLEAN", "No action required", "GREEN"
        for threshold, label, act, col in AML_RISK_LEVELS:
            if aml_prob >= threshold:
                risk_label, action, colour = label, act, col
                break

        # Signal extraction (rule-based on the sequence)
        signals = self._extract_signals(transactions)

        return {
            "aml_probability": round(aml_prob, 4),
            "is_suspicious":   is_susp,
            "risk_label":      risk_label,
            "action":          action,
            "colour":          colour,
            "signals":         signals,
            "n_transactions":  len(transactions),
            "threshold":       self.threshold,
        }

    def _extract_signals(self, transactions: List[Dict]) -> List[str]:
        """Extract human-readable AML warning signals from the sequence."""
        signals = []
        amounts  = [t.get("amount",    0)    for t in transactions]
        gaps     = [t.get("gap_hours", 24)   for t in transactions]
        dirs     = [t.get("direction", "OUT") for t in transactions]
        foreign  = [t.get("country",   "US") not in
                    {"US", "CA", "GB", "DE", "FR", "AU", "JP", "SG", "DOMESTIC"}
                    for t in transactions]

        # Structuring: multiple amounts in $8k-$9.9k
        struct = sum(1 for a in amounts if 8_000 <= a <= 9_900)
        if struct >= 3:
            signals.append(f"Structuring: {struct} deposits in $8k-$9.9k range")

        # Rapid movement: many gaps < 2h
        rapid = sum(1 for g in gaps if g < 2)
        if rapid >= 3:
            signals.append(f"Rapid movement: {rapid} transactions within 2 hours of each other")

        # Foreign exposure
        n_foreign = sum(foreign)
        if n_foreign >= 5:
            signals.append(f"High foreign exposure: {n_foreign}/{len(transactions)} foreign transactions")

        # Dormant activation: long gaps followed by rapid burst
        if gaps and max(gaps) > 500 and min(gaps[:5]) < 5:
            signals.append("Dormant account activation: sudden burst after long quiet period")

        # Large round-trip: big OUT followed by big IN
        for i in range(len(transactions) - 1):
            if dirs[i] == "OUT" and dirs[i+1] == "IN":
                if amounts[i] > 20_000 and amounts[i+1] > amounts[i] * 0.85:
                    signals.append(f"Potential round-trip: ${amounts[i]:,.0f} out, ${amounts[i+1]:,.0f} back")
                    break

        # Whale transactions
        whale = sum(1 for a in amounts if a > 50_000)
        if whale >= 3:
            signals.append(f"High-value concentration: {whale} transactions over $50k")

        return signals or ["No specific AML signals detected"]

    def score_batch(self, accounts: List[List[Dict]]) -> List[Dict]:
        """Score multiple accounts at once."""
        return [self.score_account(txns) for txns in accounts]
