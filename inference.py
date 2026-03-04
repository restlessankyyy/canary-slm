"""
inference.py — Inference Engine for Finance Fraud Detection SLM

Provides a clean API for:
  1. Loading a trained model from checkpoint
  2. Classifying a single transaction dict
  3. Batch inference on a list of transactions
  4. Explanation: which tokens drove the prediction (attention-based)
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from config import get_default_config
from model import FraudTransformer
from data.tokenizer import FinancialTokenizer, get_tokenizer


# ── Fraud risk labels ────────────────────────────────────────────────────────

RISK_LEVELS = [
    (0.90, "🚨 CRITICAL RISK",   "Block immediately"),
    (0.75, "🔴 HIGH RISK",       "Decline & alert customer"),
    (0.50, "🟠 MEDIUM RISK",     "Flag for manual review"),
    (0.25, "🟡 LOW RISK",        "Monitor"),
    (0.00, "🟢 LEGITIMATE",      "Approve"),
]


def get_risk_label(fraud_prob: float) -> Tuple[str, str]:
    """Return (risk_label, recommended_action) for a fraud probability."""
    for threshold, label, action in RISK_LEVELS:
        if fraud_prob >= threshold:
            return label, action
    return "🟢 LEGITIMATE", "Approve"


# ── Model loader ─────────────────────────────────────────────────────────────

class FraudDetector:
    """
    High-level inference wrapper for the FraudTransformer model.

    Usage:
        detector = FraudDetector.from_checkpoint("checkpoints/best_model.pt")
        result = detector.predict({...transaction...})
    """

    def __init__(
        self,
        model: FraudTransformer,
        tokenizer: FinancialTokenizer,
        max_seq_len: int = 64,
        device: torch.device = None,
        threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        self.threshold = threshold
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = "checkpoints/best_model.pt",
        vocab_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> "FraudDetector":
        """Load a FraudDetector from a saved checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "  Run: python train.py   to train the model first."
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Reconstruct config from checkpoint
        cfg = get_default_config()
        if "config" in ckpt and "model" in ckpt["config"]:
            for k, v in ckpt["config"]["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)

        model = FraudTransformer(cfg.model)
        model.load_state_dict(ckpt["model_state"])

        tokenizer_path = vocab_path or "data/vocab.json"
        tokenizer = get_tokenizer(tokenizer_path if os.path.exists(tokenizer_path) else None)

        if device is None:
            if torch.cuda.is_available():          dev = torch.device("cuda")
            elif torch.backends.mps.is_available(): dev = torch.device("mps")
            else:                                   dev = torch.device("cpu")
        else:
            dev = torch.device(device)

        return cls(model, tokenizer, cfg.model.max_seq_len, dev, threshold)

    # ── Core inference ────────────────────────────────────────────────────────

    def _encode_batch(self, transactions: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of transaction dicts → (input_ids, attention_mask)."""
        all_ids, all_masks = [], []
        for txn in transactions:
            ids, mask = self.tokenizer.encode(txn, max_length=self.max_seq_len, padding=True)
            all_ids.append(ids)
            all_masks.append(mask)
        input_ids     = torch.tensor(all_ids,   dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(all_masks, dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    def predict(self, transaction: Dict) -> Dict:
        """
        Classify a single transaction.

        Returns a result dict with:
            fraud_probability  float   — P(fraud)  [0–1]
            is_fraud           bool
            risk_label         str     — Human-readable risk level
            action             str     — Recommended action
            tokens             list    — Token strings used as input
            confidence         str     — "High" / "Medium" / "Low"
        """
        return self.predict_batch([transaction])[0]

    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Classify a list of transactions."""
        input_ids, attention_mask = self._encode_batch(transactions)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs  = F.softmax(logits, dim=-1)

        results = []
        for i, txn in enumerate(transactions):
            fraud_prob = float(probs[i, 1].item())
            legit_prob = float(probs[i, 0].item())
            is_fraud   = fraud_prob >= self.threshold

            risk_label, action = get_risk_label(fraud_prob)

            # Decode tokens used (for explanation)
            ids, mask = self.tokenizer.encode(txn, max_length=self.max_seq_len, padding=False)
            tokens = self.tokenizer.decode(ids)

            # Confidence: distance from 0.5
            dist = abs(fraud_prob - 0.5)
            confidence = "High" if dist > 0.35 else ("Medium" if dist > 0.15 else "Low")

            # Key risk factors (tokens associated with fraud)
            risk_tokens = [
                t for t in tokens
                if any(kw in t for kw in [
                    "HIGH_RISK", "FOREIGN", "TOR_VPN", "FOREIGN_IP", "EXTREME",
                    "RAPID", "DORMANT", "CVV_FAIL", "BILLING_MISMATCH", "GEO_",
                    "NIGHT", "EARLY_MORNING", "LATE_NIGHT", "CRYPTO", "MONEY_TRANSFER",
                    "GAMBLING", "AMT:10k", "AMT:20k", "AMT:50k", "AMT:100k",
                    "NEW_DEVICE", "COMPROMISED", "EMBARGOED", "MICRO", "JUST_UNDER",
                ])
            ]

            results.append({
                "fraud_probability": round(fraud_prob, 4),
                "legit_probability": round(legit_prob, 4),
                "is_fraud":          is_fraud,
                "risk_label":        risk_label,
                "action":            action,
                "confidence":        confidence,
                "tokens":            tokens,
                "risk_factors":      risk_tokens,
                "threshold":         self.threshold,
            })

        return results


# ── Convenience function ─────────────────────────────────────────────────────

_DETECTOR: Optional[FraudDetector] = None


def get_detector(
    checkpoint_path: str = "checkpoints/best_model.pt",
    **kwargs,
) -> FraudDetector:
    """Singleton accessor for the detector (avoids reloading model repeatedly)."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FraudDetector.from_checkpoint(checkpoint_path, **kwargs)
    return _DETECTOR


def predict_transaction(txn: Dict, checkpoint: str = "checkpoints/best_model.pt") -> Dict:
    """Quick one-shot prediction (loads model on first call, reuses after)."""
    return get_detector(checkpoint).predict(txn)


if __name__ == "__main__":
    print("Loading model...")
    detector = FraudDetector.from_checkpoint()

    samples = [
        {
            "name": "Normal grocery shopping",
            "txn": {
                "amount": 47.50,
                "merchant_cat": "GROCERY",
                "country": "US",
                "is_domestic": True,
                "hour": 14,
                "day_of_week": 1,
                "channel": "POS_CHIP",
                "currency": "USD",
                "velocity": "NORMAL",
                "flags": ["VERIFIED"],
            },
        },
        {
            "name": "Suspicious crypto purchase — foreign, night, Tor",
            "txn": {
                "amount": 9999.99,
                "merchant_cat": "CRYPTO",
                "country": "NG",
                "is_domestic": False,
                "hour": 3,
                "day_of_week": 6,
                "channel": "ONLINE",
                "currency": "CRYPTO_BTC",
                "velocity": "EXTREME",
                "flags": ["NEW_DEVICE", "TOR_VPN", "FOREIGN_IP"],
            },
        },
        {
            "name": "Card testing micro-transaction",
            "txn": {
                "amount": 0.99,
                "merchant_cat": "DIGITAL_GOODS",
                "country": "PH",
                "is_domestic": False,
                "hour": 2,
                "day_of_week": 5,
                "channel": "ONLINE",
                "currency": "USD",
                "velocity": "RAPID_SUCCESSION",
                "flags": ["NEW_DEVICE", "NO_HISTORY"],
            },
        },
    ]

    print(f"\n{'='*65}")
    for s in samples:
        result = detector.predict(s["txn"])
        print(f"\n📋 {s['name']}")
        print(f"   {result['risk_label']} — {result['action']}")
        print(f"   Fraud probability: {result['fraud_probability']:.2%}  (confidence: {result['confidence']})")
        if result['risk_factors']:
            print(f"   Risk factors: {', '.join(result['risk_factors'][:4])}")
    print(f"{'='*65}")
