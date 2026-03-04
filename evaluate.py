"""
evaluate.py — Full Evaluation Script for Finance Fraud Detection SLM

Loads the best checkpoint and computes:
  - Accuracy, Precision, Recall, F1 (fraud class)
  - AUC-ROC
  - Confusion Matrix
  - Full Classification Report
  - Threshold sweep (find optimal decision threshold)
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from config import get_default_config
from model import FraudTransformer
from data.dataset import FraudDataset
from data.tokenizer import get_tokenizer


def evaluate(
    checkpoint_path: str = "checkpoints/best_model.pt",
    data_path: str = "data/test.csv",
    batch_size: int = 512,
    threshold: float = 0.5,
) -> dict:
    """
    Run full evaluation on test set.

    Args:
        checkpoint_path: Path to .pt checkpoint
        data_path:       CSV test file
        batch_size:      Inference batch size
        threshold:       Decision threshold for fraud (default 0.5)

    Returns:
        metrics dict
    """
    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Re-build config from checkpoint
    cfg = get_default_config()
    if "config" in ckpt and "model" in ckpt["config"]:
        saved_model_cfg = ckpt["config"]["model"]
        for k, v in saved_model_cfg.items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)

    model = FraudTransformer(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"   Checkpoint from epoch {ckpt.get('epoch', '?') + 1}")
    print(f"   Val metrics: {ckpt.get('val_metrics', {})}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    tokenizer = get_tokenizer()
    dataset = FraudDataset(data_path, tokenizer=tokenizer, max_seq_len=cfg.model.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_labels, all_probs = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(input_ids, attention_mask)
            probs  = F.softmax(logits, dim=-1)[:, 1]
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {data_path}")
    print(f"{'='*60}")
    print(f"  Decision threshold: {threshold:.2f}")
    print(f"  Total samples:      {len(all_labels):,}")
    print(f"  Fraud samples:      {all_labels.sum():,} ({100*all_labels.mean():.1f}%)")

    report = classification_report(
        all_labels, all_preds,
        target_names=["Legitimate", "Fraud"],
        digits=4,
    )
    print(f"\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    auc_roc = roc_auc_score(all_labels, all_probs)
    auc_pr  = average_precision_score(all_labels, all_probs)

    print("  Confusion Matrix:")
    print(f"  {'':20} Predicted Legit  Predicted Fraud")
    print(f"  {'Actual Legit':20} {tn:>15,}  {fp:>15,}")
    print(f"  {'Actual Fraud':20} {fn:>15,}  {tp:>15,}")

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\n  AUC-ROC:          {auc_roc:.4f}")
    print(f"  AUC-PR:           {auc_pr:.4f}")
    print(f"  False Positive Rate: {fpr:.4f} ({fp:,} false alarms)")
    print(f"  True Positive Rate:  {tp/(tp+fn):.4f} ({tp:,} fraud caught)")

    # ── Optimal threshold sweep ───────────────────────────────────────────────
    print("\n  Threshold Sweep (maximize F1):")
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(all_labels, all_probs)
    # Compute F1 at each threshold
    f1_arr = 2 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-8)
    best_idx = f1_arr.argmax()
    best_thresh = thresh_arr[best_idx] if best_idx < len(thresh_arr) else 0.5
    best_f1 = f1_arr[best_idx]

    print(f"  Optimal threshold: {best_thresh:.4f}")
    print(f"  Optimal F1:        {best_f1:.4f}")
    print(f"  @ threshold: Precision={prec_arr[best_idx]:.4f}, Recall={rec_arr[best_idx]:.4f}")

    results = {
        "threshold": threshold,
        "accuracy":  float((all_labels == all_preds).mean()),
        "precision": float(prec_arr[best_idx] if best_idx < len(prec_arr) else 0),
        "recall":    float(rec_arr[best_idx]  if best_idx < len(rec_arr)  else 0),
        "f1":        float(best_f1),
        "auc_roc":   float(auc_roc),
        "auc_pr":    float(auc_pr),
        "false_positive_rate": float(fpr),
        "true_positive_rate":  float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "optimal_threshold": float(best_thresh),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # Save results
    out_path = "logs/eval_results.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print(f"{'='*60}\n")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Finance Fraud Detection SLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data",       type=str, default="data/test.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold",  type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
