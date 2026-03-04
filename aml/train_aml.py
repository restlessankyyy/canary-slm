"""
aml/train_aml.py — Train the AML Transformer

Reuses the FraudTransformer architecture from model.py.
The only difference is the input: 30 transactions per account
instead of a single transaction.

Usage:
    python aml/train_aml.py                    # default 20 epochs
    python aml/train_aml.py --epochs 5         # quick smoke test
    python aml/train_aml.py --n-accounts 50000 # more data
"""

import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model import FraudTransformer
from train import run_epoch, set_seed
from aml.aml_config import get_aml_config
from aml.aml_dataset import build_aml_dataloaders
from aml.generate_aml_data import generate_and_save


def train_aml(args):
    set_seed(42)
    cfg = get_aml_config()
    cfg.training.num_epochs   = args.epochs
    cfg.training.batch_size   = args.batch_size
    cfg.training.fraud_weight = args.suspicious_weight
    cfg.ensure_dirs()

    # ── Generate data if needed ───────────────────────────────────────────────
    if not os.path.exists(os.path.join(cfg.training.data_dir, "train.csv")):
        print("📊 Generating synthetic AML data...")
        generate_and_save(
            out_dir=cfg.training.data_dir,
            n_total=args.n_accounts,
            suspicious_ratio=0.05,
        )

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n🖥️  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n📦 Loading AML datasets...")
    train_loader, val_loader, _ = build_aml_dataloaders(
        data_dir=cfg.training.data_dir,
        batch_size=cfg.training.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n🧠 AML Model:")
    model = FraudTransformer(cfg.model).to(device)
    print(model.summary())

    # ── Loss ──────────────────────────────────────────────────────────────────
    weight = torch.tensor([1.0, cfg.training.fraud_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print(f"   Suspicious weight: {cfg.training.fraud_weight}×")

    # ── Optimizer + Schedule ──────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate,
                      weight_decay=cfg.training.weight_decay,
                      betas=(cfg.training.beta1, cfg.training.beta2))
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                 total_iters=cfg.training.warmup_epochs),
        CosineAnnealingLR(optimizer,
                          T_max=cfg.training.num_epochs - cfg.training.warmup_epochs,
                          eta_min=cfg.training.learning_rate * 0.01),
    ], milestones=[cfg.training.warmup_epochs])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1 = 0.0
    history = []
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "aml_best.pt")

    print(f"\n🚀 Training AML model for {cfg.training.num_epochs} epochs")
    print(f"{'='*65}")

    for epoch in range(cfg.training.num_epochs):
        t0 = time.time()
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device, cfg, is_train=True)
        val_loss,   val_m   = run_epoch(model, val_loader,   criterion, None,      device, cfg, is_train=False)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        history.append({"epoch": epoch + 1, "lr": lr,
                         "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_m.items()}})

        print(
            f"Epoch {epoch+1:03d}/{cfg.training.num_epochs} │ LR {lr:.2e} │ "
            f"Loss {train_loss:.4f}→{val_loss:.4f} │ "
            f"Val F1 {val_m['f1']:.4f} │ AUC {val_m['auc_roc']:.4f} │ "
            f"Rec {val_m['recall']:.4f} │ Prec {val_m['precision']:.4f} │ {elapsed:.1f}s"
        )

        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "val_metrics": val_m,
                "config": {"model": vars(cfg.model), "training": vars(cfg.training)}}
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(ckpt, ckpt_path)
            print(f"  ✅ Best F1: {best_f1:.4f} — saved.")

    os.makedirs(cfg.training.log_dir, exist_ok=True)
    with open(os.path.join(cfg.training.log_dir, "aml_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*65}")
    print(f"✅ Done. Best val F1: {best_f1:.4f} | Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train AML Transformer")
    p.add_argument("--epochs",             type=int,   default=20)
    p.add_argument("--batch-size",         type=int,   default=128)
    p.add_argument("--n-accounts",         type=int,   default=10_000)
    p.add_argument("--suspicious-weight",  type=float, default=20.0)
    args = p.parse_args()
    train_aml(args)
