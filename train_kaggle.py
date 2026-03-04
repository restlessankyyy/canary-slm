"""
train_kaggle.py — Training Script for the Kaggle Credit Card Fraud Dataset

Extends train.py to use:
  - KaggleFraudDataset (extended vocab with 140 PCA-bin tokens)
  - Updated model config (larger vocab to cover PCA tokens)
  - Kaggle-specific class weighting (fraud is ~0.17%, need very high weight)

Usage:
    # 1. Download & preprocess Kaggle data first:
    python setup_kaggle.py --csv /path/to/creditcard.csv

    # 2. Train:
    python train_kaggle.py --epochs 20

    # 3. Evaluate:
    python evaluate.py --checkpoint checkpoints/kaggle_best.pt --data data/kaggle_test.csv
"""

import os
import sys
import time
import argparse
import random
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from config import Config, ModelConfig, TrainingConfig, get_default_config
from model import FraudTransformer
from data.kaggle_dataset import build_kaggle_dataloaders, build_kaggle_tokenizer
from train import set_seed, compute_metrics, run_epoch   # reuse helpers


def build_kaggle_config(args) -> Config:
    """Build a Config tweaked for the Kaggle dataset."""
    cfg = get_default_config()

    # The Kaggle tokenizer has a larger vocab (standard + 140 PCA tokens)
    kaggle_tokenizer = build_kaggle_tokenizer()
    cfg.model.vocab_size = kaggle_tokenizer.vocab_size
    print(f"  Extended vocab size: {cfg.model.vocab_size} tokens")

    # Training
    cfg.training.data_dir         = args.data_dir
    cfg.training.checkpoint_dir   = args.checkpoint_dir
    cfg.training.num_epochs       = args.epochs
    cfg.training.batch_size       = args.batch_size
    cfg.training.learning_rate    = args.lr
    cfg.training.device           = args.device

    # Kaggle fraud rate is ~0.172%, so we need a much higher fraud weight
    # Heuristic: set it to ~(n_legit / n_fraud) / 5  ≈ 100
    cfg.training.fraud_weight = args.fraud_weight

    cfg.training.experiment_name = "kaggle"
    return cfg


def train_kaggle(args):
    set_seed(42)

    cfg = build_kaggle_config(args)
    cfg.ensure_dirs()

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = cfg.training.device
    if device_str == "auto":
        if torch.cuda.is_available():             device = torch.device("cuda")
        elif torch.backends.mps.is_available():   device = torch.device("mps")
        else:                                     device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"\n🖥️  Device: {device}")

    # ── Check data exists ─────────────────────────────────────────────────────
    for split in ["train", "val", "test"]:
        path = os.path.join(cfg.training.data_dir, f"kaggle_{split}.csv")
        if not os.path.exists(path):
            print(f"\n❌ Missing file: {path}")
            print("   Run first:  python setup_kaggle.py --csv /path/to/creditcard.csv")
            sys.exit(1)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n📦 Loading Kaggle datasets...")
    train_loader, val_loader, _ = build_kaggle_dataloaders(
        data_dir=cfg.training.data_dir,
        batch_size=cfg.training.batch_size,
        max_seq_len=cfg.model.max_seq_len,
        num_workers=min(cfg.training.num_workers, 2),
        pin_memory=cfg.training.pin_memory and device.type == "cuda",
        use_weighted_sampler=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n🧠 Model:")
    model = FraudTransformer(cfg.model).to(device)
    print(model.summary())

    # ── Loss ─────────────────────────────────────────────────────────────────
    fraud_weight = torch.tensor([1.0, cfg.training.fraud_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=fraud_weight)
    print(f"\n   Fraud loss weight: {cfg.training.fraud_weight}×")

    # ── Optimizer + Schedule ──────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=cfg.training.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=cfg.training.num_epochs - cfg.training.warmup_epochs,
                                         eta_min=cfg.training.learning_rate * 0.01)
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup_scheduler, cosine_scheduler],
                              milestones=[cfg.training.warmup_epochs])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    history = []
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "kaggle_best.pt")
    latest_path = os.path.join(cfg.training.checkpoint_dir, "kaggle_latest.pt")

    print(f"\n🚀 Training for {cfg.training.num_epochs} epochs on Kaggle data")
    print(f"{'='*65}")

    for epoch in range(cfg.training.num_epochs):
        t0 = time.time()

        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, cfg, is_train=True)
        val_loss,   val_metrics   = run_epoch(model, val_loader,   criterion, None,      device, cfg, is_train=False)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        row = {"epoch": epoch+1, "lr": lr,
               "train_loss": train_loss, "val_loss": val_loss,
               **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}":   v for k, v in val_metrics.items()}}
        history.append(row)

        print(
            f"Epoch {epoch+1:03d}/{cfg.training.num_epochs} │ LR {lr:.2e} │ "
            f"Train Loss {train_loss:.4f} │ Val Loss {val_loss:.4f} │ "
            f"Val F1 {val_metrics['f1']:.4f} │ Val AUC {val_metrics['auc_roc']:.4f} │ "
            f"Val Rec {val_metrics['recall']:.4f} │ Val Prec {val_metrics['precision']:.4f} │ "
            f"{elapsed:.1f}s"
        )

        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_metrics": val_metrics,
                "config": {"model": vars(cfg.model), "training": vars(cfg.training)}}

        torch.save(ckpt, latest_path)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(ckpt, ckpt_path)
            print(f"  ✅ New best F1: {best_val_f1:.4f} — saved.")

    history_path = os.path.join(cfg.training.log_dir, "kaggle_train_history.json")
    os.makedirs(cfg.training.log_dir, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*65}")
    print(f"✅ Done. Best val F1: {best_val_f1:.4f}")
    print(f"   Checkpoint: {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train SLM on Kaggle Credit Card Fraud dataset")
    p.add_argument("--data-dir",       default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--device",         default="auto")
    p.add_argument("--fraud-weight",   type=float, default=100.0,
                   help="Loss weight for fraud class (default 100 for ~0.17%% fraud rate)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_kaggle(args)
