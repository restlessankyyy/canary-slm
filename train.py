"""
train.py — Training Loop for Finance Fraud Detection SLM

Features:
  - AdamW optimizer + cosine LR schedule with linear warmup
  - Weighted Cross-Entropy loss for class imbalance
  - Best-checkpoint saving (by val F1)
  - Rich progress output with tqdm
  - Optional W&B logging
"""

import os
import time
import argparse
import random
import json
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from config import Config, get_default_config
from model import FraudTransformer
from data.dataset import build_dataloaders


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics from arrays."""
    return {
        "accuracy":  float((all_labels == all_preds).mean()),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1":        float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 0.0,
    }


# ── Single epoch ─────────────────────────────────────────────────────────────

def run_epoch(
    model:      FraudTransformer,
    loader:     DataLoader,
    criterion:  nn.CrossEntropyLoss,
    optimizer:  Optional[AdamW],
    device:     torch.device,
    cfg:        Config,
    is_train:   bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Run one train or eval epoch. Returns (avg_loss, metrics_dict)."""
    model.train(is_train)

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    iter_loader = tqdm(loader, desc="  Train" if is_train else "  Eval ", leave=False) if TQDM else loader

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, (input_ids, attention_mask, labels) in enumerate(iter_loader):
            input_ids     = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels        = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
                optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

            if TQDM and is_train and step % cfg.training.log_interval == 0:
                iter_loader.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )
    return avg_loss, metrics


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: Config, resume: Optional[str] = None):
    set_seed(cfg.training.seed)
    cfg.ensure_dirs()

    # ── Device ───────────────────────────────────────────────────────────────
    if cfg.training.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.training.device)
    print(f"\n🖥️  Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    # Generate synthetic data if not already present
    for split in ["train", "val", "test"]:
        path = os.path.join(cfg.training.data_dir, f"{split}.csv")
        if not os.path.exists(path):
            print("\n📊 Dataset not found — generating synthetic data...")
            from data.generate_synthetic import main as gen_main
            import sys as _sys
            old_argv = _sys.argv
            _sys.argv = [
                "generate_synthetic.py",
                "--n-train", str(cfg.data.num_train),
                "--n-val",   str(cfg.data.num_val),
                "--n-test",  str(cfg.data.num_test),
                "--fraud-ratio", str(cfg.data.fraud_ratio),
                "--out-dir", cfg.training.data_dir,
                "--seed",    str(cfg.data.seed),
            ]
            gen_main()
            _sys.argv = old_argv
            break

    print("\n📦 Loading datasets...")
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=cfg.training.data_dir,
        batch_size=cfg.training.batch_size,
        max_seq_len=cfg.model.max_seq_len,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and device.type == "cuda",
        use_weighted_sampler=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n🧠 Model:")
    model = FraudTransformer(cfg.model).to(device)
    print(model.summary())

    # ── Loss (weighted for class imbalance) ──────────────────────────────────
    fraud_weight = torch.tensor([1.0, cfg.training.fraud_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=fraud_weight)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.beta1, cfg.training.beta2),
    )

    # ── LR Schedule: linear warmup → cosine decay ────────────────────────────
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.training.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs - cfg.training.warmup_epochs,
        eta_min=cfg.training.learning_rate * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.training.warmup_epochs],
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_metric = 0.0
    if resume and os.path.exists(resume):
        print(f"\n⏩ Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_metric = ckpt.get("best_val_metric", 0.0)

    # ── W&B (optional) ───────────────────────────────────────────────────────
    wandb_run = None
    if cfg.training.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.training.wandb_project,
                name=cfg.training.experiment_name,
                config={"model": vars(cfg.model), "training": vars(cfg.training)},
            )
        except ImportError:
            print("  W&B not installed, skipping.")

    # ── Training Loop ────────────────────────────────────────────────────────
    history = []
    print(f"\n🚀 Training for {cfg.training.num_epochs} epochs\n{'='*60}")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        t0 = time.time()

        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, cfg, is_train=True)
        val_loss,   val_metrics   = run_epoch(model, val_loader,   criterion, None,      device, cfg, is_train=False)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        val_metric = val_metrics[cfg.training.eval_metric]

        # ── Log ──────────────────────────────────────────────────────────────
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch+1:03d}/{cfg.training.num_epochs} │ "
            f"LR {lr:.2e} │ "
            f"Train Loss {train_loss:.4f} │ "
            f"Val Loss {val_loss:.4f} │ "
            f"Val F1 {val_metrics['f1']:.4f} │ "
            f"Val AUC {val_metrics['auc_roc']:.4f} │ "
            f"Val Rec {val_metrics['recall']:.4f} │ "
            f"{elapsed:.1f}s"
        )

        if wandb_run:
            wandb_run.log(row)

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt = {
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "val_metrics":      val_metrics,
            "best_val_metric":  best_val_metric,
            "config":           {
                "model":    vars(cfg.model),
                "training": vars(cfg.training),
            },
        }
        # Always save latest
        torch.save(ckpt, os.path.join(cfg.training.checkpoint_dir, "latest.pt"))

        # Save best
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            ckpt["best_val_metric"] = best_val_metric
            torch.save(ckpt, os.path.join(cfg.training.checkpoint_dir, "best_model.pt"))
            print(f"  ✅ New best {cfg.training.eval_metric}: {best_val_metric:.4f} — saved.")

    # ── Save history ─────────────────────────────────────────────────────────
    history_path = os.path.join(cfg.training.log_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Training complete. Best val {cfg.training.eval_metric}: {best_val_metric:.4f}")
    print(f"   Checkpoint: {cfg.training.checkpoint_dir}/best_model.pt")
    print(f"   History:    {history_path}")

    if wandb_run:
        wandb_run.finish()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Finance Fraud Detection SLM")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--device",     type=str,   default=None)
    parser.add_argument("--data-dir",   type=str,   default=None)
    parser.add_argument("--resume",     type=str,   default=None)
    args = parser.parse_args()

    cfg = get_default_config()
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.device:
        cfg.training.device = args.device
    if args.data_dir:
        cfg.training.data_dir = args.data_dir
    return cfg, args.resume


if __name__ == "__main__":
    cfg, resume = parse_args()
    train(cfg, resume=resume)
