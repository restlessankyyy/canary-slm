# 🐦 CANARY
### Contextual Anomaly Network for Anomaly Recognition & analYsis

> *Like a canary in a coal mine — a ~600K parameter Transformer that detects financial fraud before it spreads.*

[![CI](https://github.com/ankit-raj_hmgroup/canary-slm/actions/workflows/ci.yml/badge.svg)](https://github.com/ankit-raj_hmgroup/canary-slm/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```
Transaction → [CLS][AMT:1k-2k][MCC:CRYPTO][CTRY:FOREIGN][TIME:EARLY_MORNING][VEL:EXTREME]...
                                         ↓ CANARY
                         🚨 CRITICAL RISK — fraud_prob: 0.97
```

---

## 🏗️ Architecture

```
Tokenized Transaction (max 64 tokens)
        │
        ▼
Token Embedding (vocab=512) + Learned Positional Encoding
        │
        ▼
 ┌──────────────────────────────────┐
 │  TransformerEncoder × 4 layers  │
 │  d_model=128 │ heads=4 │ FFN=256│
 │  Pre-LayerNorm │ GELU │ Dropout │
 └──────────────────────────────────┘
        │
      [CLS] pool
        │
        ▼
 Dense(128→64) → GELU → Dense(64→2)
        │
        ▼
   P(fraud) [0–1]
```

### Parameter Budget

| Component | Params |
|:---|---:|
| Token Embedding | 65,536 |
| Positional Encoding | 8,192 |
| Transformer Encoder ×4 | 530,176 |
| Classification Head | 8,386 |
| **Total** | **~612K** |

---

## 🚀 Quick Start

```bash
git clone https://github.com/restlessankyyy/canary-slm
cd canary-slm
pip install -r requirements.txt

# Option A: Synthetic data (no download needed)
python train.py --epochs 20

# Option B: Real Kaggle data (~284K real transactions)
python download_kaggle_data.py   # downloads via kagglehub
python setup_kaggle.py --csv data/creditcard.csv
python train_kaggle.py --epochs 20

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt

# Interactive demo
python demo.py
```

---

## 📂 Project Structure

```
canary-slm/
├── model.py                   # CANARY Transformer Encoder
├── config.py                  # Model & training config
├── train.py                   # Training loop (synthetic data)
├── train_kaggle.py            # Training on real Kaggle data
├── evaluate.py                # Metrics: F1, AUC-ROC, confusion matrix
├── inference.py               # FraudDetector inference API
├── demo.py                    # Interactive CLI demo
├── download_kaggle_data.py    # Download creditcard.csv via kagglehub
├── setup_kaggle.py            # Preprocess Kaggle CSV
├── requirements.txt
├── data/
│   ├── tokenizer.py           # 512-token financial domain vocabulary
│   ├── generate_synthetic.py  # 5-profile fraud data generator
│   ├── dataset.py             # PyTorch Dataset + WeightedRandomSampler
│   ├── preprocess_kaggle.py   # V1–V28 quantile binner
│   └── kaggle_dataset.py      # Extended tokenizer + KaggleFraudDataset
└── .github/workflows/
    ├── ci.yml                 # Lint + tests on every push
    └── train.yml              # On-demand training workflow
```

---

## 🔬 Data: 5 Fraud Profiles (Synthetic)

| Profile | Key Signals |
|---|---|
| **Card Testing** | Micro amounts (<$1), rapid succession, new device |
| **Account Takeover** | $1.5K–$15K, foreign country, late night |
| **CNP Fraud** | Online, billing mismatch, foreign IP |
| **ATM Skimming** | Cloned card, foreign ATM, geo-impossible |
| **Money Mule** | Crypto/transfer, extreme velocity, Tor/VPN |

For the real Kaggle dataset, V1–V28 PCA features are binned into 5 quantile-based tokens per feature (140 additional tokens).

---

## 🔌 Inference API

```python
from inference import FraudDetector

detector = FraudDetector.from_checkpoint("checkpoints/best_model.pt")

result = detector.predict({
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
})

print(result["risk_label"])         # 🚨 CRITICAL RISK
print(result["fraud_probability"])  # 0.97
print(result["action"])             # Block immediately
print(result["risk_factors"])       # ['CTRY:FOREIGN', 'VEL:EXTREME', ...]
```

---

## 📈 Training Hyperparameters

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=1e-2) |
| LR Schedule | Linear warmup (2 ep) → Cosine decay |
| Loss | Weighted Cross-Entropy (fraud weight=10× synthetic / 150× Kaggle) |
| Sampler | WeightedRandomSampler |
| Batch size | 256 |
| Grad clip | 1.0 |

---

## 📝 License

MIT
