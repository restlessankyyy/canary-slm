"""
config.py — Model & Training Configuration for Finance Fraud Detection SLM
~1M parameter Transformer Encoder for binary transaction classification.
"""

from dataclasses import dataclass, field
import os


@dataclass
class ModelConfig:
    """Transformer Encoder architecture hyperparameters.

    Targeting ~1M total parameters:
      Embedding:       vocab_size(512) × d_model(128)     = 65,536
      Pos Encoding:    seq_len(64)    × d_model(128)      = 8,192
      Encoder ×4:      4 × [attn(128²×4×2) + ffn(128×256×2) + norms] ≈ 660,000
      Classifier:      128×64 + 64×2                      = 8,322
      Total:                                              ≈ 742,050
    """
    # Vocabulary (domain-specific financial tokens)
    vocab_size: int = 512
    pad_token_id: int = 0
    cls_token_id: int = 1
    unk_token_id: int = 2

    # Sequence
    max_seq_len: int = 64          # Max tokens per transaction

    # Transformer dims
    d_model: int = 128             # Hidden size
    nhead: int = 4                 # Attention heads (d_model / nhead = 32)
    num_encoder_layers: int = 4    # Encoder depth
    dim_feedforward: int = 256     # FFN inner dim (2× d_model)
    dropout: float = 0.1

    # Classification
    num_classes: int = 2           # 0=Legitimate, 1=Fraud
    classifier_hidden: int = 64    # Head hidden dim

    # Learned positional encoding
    learned_pos_enc: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    clip_grad_norm: float = 1.0

    # Schedule
    num_epochs: int = 20
    warmup_epochs: int = 2          # Linear warmup then cosine decay

    # Data
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.1
    test_split: float = 0.1

    # Class imbalance (fraud ~2% of transactions)
    fraud_weight: float = 10.0      # Loss weight for fraud class
    oversampling_ratio: float = 0.1 # Oversample fraud to this fraction in training

    # Evaluation
    eval_metric: str = "f1"         # Metric to use for best checkpoint
    eval_every_n_epochs: int = 1

    # Logging
    log_interval: int = 50          # Steps between log prints
    use_wandb: bool = False
    wandb_project: str = "fraud-detection-slm"
    experiment_name: str = "baseline"

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cpu"            # "cuda", "mps", or "cpu"


@dataclass
class DataConfig:
    """Synthetic data generation configuration."""
    # Dataset sizes
    num_train: int = 100_000
    num_val: int = 10_000
    num_test: int = 10_000

    # Fraud ratio in generated data (real-world: ~0.1–2%)
    fraud_ratio: float = 0.05      # 5% in synthetic for easier training

    # Random seed
    seed: int = 42

    # Output paths
    train_path: str = "data/train.csv"
    val_path: str = "data/val.csv"
    test_path: str = "data/test.csv"
    vocab_path: str = "data/vocab.json"


@dataclass
class Config:
    """Master config combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def ensure_dirs(self):
        """Create required directories if they don't exist."""
        for d in [self.training.checkpoint_dir, self.training.log_dir, self.training.data_dir]:
            os.makedirs(d, exist_ok=True)


def get_default_config() -> Config:
    return Config()
