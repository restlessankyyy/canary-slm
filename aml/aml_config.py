"""
aml/aml_config.py — Configuration for the AML Transformer model

AML differs from fraud detection in one key way:
  - Fraud: 1 transaction → classify (max_seq_len=64)
  - AML:   30 transactions per account → classify (max_seq_len=160)

We reuse ModelConfig / TrainingConfig from config.py with overrides.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass
from config import ModelConfig, TrainingConfig


@dataclass
class AMLModelConfig(ModelConfig):
    """
    Transformer config for AML sequence modelling.
    Each account sample = 30 transactions × 5 tokens = 150 tokens + [CLS].
    """
    vocab_size:          int   = 600    # base 512 + AML-specific tokens
    max_seq_len:         int   = 160    # 30 txns × 5 tokens + [CLS]
    d_model:             int   = 128
    nhead:               int   = 4
    num_encoder_layers:  int   = 4
    dim_feedforward:     int   = 256
    dropout:             float = 0.1
    num_classes:         int   = 2      # 0=clean, 1=suspicious
    classifier_hidden:   int   = 64
    learned_pos_enc:     bool  = True
    pad_token_id:        int   = 0


@dataclass
class AMLTrainingConfig(TrainingConfig):
    """Training config tuned for AML (more imbalanced than fraud)."""
    num_epochs:     int   = 20
    batch_size:     int   = 128
    learning_rate:  float = 3e-4
    fraud_weight:   float = 20.0    # AML suspicious accounts ~5% of data
    warmup_epochs:  int   = 2
    eval_metric:    str   = "f1"
    data_dir:       str   = "aml/data"
    checkpoint_dir: str   = "checkpoints"
    log_dir:        str   = "logs"
    device:         str   = "auto"
    num_workers:    int   = 0
    pin_memory:     bool  = False
    experiment_name: str  = "aml"


def get_aml_config():

    class AMLConfig:
        def __init__(self):
            self.model    = AMLModelConfig()
            self.training = AMLTrainingConfig()

        def ensure_dirs(self):
            for d in [self.training.data_dir,
                      self.training.checkpoint_dir,
                      self.training.log_dir]:
                os.makedirs(d, exist_ok=True)

    return AMLConfig()
