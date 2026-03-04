"""
model.py — ~1M Parameter Transformer Encoder for Finance Fraud Detection

Architecture:
  Embedding + Learned Positional Encoding
  → 4× TransformerEncoderLayer (d_model=128, nhead=4, ffn=256)
  → [CLS] token pooling
  → Classification Head (128 → 64 → 2)

Total params: ~742K–1.1M depending on config.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelConfig


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional embeddings (like BERT)."""

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.pos_embedding(positions)                            # (1, seq_len, d_model)
        return self.dropout(x + pos_emb)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (original Transformer paper)."""

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ClassificationHead(nn.Module):
    """Two-layer MLP classification head with dropout."""

    def __init__(self, d_model: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FraudTransformer(nn.Module):
    """
    Transformer Encoder for financial transaction fraud detection.

    Processes tokenized transaction sequences and classifies them as
    Legitimate (0) or Fraudulent (1).

    Input:  (batch_size, seq_len) — integer token IDs
    Output: (batch_size, 2)       — logits for [Legitimate, Fraud]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Token Embedding ──────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

        # ── Positional Encoding ──────────────────────────────────────────────
        if config.learned_pos_enc:
            self.pos_encoder = LearnedPositionalEncoding(
                config.max_seq_len, config.d_model, config.dropout
            )
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(
                config.max_seq_len, config.d_model, config.dropout
            )

        # ── Transformer Encoder ──────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,   # (batch, seq, feat) convention
            norm_first=True,    # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )

        # ── Classification Head ──────────────────────────────────────────────
        self.classifier = ClassificationHead(
            config.d_model,
            config.classifier_hidden,
            config.num_classes,
            config.dropout,
        )

        # ── Weight Initialization ────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Xavier/truncated-normal initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len) — token IDs including [CLS] at position 0
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            logits: (batch, num_classes)
        """
        # Convert padding mask to PyTorch's key_padding_mask format (True = ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # (batch, seq_len)

        # Embed tokens and add positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Encode with Transformer
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (batch, seq_len, d_model)

        # Pool [CLS] token (position 0) as the sequence representation
        cls_repr = x[:, 0, :]  # (batch, d_model)

        # Classify
        logits = self.classifier(cls_repr)  # (batch, num_classes)
        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns fraud probabilities (softmax of logits)."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            return F.softmax(logits, dim=-1)

    def count_parameters(self) -> dict:
        """Return parameter counts by component."""
        counts = {}
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                component = name.split(".")[0]
                counts[component] = counts.get(component, 0) + param.numel()
                total += param.numel()
        counts["TOTAL"] = total
        return counts

    def summary(self) -> str:
        """Human-readable model summary."""
        counts = self.count_parameters()
        lines = [
            "=" * 55,
            "  Finance Fraud Detection Transformer (SLM ~1M params)",
            "=" * 55,
            f"  Vocab size:         {self.config.vocab_size:,}",
            f"  d_model:            {self.config.d_model}",
            f"  Attention heads:    {self.config.nhead}",
            f"  Encoder layers:     {self.config.num_encoder_layers}",
            f"  FFN dim:            {self.config.dim_feedforward}",
            f"  Max seq len:        {self.config.max_seq_len}",
            "-" * 55,
        ]
        for component, count in counts.items():
            if component != "TOTAL":
                lines.append(f"  {component:<22} {count:>12,} params")
        lines += [
            "-" * 55,
            f"  {'TOTAL':<22} {counts['TOTAL']:>12,} params",
            "=" * 55,
        ]
        return "\n".join(lines)



if __name__ == "__main__":
    from config import ModelConfig
    cfg = ModelConfig()
    model = FraudTransformer(cfg)
    print(model.summary())

    # Smoke test forward pass
    batch = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len))
    mask = torch.ones(4, cfg.max_seq_len, dtype=torch.long)
    logits = model(batch, mask)
    print(f"\nInput:  {batch.shape}")
    print(f"Output: {logits.shape}")
    probs = model.predict_proba(batch, mask)
    print(f"Fraud probabilities: {probs[:, 1].tolist()}")
