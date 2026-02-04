"""
Regularizers used during local updates (label-free).

- KL stability regularizer: encourages local model logits to remain close to global model logits
  on an unlabeled probe batch.

This keeps claims aligned with "stability" rather than supervised quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F


@dataclass
class KLRegConfig:
    enabled: bool = False
    lambda_kl: float = 0.0
    temperature: float = 1.0


def kl_logits(p_logits: torch.Tensor, q_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    KL( softmax(q/T) || softmax(p/T) ) averaged over tokens.
    p_logits: local model logits [B, T, V]
    q_logits: global model logits [B, T, V]
    """
    T = float(temperature)
    p = F.log_softmax(p_logits / T, dim=-1)
    q = F.softmax(q_logits / T, dim=-1)
    # kl_div expects log-prob input and prob target
    kl = F.kl_div(p, q, reduction="batchmean")
    return kl * (T * T)


def apply_kl_regularizer(
    loss: torch.Tensor,
    local_logits: torch.Tensor,
    global_logits: torch.Tensor,
    cfg: KLRegConfig
) -> torch.Tensor:
    if not cfg.enabled or cfg.lambda_kl <= 0:
        return loss
    kl = kl_logits(local_logits, global_logits, temperature=cfg.temperature)
    return loss + cfg.lambda_kl * kl
