"""
Aggregation utilities for federated learning.

We aggregate trainable parameters only (e.g., LoRA adapters, or full model if configured).
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple
import torch


def weighted_average_state_dict(state_dicts: Iterable[Dict[str, torch.Tensor]], weights: Iterable[float]) -> Dict[str, torch.Tensor]:
    state_dicts = list(state_dicts)
    weights = list(weights)
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("Sum of aggregation weights must be > 0.")

    # Initialize
    avg: Dict[str, torch.Tensor] = {}
    for k in state_dicts[0].keys():
        avg[k] = torch.zeros_like(state_dicts[0][k])

    for sd, w in zip(state_dicts, weights):
        w = float(w) / wsum
        for k, v in sd.items():
            avg[k] = avg[k] + v.detach() * w

    return avg
