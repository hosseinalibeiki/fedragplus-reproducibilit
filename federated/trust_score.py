"""
Trust score computation (label-free, retrieval-consistency based).

We compute each client's trust using retrieval distribution stability on a fixed probe prompt set.
This is reproducible and does not require relevance labels or reference answers.

Trust score s_i is mapped to [clip_min, clip_max] after smoothing (EMA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TrustConfig:
    enabled: bool = True
    smoothing: float = 0.6  # EMA coefficient
    clip_min: float = 0.05
    clip_max: float = 1.0
    score_mode: str = "retrieval_consistency"  # only mode implemented here


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    def _kl(a, b):
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def trust_from_hist(client_hist: np.ndarray, global_hist: np.ndarray) -> float:
    jsd = _js_divergence(client_hist, global_hist)
    # Map stability -> trust (bounded, monotone decreasing in JSD)
    return float(np.exp(-jsd))


def update_trust_ema(prev: float, new: float, smoothing: float) -> float:
    a = float(smoothing)
    return a * prev + (1.0 - a) * new


def clip_trust(s: float, clip_min: float, clip_max: float) -> float:
    return float(min(max(s, clip_min), clip_max))
