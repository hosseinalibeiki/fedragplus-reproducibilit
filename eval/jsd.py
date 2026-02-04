"""
Jensen-Shannon divergence utilities (numpy).
"""

from __future__ import annotations

import numpy as np


def _safe_norm(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = p.astype(np.float64)
    s = float(p.sum())
    if s <= eps:
        return np.ones_like(p, dtype=np.float64) / float(len(p))
    return p / (s + eps)


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_norm(p, eps)
    q = _safe_norm(q, eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def js_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_norm(p, eps)
    q = _safe_norm(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)
