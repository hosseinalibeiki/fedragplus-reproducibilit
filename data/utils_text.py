"""
Text utilities (deterministic cleaning + hashing).
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional


_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_NONPRINT_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+")


def stable_hash(text: str, prefix: str = "") -> str:
    h = hashlib.sha1((prefix + text).encode("utf-8", errors="ignore")).hexdigest()
    return h


def strip_html(text: str) -> str:
    # Lightweight HTML tag removal (good enough for StackExchange bodies).
    return _TAG_RE.sub(" ", text)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _NONPRINT_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = strip_html(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_usable(text: str, min_chars: int = 200) -> bool:
    return text is not None and len(text) >= min_chars
