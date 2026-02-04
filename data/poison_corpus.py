"""
Poison a client's retrieval corpus by replacing a fraction of chunks.

Poison types (deterministic):
- shuffle: shuffle words inside the chunk
- off_topic: replace chunk text with a deterministic fixed boilerplate + hashed id
- shuffle_or_offtopic: alternates based on chunk_id hash

Input: --client_dir federated_data/client_X
Output: overwrites chunks.jsonl unless --out_client_dir provided
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List

from .utils_text import stable_hash, normalize_text


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _poison_text(text: str, chunk_id: str, poison_type: str, rng: random.Random) -> str:
    words = text.split()
    if poison_type == "shuffle":
        rng.shuffle(words)
        return " ".join(words)
    if poison_type == "off_topic":
        return normalize_text(f"This passage is corrupted and does not contain relevant evidence. id={chunk_id}")
    if poison_type == "shuffle_or_offtopic":
        # deterministic branch by hash
        h = int(stable_hash(chunk_id, prefix="branch:")[:8], 16)
        if h % 2 == 0:
            rng.shuffle(words)
            return " ".join(words)
        return normalize_text(f"This passage is corrupted and does not contain relevant evidence. id={chunk_id}")
    raise ValueError(f"Unknown poison_type: {poison_type}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_dir", required=True, help="Path like federated_data/client_4")
    ap.add_argument("--poison_rate", type=float, default=0.20)
    ap.add_argument("--poison_type", default="shuffle_or_offtopic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_client_dir", default=None, help="Optional output dir (else overwrite)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    client_dir = Path(args.client_dir)
    in_path = client_dir / "chunks.jsonl"
    rows = _read_jsonl(in_path)

    n = len(rows)
    k = int(round(n * args.poison_rate))
    idxs = list(range(n))
    rng.shuffle(idxs)
    poison_idxs = set(idxs[:k])

    out_rows = []
    for j, r in enumerate(rows):
        if j in poison_idxs:
            r2 = dict(r)
            r2["text"] = _poison_text(r2.get("text", ""), r2.get("chunk_id", ""), args.poison_type, rng)
            r2.setdefault("meta", {})
            r2["meta"]["poisoned"] = True
            out_rows.append(r2)
        else:
            r.setdefault("meta", {})
            r["meta"]["poisoned"] = False
            out_rows.append(r)

    out_dir = Path(args.out_client_dir) if args.out_client_dir else client_dir
    out_path = out_dir / "chunks.jsonl"
    _write_jsonl(out_path, out_rows)

    print(f"[poison] in={in_path} poisoned={k}/{n} ({args.poison_rate:.2f}) -> {out_path}")


if __name__ == "__main__":
    main()
