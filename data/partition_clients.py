"""
Partition chunked corpora into federated clients (deterministic).

Inputs: one or more chunk JSONL files (from chunk_documents.py)
Output: directory federated_data/client_{i}/chunks.jsonl

Non-IID options:
- zipf: sample more frequently from earlier-ranked chunks (after deterministic shuffle).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

from .utils_text import stable_hash


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _zipf_sample(items: List[Dict[str, Any]], n: int, alpha: float, rng: random.Random) -> List[Dict[str, Any]]:
    # Deterministic shuffle first, then assign ranks
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    # Zipf weights on ranks 1..N
    weights = [1.0 / ((r + 1) ** alpha) for r in range(len(idxs))]
    # Normalize cumulative distribution
    total = sum(weights)
    cdf = []
    s = 0.0
    for w in weights:
        s += w / total
        cdf.append(s)

    def draw_one() -> int:
        u = rng.random()
        # linear scan is fine for moderate sizes; users can optimize later
        for j, p in enumerate(cdf):
            if u <= p:
                return idxs[j]
        return idxs[-1]

    sampled = []
    seen = set()
    while len(sampled) < n and len(seen) < len(items):
        j = draw_one()
        cid = items[j]["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        sampled.append(items[j])
    return sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_jsonl", nargs="+", required=True, help="One or more chunk JSONL files")
    ap.add_argument("--out_dir", required=True, help="Output directory (federated_data)")
    ap.add_argument("--num_clients", type=int, default=4)
    ap.add_argument("--docs_per_client", type=int, default=22000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--non_iid", choices=["zipf", "uniform"], default="zipf")
    ap.add_argument("--zipf_alpha", type=float, default=1.2)
    ap.add_argument("--client_mix", default="wikipedia,pubmed,stackexchange,mixed",
                    help="Comma-separated client source pattern for 4 clients; use 'mixed' for pooled")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for p in args.chunk_jsonl:
        chunks = _read_jsonl(Path(p))
        all_chunks.extend(chunks)
        for c in chunks:
            by_source.setdefault(c.get("source", "unknown"), []).append(c)

    patterns = [x.strip() for x in args.client_mix.split(",")]
    if args.num_clients != len(patterns):
        raise ValueError("client_mix length must equal num_clients for deterministic mapping")

    for i in range(1, args.num_clients + 1):
        pat = patterns[i - 1]
        if pat == "mixed":
            pool = all_chunks
        else:
            pool = by_source.get(pat, [])
        if len(pool) == 0:
            raise ValueError(f"No chunks available for client {i} pattern '{pat}'")

        if args.non_iid == "zipf":
            picked = _zipf_sample(pool, args.docs_per_client, args.zipf_alpha, rng)
        else:
            idxs = list(range(len(pool)))
            rng.shuffle(idxs)
            picked = [pool[j] for j in idxs[: args.docs_per_client]]

        client_dir = out_root / f"client_{i}"
        client_dir.mkdir(parents=True, exist_ok=True)
        out_path = client_dir / "chunks.jsonl"
        with out_path.open("w", encoding="utf-8") as out:
            for rec in picked:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[partition] client_{i} pattern={pat} chunks={len(picked)} -> {out_path}")


if __name__ == "__main__":
    main()
