"""
Embed chunks using a SentenceTransformer (MiniLM by default).

Input: --chunks_jsonl (from federated_data/client_i/chunks.jsonl)
Output: --out_dir containing:
  - embeddings.npy (float32)
  - ids.jsonl (chunk_id, doc_id, source, title, meta)
  - stats.json

Deterministic: fixed batch ordering as read from file; no shuffling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

from data.utils_text import normalize_text


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--out_dir", required=True, help="Output directory for embeddings + ids")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--normalize_embeddings", action="store_true", help="L2-normalize embeddings")
    ap.add_argument("--max_chars", type=int, default=2000, help="Truncate chunk text for embedding")
    args = ap.parse_args()

    in_path = Path(args.chunks_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(in_path)
    texts = [normalize_text(r.get("text", ""))[: args.max_chars] for r in rows]

    model = SentenceTransformer(args.model_name)
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize_embeddings,
    ).astype(np.float32)

    # Save
    np.save(out_dir / "embeddings.npy", embs)

    ids_path = out_dir / "ids.jsonl"
    with ids_path.open("w", encoding="utf-8") as f:
        for r in rows:
            keep = {
                "chunk_id": r.get("chunk_id"),
                "doc_id": r.get("doc_id"),
                "source": r.get("source"),
                "title": r.get("title", ""),
                "meta": r.get("meta", {}),
            }
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")

    stats = {
        "n": int(embs.shape[0]),
        "dim": int(embs.shape[1]),
        "model_name": args.model_name,
        "normalize_embeddings": bool(args.normalize_embeddings),
        "batch_size": int(args.batch_size),
        "max_chars": int(args.max_chars),
        "input": str(in_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[embed] n={stats['n']} dim={stats['dim']} -> {out_dir}")


if __name__ == "__main__":
    main()
