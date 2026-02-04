"""
Search FAISS index and return top-k results.

Input:
  --index_dir containing faiss.index and ids.jsonl
  --query "text"
Output:
  prints JSON with top-k hits: (chunk_id, score, source, doc_id, title, meta)

Deterministic: fixed model, fixed top-k, fixed ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

from data.utils_text import normalize_text

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def _read_ids(path: Path) -> List[Dict[str, Any]]:
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
    ap.add_argument("--index_dir", required=True, help="Directory with faiss.index and ids.jsonl")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--normalize_embeddings", action="store_true")
    args = ap.parse_args()

    if faiss is None:
        raise RuntimeError("faiss is required. Install faiss-cpu or faiss-gpu.")

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    ids = _read_ids(index_dir / "ids.jsonl")

    model = SentenceTransformer(args.model_name)
    q = normalize_text(args.query)
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=args.normalize_embeddings).astype(np.float32)

    scores, idxs = index.search(q_emb, args.top_k)
    hits = []
    for rank, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), start=1):
        if idx < 0 or idx >= len(ids):
            continue
        rec = ids[idx]
        hits.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": rec.get("chunk_id"),
            "doc_id": rec.get("doc_id"),
            "source": rec.get("source"),
            "title": rec.get("title", ""),
            "meta": rec.get("meta", {}),
        })

    print(json.dumps({"query": q, "top_k": args.top_k, "hits": hits}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
