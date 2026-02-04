"""
Label-free retrieval diagnostics for a given client index + probe prompts.

Computes:
- mean/std of top-k similarity scores
- mean/std per rank

This module focuses on immediate, reproducible diagnostics without relevance labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data.utils_text import normalize_text

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


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
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--prompts_json", required=True, help="JSON list of prompts or JSONL with {prompt: ...}")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--normalize_embeddings", action="store_true")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    if faiss is None:
        raise RuntimeError("faiss is required. Install faiss-cpu or faiss-gpu.")

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))

    # Load prompts
    p_path = Path(args.prompts_json)
    prompts: List[str] = []
    if p_path.suffix.lower() == ".jsonl":
        for r in _read_jsonl(p_path):
            prompts.append(str(r.get("prompt", "")))
    else:
        prompts = json.loads(p_path.read_text(encoding="utf-8"))
    prompts = [normalize_text(p) for p in prompts if p and p.strip()]

    model = SentenceTransformer(args.model_name)

    all_scores = []
    for p in tqdm(prompts, desc="diagnostics"):
        q_emb = model.encode([p], convert_to_numpy=True, normalize_embeddings=args.normalize_embeddings).astype(np.float32)
        scores, _ = index.search(q_emb, args.top_k)
        all_scores.append(scores[0].astype(np.float32))

    all_scores = np.stack(all_scores, axis=0)  # [N, k]
    stats = {
        "n_prompts": int(all_scores.shape[0]),
        "top_k": int(args.top_k),
        "score_mean": float(all_scores.mean()),
        "score_std": float(all_scores.std(ddof=0)),
        "score_mean_per_rank": [float(x) for x in all_scores.mean(axis=0).tolist()],
        "score_std_per_rank": [float(x) for x in all_scores.std(axis=0, ddof=0).tolist()],
        "model_name": args.model_name,
        "normalize_embeddings": bool(args.normalize_embeddings),
        "index_dir": str(index_dir),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[diag] -> {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
