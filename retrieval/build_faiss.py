"""
Build FAISS index for dense retrieval.

Input: --embeddings_npy and --ids_jsonl (from embed_minilm.py)
Output: --out_dir containing:
  - faiss.index
  - ids.jsonl (copied)
  - index_meta.json

Supports:
- FlatIP (inner product) for normalized embeddings (recommended)
- FlatL2

Deterministic: no training randomness for Flat indexes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_npy", required=True)
    ap.add_argument("--ids_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--index_type", choices=["FlatIP", "FlatL2"], default="FlatIP")
    args = ap.parse_args()

    if faiss is None:
        raise RuntimeError("faiss is required. Install faiss-cpu (portable) or faiss-gpu (optional).")

    embs = np.load(args.embeddings_npy).astype(np.float32)
    n, d = embs.shape

    if args.index_type == "FlatIP":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(embs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    shutil.copyfile(args.ids_jsonl, out_dir / "ids.jsonl")

    meta = {
        "n": int(n),
        "dim": int(d),
        "index_type": args.index_type,
        "embeddings_npy": str(Path(args.embeddings_npy).resolve()),
        "ids_jsonl": str(Path(args.ids_jsonl).resolve()),
    }
    (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[faiss] built {args.index_type} index: n={n}, dim={d} -> {out_dir}")


if __name__ == "__main__":
    main()
