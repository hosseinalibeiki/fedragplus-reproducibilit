"""
Chunk documents into fixed-size token chunks with overlap (deterministic).

Input: JSONL of docs with fields: doc_id, title, text, source, meta (optional)
Output: JSONL of chunks with fields:
  chunk_id, doc_id, source, title, text, n_tokens, chunk_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Optional

from .utils_text import stable_hash, normalize_text


def _get_tokenizer(tokenizer_name: Optional[str]):
    if tokenizer_name is None:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer '{tokenizer_name}'. "
            f"Install transformers or choose tokenizer_name=None for whitespace tokenization. "
            f"Original error: {e}"
        )


def _tokenize(text: str, tok) -> list:
    if tok is None:
        return text.split()
    return tok.tokenize(text)


def _detokenize(tokens: list, tok) -> str:
    if tok is None:
        return " ".join(tokens)
    # For fast tokenizers, convert tokens back to string approximately.
    # We avoid decode(ids) because we don't have ids here; keep safe + deterministic.
    return tok.convert_tokens_to_string(tokens)


def iter_docs(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL docs")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL chunks")
    ap.add_argument("--chunk_tokens", type=int, default=384)
    ap.add_argument("--overlap_tokens", type=int, default=50)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--tokenizer_name", default=None, help="Optional HF tokenizer name (else whitespace tokens)")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = _get_tokenizer(args.tokenizer_name)

    chunk_T = args.chunk_tokens
    overlap = args.overlap_tokens
    assert chunk_T > 0 and overlap >= 0 and overlap < chunk_T

    n_in = 0
    n_out = 0

    with out_path.open("w", encoding="utf-8") as out:
        for d in iter_docs(in_path):
            n_in += 1
            doc_id = d["doc_id"]
            source = d.get("source", "unknown")
            title = d.get("title", "")
            text = normalize_text(d.get("text", ""))

            if len(text) < args.min_chars:
                continue

            tokens = _tokenize(text, tok)
            if len(tokens) == 0:
                continue

            start = 0
            chunk_idx = 0
            while start < len(tokens):
                end = min(start + chunk_T, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = _detokenize(chunk_tokens, tok).strip()
                if len(chunk_text) >= args.min_chars:
                    chunk_id = stable_hash(f"{doc_id}:{chunk_idx}:{chunk_text}", prefix="chunk:")
                    rec = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "source": source,
                        "title": title,
                        "chunk_index": chunk_idx,
                        "n_tokens": int(len(chunk_tokens)),
                        "text": chunk_text,
                        "meta": d.get("meta", {}),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1

                if end == len(tokens):
                    break
                start = end - overlap
                chunk_idx += 1

    print(f"[chunk] docs_read={n_in} chunks_written={n_out} -> {out_path}")


if __name__ == "__main__":
    main()
