"""
Generate an unlabeled probe prompt set deterministically.

Two modes:
1) From federated chunks (recommended): sample chunk titles / lead sentences from client chunks.jsonl
2) From docs JSONL (wiki_docs.jsonl etc.)

Outputs JSON list of prompt strings (default) OR JSONL with {"prompt": "..."}.

Design goal:
- No reference answers
- Stable across runs (seeded)
- Domain coverage across clients

Example:
  python prompts/generate_prompts.py \
    --sources federated_data/client_1/chunks.jsonl federated_data/client_2/chunks.jsonl \
    --n_prompts 300 --seed 42 --out prompts/prompts_seed42.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List

from data.utils_text import normalize_text


_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")


TEMPLATES = [
    "Explain the concept of {topic}.",
    "Summarize the key idea of {topic}.",
    "What is the role of {topic} in its context?",
    "Provide an overview of {topic} and why it matters.",
    "Describe {topic} in simple terms and mention one example.",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_topic(rec: Dict[str, Any]) -> str:
    title = normalize_text(str(rec.get("title", "")))
    if title and len(title) >= 5:
        return title[:120]
    # fallback: use first sentence from text
    text = normalize_text(str(rec.get("text", "")))
    sents = _SENT_SPLIT.split(text)
    topic = sents[0] if sents else text
    topic = topic[:140]
    # avoid empty
    return topic if topic else "the topic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="One or more JSONL files (chunks.jsonl or docs.jsonl)")
    ap.add_argument("--n_prompts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="Output .json or .jsonl")
    ap.add_argument("--per_source", type=int, default=0, help="Optional cap per source file (0 = no cap)")
    ap.add_argument("--jsonl", action="store_true", help="Write JSONL instead of JSON list")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load and pool records deterministically
    all_recs: List[Dict[str, Any]] = []
    for src in args.sources:
        rows = _read_jsonl(Path(src))
        if args.per_source and len(rows) > args.per_source:
            # deterministic subsample
            idxs = list(range(len(rows)))
            rng.shuffle(idxs)
            rows = [rows[i] for i in idxs[: args.per_source]]
        all_recs.extend(rows)

    if not all_recs:
        raise ValueError("No records loaded from sources.")

    # Deterministic shuffle
    idxs = list(range(len(all_recs)))
    rng.shuffle(idxs)

    prompts: List[str] = []
    seen = set()

    for i in idxs:
        topic = _extract_topic(all_recs[i])
        template = rng.choice(TEMPLATES)
        p = normalize_text(template.format(topic=topic))
        if p in seen:
            continue
        seen.add(p)
        prompts.append(p)
        if len(prompts) >= args.n_prompts:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.jsonl or out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for p in prompts:
                f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    else:
        out_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prompts] n={len(prompts)} seed={args.seed} -> {out_path}")


if __name__ == "__main__":
    main()
