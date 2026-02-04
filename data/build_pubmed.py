"""
Build PubMed abstract corpus from a PubMed baseline XML (.xml.gz).

Input: --pubmed_xml_gz PATH
Output: docs JSONL with fields: doc_id, title, text, source, meta

This script is deterministic and streaming (iterparse).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Any, Optional
import xml.etree.ElementTree as ET

from .utils_text import stable_hash, normalize_text, is_usable


def _text_or_none(elem) -> Optional[str]:
    if elem is None:
        return None
    t = "".join(elem.itertext()).strip()
    return t if t else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubmed_xml_gz", required=True, help="Path to pubmed baseline .xml.gz")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL docs")
    ap.add_argument("--min_chars", type=int, default=200)
    args = ap.parse_args()

    in_path = Path(args.pubmed_xml_gz)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_seen = 0

    with gzip.open(in_path, "rb") as f, out_path.open("w", encoding="utf-8") as out:
        # Streaming parse
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag != "PubmedArticle":
                continue

            n_seen += 1
            pmid_elem = elem.find(".//PMID")
            pmid = _text_or_none(pmid_elem) or f"pmid_{n_seen}"

            title = _text_or_none(elem.find(".//ArticleTitle")) or ""
            abstract_parts = []
            for ab in elem.findall(".//Abstract/AbstractText"):
                part = _text_or_none(ab)
                if part:
                    abstract_parts.append(part)
            abstract = " ".join(abstract_parts)

            text = normalize_text((title + " " + abstract).strip())
            if not is_usable(text, min_chars=args.min_chars):
                elem.clear()
                continue

            doc_id = stable_hash(f"pubmed:{pmid}", prefix="doc:")
            rec = {
                "doc_id": doc_id,
                "source": "pubmed",
                "title": normalize_text(title),
                "text": text,
                "meta": {"pmid": pmid},
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            elem.clear()

    print(f"[pubmed] articles_seen={n_seen} docs_written={n_written} -> {out_path}")


if __name__ == "__main__":
    main()
