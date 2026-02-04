"""
Build Wikipedia corpus from Wikimedia XML dump.

Supports:
- Single-file pages-articles dump (*.xml.bz2)
- Multistream pages-articles-multistream.xml.bz2 (without using the index; we stream parse XML)

IMPORTANT:
- This script extracts plain text from <text> using mwparserfromhell (recommended).
- If mwparserfromhell is not installed, it falls back to a minimal regex cleanup that is less accurate.
- Output is deterministic and streaming.

Input: --wikipedia_xml_bz2 PATH
Output: docs JSONL with fields: doc_id, title, text, source, meta
"""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from .utils_text import stable_hash, normalize_text, is_usable

try:
    import mwparserfromhell  # type: ignore
except Exception:
    mwparserfromhell = None


def _strip_wikimarkup(raw: str) -> str:
    if mwparserfromhell is not None:
        try:
            code = mwparserfromhell.parse(raw)
            return code.strip_code()
        except Exception:
            pass
    # Fallback: remove templates and brackets crudely (not perfect, but deterministic)
    text = raw
    text = text.replace("{{", " ").replace("}}", " ")
    text = text.replace("[[", " ").replace("]]", " ")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikipedia_xml_bz2", required=True, help="Path to Wikipedia pages-articles*.xml.bz2")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL docs")
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_pages", type=int, default=0, help="Optional limit for debugging (0 = no limit)")
    args = ap.parse_args()

    in_path = Path(args.wikipedia_xml_bz2)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pages = 0
    n_written = 0

    # Use iterparse over decompressed stream
    with bz2.open(in_path, "rb") as f, out_path.open("w", encoding="utf-8") as out:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            # Wikipedia dump has namespace; match by suffix
            if not elem.tag.endswith("page"):
                continue

            n_pages += 1
            title_elem = elem.find("./{*}title")
            ns_elem = elem.find("./{*}ns")
            rev_text_elem = elem.find("./{*}revision/{*}text")

            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            ns = ns_elem.text.strip() if ns_elem is not None and ns_elem.text else "0"

            # keep only main namespace (ns=0)
            if ns != "0":
                elem.clear()
                if args.max_pages and n_pages >= args.max_pages:
                    break
                continue

            raw = rev_text_elem.text if rev_text_elem is not None and rev_text_elem.text else ""
            plain = _strip_wikimarkup(raw)
            text = normalize_text((title + " " + plain).strip())

            if is_usable(text, min_chars=args.min_chars):
                doc_id = stable_hash(f"wiki:{title}", prefix="doc:")
                rec = {
                    "doc_id": doc_id,
                    "source": "wikipedia",
                    "title": normalize_text(title),
                    "text": text,
                    "meta": {"ns": ns},
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

            elem.clear()

            if args.max_pages and n_pages >= args.max_pages:
                break

    print(f"[wikipedia] pages_seen={n_pages} docs_written={n_written} -> {out_path}")
    if mwparserfromhell is None:
        print("[wikipedia] NOTE: mwparserfromhell not installed; fallback markup stripping was used.")


if __name__ == "__main__":
    main()
