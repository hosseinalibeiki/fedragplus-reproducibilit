"""
Build StackExchange corpus from a community .7z archive containing Posts.xml.

Input: --stackexchange_7z PATH (e.g., ai.stackexchange.com.7z)
Output: docs JSONL with fields: doc_id, title, text, source, meta

Notes:
- Extracts Posts.xml to a temp folder (deterministic, no random sampling here).
- Uses only questions (PostTypeId=1) by default; include answers optionally.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from .utils_text import stable_hash, normalize_text, is_usable

try:
    import py7zr  # type: ignore
except Exception as e:
    py7zr = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stackexchange_7z", required=True, help="Path to stackexchange community .7z")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL docs")
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--include_answers", action="store_true", help="Also include answers as separate docs")
    args = ap.parse_args()

    if py7zr is None:
        raise RuntimeError("py7zr is required to read .7z files. Install with: pip install py7zr")

    in_path = Path(args.stackexchange_7z)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_seen = 0

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with py7zr.SevenZipFile(in_path, mode="r") as z:
            z.extractall(path=td_path)

        # Find Posts.xml
        posts = list(td_path.rglob("Posts.xml"))
        if not posts:
            raise FileNotFoundError("Posts.xml not found inside archive")
        posts_path = posts[0]

        # Streaming parse: <row ... />
        with out_path.open("w", encoding="utf-8") as out:
            for event, elem in ET.iterparse(posts_path, events=("end",)):
                if elem.tag != "row":
                    continue
                n_seen += 1
                attrs = elem.attrib
                post_type = attrs.get("PostTypeId", "")
                if post_type == "1":  # question
                    post_id = attrs.get("Id", f"q_{n_seen}")
                    title = normalize_text(attrs.get("Title", ""))
                    body = normalize_text(attrs.get("Body", ""))
                    text = normalize_text((title + " " + body).strip())
                    if is_usable(text, min_chars=args.min_chars):
                        doc_id = stable_hash(f"stackex:{post_id}", prefix="doc:")
                        rec = {
                            "doc_id": doc_id,
                            "source": "stackexchange",
                            "title": title,
                            "text": text,
                            "meta": {"post_id": post_id, "post_type": "question"},
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_written += 1

                elif args.include_answers and post_type == "2":  # answer
                    post_id = attrs.get("Id", f"a_{n_seen}")
                    body = normalize_text(attrs.get("Body", ""))
                    text = body
                    if is_usable(text, min_chars=args.min_chars):
                        doc_id = stable_hash(f"stackex:{post_id}", prefix="doc:")
                        rec = {
                            "doc_id": doc_id,
                            "source": "stackexchange",
                            "title": "",
                            "text": text,
                            "meta": {"post_id": post_id, "post_type": "answer"},
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_written += 1
                elem.clear()

    print(f"[stackexchange] rows_seen={n_seen} docs_written={n_written} -> {out_path}")


if __name__ == "__main__":
    main()
