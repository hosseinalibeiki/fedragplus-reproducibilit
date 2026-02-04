"""
Aggregate retrieval diagnostics produced by retrieval/retrieval_diagnostics.py
and format them for paper tables.
"""

import json
from pathlib import Path
import csv

def main():
    diag_dir = Path("results/diagnostics")
    rows = []

    for p in diag_dir.glob("*.json"):
        d = json.loads(p.read_text())
        rows.append({
            "index": p.stem,
            "score_mean": d["score_mean"],
            "score_std": d["score_std"],
        })

    out = Path("results/retrieval_summary.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "score_mean", "score_std"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
