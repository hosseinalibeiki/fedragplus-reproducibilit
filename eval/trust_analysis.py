"""
Trust analysis utilities: read federated_logs.json and export summary tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="Path to results/federated_logs.json")
    ap.add_argument("--out_csv", required=True, help="Output CSV for trust per round")
    args = ap.parse_args()

    logs = json.loads(Path(args.logs).read_text(encoding="utf-8"))
    rounds = logs.get("rounds", [])

    rows = []
    for r in rounds:
        t = int(r["t"])
        clients = r.get("clients", {})
        for cid, cinfo in clients.items():
            rows.append({
                "round": t,
                "client": int(cid),
                "trust": float(cinfo.get("trust", 1.0)),
                "weight": float(cinfo.get("weight", 1.0)),
            })

    df = pd.DataFrame(rows).sort_values(["round", "client"])
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[trust] -> {out} (rows={len(df)})")


if __name__ == "__main__":
    main()
