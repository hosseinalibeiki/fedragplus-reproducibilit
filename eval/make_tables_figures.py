"""
Single entry point to produce tables + figures for the paper.

Inputs:
- federated logs (results/federated_logs.json)

Outputs:
- results/tables/trust_over_rounds.csv
- results/figures/fig_trust_over_rounds.png
- results/figures/fig_weights_over_rounds.png

This keeps the "Results" section grounded: every figure/table is generated from code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .trust_analysis import main as trust_main
from .plots import main as plots_main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="Path to federated_logs.json")
    ap.add_argument("--out_root", required=True, help="Root output dir, e.g., results")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    tables = out_root / "tables"
    figs = out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # run trust_analysis
    trust_csv = tables / "trust_over_rounds.csv"
    import sys
    argv_bak = sys.argv
    try:
        sys.argv = ["trust_analysis.py", "--logs", str(args.logs), "--out_csv", str(trust_csv)]
        trust_main()
    finally:
        sys.argv = argv_bak

    # run plots
    argv_bak = sys.argv
    try:
        sys.argv = ["plots.py", "--trust_csv", str(trust_csv), "--out_dir", str(figs)]
        plots_main()
    finally:
        sys.argv = argv_bak

    print("[make] done.")


if __name__ == "__main__":
    main()
