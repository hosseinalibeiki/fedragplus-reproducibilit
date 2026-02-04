"""
Generate paper-ready figures (matplotlib) from results artifacts.

Figures:
- Trust over rounds (per client)
- Aggregation weights over rounds (per client)
- Retrieval diagnostics summary (if provided)

This script does not set custom colors/styles, to keep it portable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_trust(csv_path: Path, out_png: Path) -> None:
    df = pd.read_csv(csv_path)
    plt.figure()
    for cid, g in df.groupby("client"):
        plt.plot(g["round"], g["trust"], label=f"Client {cid}")
    plt.xlabel("Round")
    plt.ylabel("Trust score")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_weights(csv_path: Path, out_png: Path) -> None:
    df = pd.read_csv(csv_path)
    plt.figure()
    for cid, g in df.groupby("client"):
        plt.plot(g["round"], g["weight"], label=f"Client {cid}")
    plt.xlabel("Round")
    plt.ylabel("Aggregation weight")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trust_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_trust(Path(args.trust_csv), out_dir / "fig_trust_over_rounds.png")
    plot_weights(Path(args.trust_csv), out_dir / "fig_weights_over_rounds.png")

    print(f"[plots] saved figures to {out_dir}")


if __name__ == "__main__":
    main()
