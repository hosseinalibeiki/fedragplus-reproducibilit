"""
Plot trust evolution across federated rounds.

Input: federated_logs.json
Output: trust_curve.png + trust_curve.pdf
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    log_path = Path("results/federated_logs.json")
    logs = json.loads(log_path.read_text())

    rounds = []
    trust = {}

    for r in logs["rounds"]:
        t = r["t"]
        rounds.append(t)
        for cid, rec in r["clients"].items():
            trust.setdefault(cid, []).append(rec["trust"])

    for cid, vals in trust.items():
        plt.plot(rounds, vals, label=f"Client {cid}")

    plt.xlabel("Federated round")
    plt.ylabel("Trust weight")
    plt.title("Trust evolution across rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/trust_curve.png", dpi=300)
    plt.savefig("results/trust_curve.pdf")
    plt.close()

if __name__ == "__main__":
    main()
