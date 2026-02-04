"""
Generate a summary table (CSV) of final trust weights.

Input: federated_logs.json
Output: trust_summary.csv
"""

import json
from pathlib import Path
import csv

def main():
    log_path = Path("results/federated_logs.json")
    logs = json.loads(log_path.read_text())

    last = logs["rounds"][-1]["clients"]

    out = Path("results/trust_summary.csv")
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "final_trust"])
        for cid, rec in last.items():
            writer.writerow([cid, rec["trust"]])

if __name__ == "__main__":
    main()
