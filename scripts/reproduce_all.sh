#!/usr/bin/env bash
set -euo pipefail

# Reproduce all main runs (after you implement data + retrieval + federated modules).
# For now, this script validates configs and creates an experiment folder.
#
# Usage:
#   bash scripts/reproduce_all.sh

STAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT="results/run_${STAMP}"
mkdir -p "${ROOT_OUT}"

echo "[reproduce] Output folder: ${ROOT_OUT}"

for CFG in configs/fedavg_rag.yaml configs/fedprox_rag.yaml configs/centralized_rag.yaml configs/fedragplus.yaml; do
  echo "[reproduce] Validating ${CFG}"
  python -m configs.load_config --config "${CFG}" --out "${ROOT_OUT}/merged_$(basename "${CFG}")"
done

echo "[reproduce] Config merge complete."
echo "[reproduce] Next (after implementation): call federated/run_federated.py with each merged config."
echo "[reproduce] Done."
