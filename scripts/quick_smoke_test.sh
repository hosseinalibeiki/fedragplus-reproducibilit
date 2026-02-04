#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test: verifies that YAML includes merge correctly and that the
# project output directory can be created deterministically.
#
# Usage:
#   bash scripts/quick_smoke_test.sh configs/fedragplus.yaml

CFG="${1:-configs/fedragplus.yaml}"

echo "[smoke] Using config: ${CFG}"

python -m configs.load_config --config "${CFG}" --out /tmp/merged_config.yaml

echo "[smoke] Merged config written to /tmp/merged_config.yaml"
echo "[smoke] Top-level keys:"
python - <<'PY'
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path("/tmp/merged_config.yaml").read_text())
print(sorted(list(cfg.keys())))
PY

OUTDIR=$(python - <<'PY'
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path("/tmp/merged_config.yaml").read_text())
print(cfg["project"]["output_dir"])
PY
)

mkdir -p "${OUTDIR}"
echo "[smoke] OK: created output dir '${OUTDIR}'"
echo "[smoke] Done."
