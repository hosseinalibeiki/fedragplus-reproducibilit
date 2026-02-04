#!/usr/bin/env bash
set -euo pipefail

# End-to-end reproducible pipeline for Fed-RAG+
#
# Assumptions:
# - You have downloaded the raw sources (Wikipedia XML bz2, PubMed XML gz, StackExchange .7z)
# - You update configs/base.yaml data.raw_paths with correct local paths
#
# Usage:
#   bash scripts/reproduce_pipeline.sh configs/fedragplus.yaml
#
# Notes:
# - This script does NOT download raw dumps for you.
# - It rebuilds everything deterministically from local raw files.

CFG="${1:-configs/fedragplus.yaml}"

echo "[pipeline] Config: ${CFG}"

# Merge config to a temp file
MERGED="/tmp/fedragplus_merged.yaml"
python -m configs.load_config --config "${CFG}" --out "${MERGED}"

# Extract required paths + parameters from merged config
PY_GET='python - <<PY
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("'"${MERGED}"'").read_text())
print(cfg)
PY'

# Use python snippets to read values
WIKI_XML=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(cfg["data"]["raw_paths"]["wikipedia_multistream_xml_bz2"])
PY
)
PUBMED_XML=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(cfg["data"]["raw_paths"]["pubmed_xml_gz"])
PY
)
STACKEX_7Z=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(cfg["data"]["raw_paths"]["stackexchange_7z"])
PY
)

CHUNK_TOK=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(int(cfg["data"]["chunking"]["chunk_tokens"]))
PY
)
OVERLAP_TOK=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(int(cfg["data"]["chunking"]["overlap_tokens"]))
PY
)
MIN_CHARS=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(int(cfg["data"]["chunking"]["min_chars"]))
PY
)

NCLIENTS=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(int(cfg["data"]["clients"]["num_clients"]))
PY
)
DOCS_PER_CLIENT=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(int(cfg["data"]["clients"]["docs_per_client"]))
PY
)
ZIPF_ALPHA=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(float(cfg["data"]["clients"]["non_iid"]["zipf_alpha"]))
PY
)
SEED=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(int(cfg["project"]["seed"]))
PY
)

POISON_ENABLED=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(bool(cfg["poisoning"]["enabled"]))
PY
)
POISON_CLIENT=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(int(cfg["poisoning"]["poisoned_client_id"]))
PY
)
POISON_RATE=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(float(cfg["poisoning"]["poison_rate"]))
PY
)
POISON_TYPE=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(str(cfg["poisoning"]["poison_type"]))
PY
)

TOPK=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(int(cfg["retrieval"]["top_k"]))
PY
)
EMBED_MODEL=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(str(cfg["retrieval"]["embedder"]["name"]))
PY
)
NORM_EMB=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(bool(cfg["retrieval"]["embedder"].get("normalize_embeddings", True)))
PY
)

OUTDIR=$(python - <<'PY'
import yaml, pathlib
cfg=yaml.safe_load(pathlib.Path("/tmp/fedragplus_merged.yaml").read_text())
print(str(cfg["project"]["output_dir"]))
PY
)

echo "[pipeline] Raw paths:"
echo "  wiki   = ${WIKI_XML}"
echo "  pubmed = ${PUBMED_XML}"
echo "  stack  = ${STACKEX_7Z}"

mkdir -p data_out federated_data indexes "${OUTDIR}"

echo "[pipeline] 1) Build docs"
python data/build_wikipedia.py   --wikipedia_xml_bz2 "${WIKI_XML}"   --out_jsonl data_out/wiki_docs.jsonl   --min_chars "${MIN_CHARS}"
python data/build_pubmed.py      --pubmed_xml_gz     "${PUBMED_XML}" --out_jsonl data_out/pubmed_docs.jsonl --min_chars "${MIN_CHARS}"
python data/build_stackexchange.py --stackexchange_7z "${STACKEX_7Z}" --out_jsonl data_out/stackex_docs.jsonl --min_chars "${MIN_CHARS}"

echo "[pipeline] 2) Chunk docs"
python data/chunk_documents.py --in_jsonl data_out/wiki_docs.jsonl    --out_jsonl data_out/wiki_chunks.jsonl    --chunk_tokens "${CHUNK_TOK}" --overlap_tokens "${OVERLAP_TOK}" --min_chars "${MIN_CHARS}"
python data/chunk_documents.py --in_jsonl data_out/pubmed_docs.jsonl  --out_jsonl data_out/pubmed_chunks.jsonl  --chunk_tokens "${CHUNK_TOK}" --overlap_tokens "${OVERLAP_TOK}" --min_chars "${MIN_CHARS}"
python data/chunk_documents.py --in_jsonl data_out/stackex_docs.jsonl --out_jsonl data_out/stackex_chunks.jsonl --chunk_tokens "${CHUNK_TOK}" --overlap_tokens "${OVERLAP_TOK}" --min_chars "${MIN_CHARS}"

echo "[pipeline] 3) Partition clients"
python data/partition_clients.py \
  --chunk_jsonl data_out/wiki_chunks.jsonl data_out/pubmed_chunks.jsonl data_out/stackex_chunks.jsonl \
  --out_dir federated_data \
  --num_clients "${NCLIENTS}" \
  --docs_per_client "${DOCS_PER_CLIENT}" \
  --seed "${SEED}" \
  --non_iid zipf \
  --zipf_alpha "${ZIPF_ALPHA}" \
  --client_mix wikipedia,pubmed,stackexchange,mixed

if [ "${POISON_ENABLED}" = "True" ] || [ "${POISON_ENABLED}" = "true" ]; then
  echo "[pipeline] 4) Poison client ${POISON_CLIENT}"
  python data/poison_corpus.py \
    --client_dir "federated_data/client_${POISON_CLIENT}" \
    --poison_rate "${POISON_RATE}" \
    --poison_type "${POISON_TYPE}" \
    --seed "${SEED}"
fi

echo "[pipeline] 5) Embed + index per client"
for i in $(seq 1 "${NCLIENTS}"); do
  echo "  [client_${i}] embed"
  EMB_DIR="indexes/client_${i}"
  mkdir -p "${EMB_DIR}"
  if [ "${NORM_EMB}" = "True" ] || [ "${NORM_EMB}" = "true" ]; then
    python retrieval/embed_minilm.py --chunks_jsonl "federated_data/client_${i}/chunks.jsonl" --out_dir "${EMB_DIR}" --model_name "${EMBED_MODEL}" --normalize_embeddings
  else
    python retrieval/embed_minilm.py --chunks_jsonl "federated_data/client_${i}/chunks.jsonl" --out_dir "${EMB_DIR}" --model_name "${EMBED_MODEL}"
  fi
  echo "  [client_${i}] index"
  python retrieval/build_faiss.py --embeddings_npy "${EMB_DIR}/embeddings.npy" --ids_jsonl "${EMB_DIR}/ids.jsonl" --out_dir "${EMB_DIR}" --index_type FlatIP
done

echo "[pipeline] 6) Generate probe prompts (commit prompts/prompts_seed42.json once created)"
python prompts/generate_prompts.py \
  --sources federated_data/client_1/chunks.jsonl federated_data/client_2/chunks.jsonl federated_data/client_3/chunks.jsonl federated_data/client_4/chunks.jsonl \
  --n_prompts 300 --seed "${SEED}" --out prompts/prompts_seed42.json

echo "[pipeline] 7) Run federated training"
python federated/run_federated.py --config "${CFG}" --prompts prompts/prompts_seed42.json --federated_data federated_data --indexes indexes --out_dir "${OUTDIR}"

echo "[pipeline] 8) Generate tables + figures"
python -m eval.make_tables_figures --logs "${OUTDIR}/federated_logs.json" --out_root "${OUTDIR}"

echo "[pipeline] DONE: outputs in ${OUTDIR}"
