# Probe Prompts (Unlabeled)

This folder supports an **unlabeled probe evaluation** for federated retrieval-augmented generation.

## Why prompts are unlabeled
The paper evaluates **stability and robustness** (trust dynamics, JSD agreement stability, and label-free retrieval diagnostics) under **free-form generation** without reference answers. Therefore, the evaluation uses a fixed set of **unlabeled probe prompts**.

## How prompts are generated
`generate_prompts.py` constructs prompts deterministically from public corpora or federated chunks:

- Extract topic candidates from titles (preferred) or first sentences.
- Apply a fixed set of prompt templates.
- Use a fixed RNG seed to ensure the same prompts are generated across runs.

## Recommended workflow
Generate prompts from the federated clients to ensure domain coverage:

```bash
python prompts/generate_prompts.py \
  --sources federated_data/client_1/chunks.jsonl federated_data/client_2/chunks.jsonl federated_data/client_3/chunks.jsonl federated_data/client_4/chunks.jsonl \
  --n_prompts 300 --seed 42 --out prompts/prompts_seed42.json
```

Commit the resulting `prompts_seed42.json` to the repository to freeze the evaluation set.
