# Fed-RAG+ (Reproducibility Repository)

This repository contains the reproducible pipeline for **Fed-RAG+**, a federated retrieval-augmented generation framework with **trust-weighted aggregation** and **stability regularization**.  
Because the paper targets **robustness and stability** in federated RAG under **free-form generation without reference answers**, this repo focuses on **label-free, reproducible diagnostics** (trust dynamics, JSD stability, and retrieval behavior), and on deterministic dataset construction from **public sources**.

## What this repo provides
- Public-data **corpus builders** (Wikipedia dump, PubMed baseline, StackExchange archive)
- Deterministic **chunking**, **client partitioning**, and **non-IID skewing**
- Controlled **retrieval poisoning** to stress-test robustness
- **MiniLM + FAISS** retrieval pipeline (client-local indexes)
- Federated training loop with **FedAvg-RAG**, **FedProx-RAG**, **Centralized-RAG**, and **Fed-RAG+**
- Evaluation tools:
  - Trust score trajectories and aggregation weights
  - JSD-based agreement stability
  - Label-free retrieval diagnostics (score statistics, overlap, context utilization)

---

## Repository structure
```
fed-rag-plus/
├── configs/
│   ├── fedragplus.yaml
│   ├── fedavg_rag.yaml
│   ├── fedprox_rag.yaml
│   └── centralized_rag.yaml
├── data/
│   ├── build_wikipedia.py
│   ├── build_pubmed.py
│   ├── build_stackexchange.py
│   ├── chunk_documents.py
│   ├── partition_clients.py
│   ├── poison_corpus.py
│   └── utils_text.py
├── prompts/
│   ├── generate_prompts.py
│   └── prompts_seed42.json
├── retrieval/
│   ├── embed_minilm.py
│   ├── build_faiss.py
│   └── search.py
├── federated/
│   ├── client_update.py
│   ├── aggregation.py
│   ├── trust_score.py
│   ├── regularizers.py
│   ├── run_federated.py
│   └── baselines.py
├── eval/
│   ├── jsd.py
│   ├── retrieval_diagnostics.py
│   ├── trust_analysis.py
│   └── make_tables_figures.py
├── scripts/
│   ├── reproduce_all.sh
│   └── quick_smoke_test.sh
├── results/
│   └── (generated outputs: tables, figures, logs)
├── LICENSE
└── README.md
```

---

## Environment setup

### 1) Create a Python environment
Recommended: Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
```

### 2) Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

### Suggested GPU stack
- CUDA-enabled PyTorch
- One A100 80GB was used in the paper; smaller GPUs can run reduced configs.

> If you cannot match GPU hardware, keep the same seeds and report the hardware difference. The pipeline is still reproducible in logic and metrics.

---

## Data sources (public)
This repo is designed to **rebuild data locally**. Do not commit raw dumps to GitHub.

### Wikipedia
Wikimedia dumps (choose one snapshot date and keep it fixed in configs):
- Example (replace date as needed):
```bash
wget https://dumps.wikimedia.org/enwiki/20240101/enwiki-20240101-pages-articles.xml.bz2
```

### PubMed
NCBI PubMed baseline:
```bash
wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed24n0001.xml.gz
# repeat for multiple baseline files if desired
```

### StackExchange
Use Internet Archive StackExchange dumps:
```bash
# Large collection; download only what you need
# You can download specific communities (e.g., stackoverflow.com) via archive.org
```

---

## Step-by-step reproduction

### Step A: Build corpora (clean + deterministic)
1) Convert dumps to cleaned text (document-level)
2) Chunk into passages (deterministic chunk length and overlap)

Example flow:
```bash
python data/build_wikipedia.py --dump_path /path/enwiki.xml.bz2 --out_dir ./data_out/wiki
python data/build_pubmed.py --dump_path /path/pubmed.xml.gz --out_dir ./data_out/pubmed
python data/build_stackexchange.py --dump_path /path/stackexchange.7z --out_dir ./data_out/stackex

python data/chunk_documents.py --in_dir ./data_out/wiki --out_dir ./data_out/wiki_chunks
python data/chunk_documents.py --in_dir ./data_out/pubmed --out_dir ./data_out/pubmed_chunks
python data/chunk_documents.py --in_dir ./data_out/stackex --out_dir ./data_out/stackex_chunks
```

**Reproducibility rule:** chunking parameters must be fixed:
- `chunk_tokens` (e.g., 384–512)
- `overlap_tokens` (e.g., 50)
- tokenizer choice (must be recorded)

### Step B: Create federated clients (non-IID)
We recommend 4 clients aligned with domain bias:
- Client 1: Wikipedia-heavy (general)
- Client 2: PubMed-heavy (scientific)
- Client 3: StackExchange-heavy (technical)
- Client 4: Mixed + poisoned retrieval set

```bash
python data/partition_clients.py \
  --wiki ./data_out/wiki_chunks \
  --pubmed ./data_out/pubmed_chunks \
  --stackex ./data_out/stackex_chunks \
  --num_clients 4 \
  --seed 42 \
  --zipf_alpha 1.2 \
  --docs_per_client 22000 \
  --out_dir ./federated_data
```

### Step C: Inject retrieval poisoning (controlled)
Poison only the **retrieval corpus** for the selected client to simulate retrieval corruption.
```bash
python data/poison_corpus.py \
  --client_dir ./federated_data/client_4 \
  --poison_rate 0.20 \
  --poison_type shuffle_or_offtopic \
  --seed 42
```

---

## Retrieval pipeline (MiniLM + FAISS)

### Step D: Embed and index (per client)
```bash
python retrieval/embed_minilm.py --client_dir ./federated_data/client_1 --out_dir ./indexes/client_1
python retrieval/build_faiss.py  --emb_dir ./indexes/client_1 --out_index ./indexes/client_1/faiss.index

# repeat for all clients
```

**Reproducibility rule:** record
- MiniLM checkpoint name
- embedding normalization
- FAISS index type + parameters (e.g., IVF, HNSW, Flat)

---

## Probe prompts (unlabeled, fixed seed)

### Step E: Generate prompts
This is an unlabeled probe set used for stability measurements. It does not require answers.
```bash
python prompts/generate_prompts.py \
  --sources ./federated_data \
  --n_prompts 300 \
  --seed 42 \
  --out ./prompts/prompts_seed42.json
```

**Important:** keep prompts fixed across methods and rounds.

---

## Run experiments (federated training)

### Step F: Execute baselines and Fed-RAG+
```bash
python federated/run_federated.py --config configs/fedavg_rag.yaml
python federated/run_federated.py --config configs/fedprox_rag.yaml
python federated/run_federated.py --config configs/centralized_rag.yaml
python federated/run_federated.py --config configs/fedragplus.yaml
```

Each run should produce:
- per-round logs (trust, weights, stability)
- retrieval diagnostics logs
- final tables/figures in `./results/`

---

## Evaluation (label-free and reproducible)

### Metrics supported without labels
Because we do not assume reference answers or relevance annotations, we evaluate:
1) **Trust dynamics:** trust score trajectory per client and final trust value  
2) **Agreement stability (JSD):** divergence of attribution-based distributions across rounds/methods  
3) **Retrieval diagnostics:**  
   - mean / std of top-k similarity scores  
   - context utilization after truncation  
   - overlap@k between rounds (Jaccard on retrieved doc IDs)

Generate tables/figures:
```bash
python eval/make_tables_figures.py --run_dir ./results/latest --out_dir ./results/figures
```

---

## Reproducibility checklist (what you should record in the paper)
- Snapshot dates for Wikipedia / PubMed / StackExchange
- Exact chunking parameters and tokenizer
- Partition seed and non-IID settings (Zipf alpha)
- Poisoning rate and poisoning type
- Retriever checkpoint and FAISS index configuration
- Federated config: rounds, local epochs, batch size, learning rate
- Hardware and runtime (optional but recommended)

---

## Notes on claims and scope
This repository supports **robustness and stability** claims under free-form generation without references.  
It does **not** claim improved answer correctness. If you later add reference answers or human evaluation, add them as a new experiment with clear protocols and licensing.

---

## License
Choose a permissive license (recommended: MIT or Apache-2.0).  
Do not redistribute raw dumps; only provide scripts to download and process public sources.

---

## Citation
If you use this repository, please cite the accompanying paper:

```bibtex
@article{fedragplus2026,
  title   = {Fed-RAG+: A Trust-Weighted Federated Fine-Tuning Framework for Retrieval-Augmented LLMs},
  author  = {<Authors>},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Code and reproducibility pipeline: GitHub repository ``fed-rag-plus''}
}
```

---

## Contact
For questions about reproduction, open an issue in this repository and include:
- the config file you used
- logs under `results/`
- your environment details (GPU/CPU, CUDA, PyTorch versions)
