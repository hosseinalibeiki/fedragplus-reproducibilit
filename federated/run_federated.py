"""
Federated experiment runner.

Pipeline:
1) Load merged config (supports include via configs/load_config.py).
2) Assume federated_data/client_i/chunks.jsonl exists.
3) Assume indexes/client_i/faiss.index and ids.jsonl exist.
4) Load probe prompts (prompts/prompts_seed42.json).
5) Run rounds:
   - local_update per client (optionally with KL stability regularizer)
   - compute trust scores from retrieval distributions
   - aggregate global model with trust weights (Fed-RAG+), or uniform (FedAvg), etc.
6) Save logs for reproducibility.

This runner is designed to be *honest* and *reproducible* in an unlabeled setting.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configs.load_config import load_config

from .client_update import LocalTrainConfig, local_update
from .regularizers import KLRegConfig
from .aggregation import weighted_average_state_dict
from .trust_score import TrustConfig, trust_from_hist, update_trust_ema, clip_trust


def _read_prompts(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("prompts file must be a JSON list of strings")


def _tokenize_probe(tokenizer, prompts: List[str], max_len: int = 256) -> Dict[str, torch.Tensor]:
    # Create a small unlabeled probe batch for KL stability
    enc = tokenizer(
        prompts,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _retrieval_histograms(index_dir: Path, prompts: List[str], top_k: int, embed_model_name: str, normalize_embeddings: bool) -> np.ndarray:
    """
    Build a histogram over retrieved positions for the prompts.
    We use retrieved *indices* (0..N-1) as categories to avoid needing document text.
    """
    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("faiss and sentence-transformers are required for trust histograms.") from e

    index = faiss.read_index(str(index_dir / "faiss.index"))
    model = SentenceTransformer(embed_model_name)

    counts = np.zeros((index.ntotal,), dtype=np.float64) if index.ntotal > 0 else np.zeros((1,), dtype=np.float64)

    for p in prompts:
        q = p.strip()
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=normalize_embeddings).astype(np.float32)
        _, idxs = index.search(q_emb, top_k)
        for idx in idxs[0].tolist():
            if idx >= 0 and idx < counts.shape[0]:
                counts[idx] += 1.0

    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (can include: [...])")
    ap.add_argument("--prompts", default="prompts/prompts_seed42.json")
    ap.add_argument("--federated_data", default="federated_data")
    ap.add_argument("--indexes", default="indexes")
    ap.add_argument("--out_dir", default=None, help="Override project.output_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(args.out_dir or cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(Path(args.prompts))
    # Use a small deterministic subset for KL probe
    probe_prompts = prompts[: min(16, len(prompts))]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["generator_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    global_model = AutoModelForCausalLM.from_pretrained(cfg["model"]["generator_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    global_model.eval()

    # configs
    rounds = int(cfg["federated"]["rounds"])
    local_epochs = int(cfg["federated"]["local_epochs"])
    top_k = int(cfg["retrieval"]["top_k"])
    embed_model_name = cfg["retrieval"]["embedder"]["name"]
    normalize_embeddings = bool(cfg["retrieval"]["embedder"].get("normalize_embeddings", True))

    train_cfg = LocalTrainConfig(
        model_name=cfg["model"]["generator_name"],
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
        batch_size=int(cfg["optim"].get("batch_size", 2)),
        max_length=int(cfg["retrieval"]["max_input_tokens"]),
        local_epochs=local_epochs,
        fp16=bool(cfg["compute"].get("fp16", True)),
        grad_clip_norm=float(cfg["optim"].get("grad_clip_norm", 1.0)),
        seed=seed,
        top_k=top_k,
    )

    kl_cfg = KLRegConfig(
        enabled=bool(cfg.get("regularizers", {}).get("kl", {}).get("enabled", False)),
        lambda_kl=float(cfg.get("regularizers", {}).get("kl", {}).get("lambda_kl", 0.0)),
        temperature=float(cfg.get("regularizers", {}).get("kl", {}).get("temperature", 1.0)),
    )

    trust_cfg = TrustConfig(
        enabled=bool(cfg.get("trust", {}).get("enabled", False)),
        smoothing=float(cfg.get("trust", {}).get("smoothing", 0.6)),
        clip_min=float(cfg.get("trust", {}).get("clip_min", 0.05)),
        clip_max=float(cfg.get("trust", {}).get("clip_max", 1.0)),
        score_mode=str(cfg.get("trust", {}).get("score_mode", "retrieval_consistency")),
    )

    # probe batch for KL stability
    probe_batch = _tokenize_probe(tokenizer, probe_prompts, max_len=256)

    # Trust initialization
    num_clients = int(cfg["data"]["clients"]["num_clients"])
    trust = {i: 1.0 for i in range(1, num_clients + 1)}

    logs = {
        "config": cfg,
        "rounds": [],
    }

    for t in range(1, rounds + 1):
        print(f"[round {t}/{rounds}]")

        client_states = []
        client_weights = []

        # Build global retrieval histogram for trust reference (mean of client hists)
        client_hists = []
        for i in range(1, num_clients + 1):
            idx_dir = Path(args.indexes) / f"client_{i}"
            h = _retrieval_histograms(idx_dir, prompts[: min(50, len(prompts))], top_k, embed_model_name, normalize_embeddings)
            client_hists.append(h)
        global_hist = np.mean(np.stack(client_hists, axis=0), axis=0)

        round_rec = {"t": t, "clients": {}}

        for i in range(1, num_clients + 1):
            client_dir = Path(args.federated_data) / f"client_{i}"
            idx_dir = Path(args.indexes) / f"client_{i}"

            # local training
            sd = local_update(
                global_model=global_model,
                tokenizer=tokenizer,
                client_chunks_jsonl=client_dir / "chunks.jsonl",
                client_index_dir=idx_dir,
                train_cfg=train_cfg,
                kl_cfg=kl_cfg,
                probe_batch=probe_batch if kl_cfg.enabled else None,
            )
            client_states.append(sd)

            # trust update
            if trust_cfg.enabled:
                client_hist = client_hists[i - 1]
                raw = trust_from_hist(client_hist, global_hist)
                trust[i] = clip_trust(update_trust_ema(trust[i], raw, trust_cfg.smoothing), trust_cfg.clip_min, trust_cfg.clip_max)
                w = trust[i]
            else:
                w = 1.0

            client_weights.append(w)
            round_rec["clients"][str(i)] = {"trust": float(trust[i]), "weight": float(w)}

        # aggregate
        avg_sd = weighted_average_state_dict(client_states, client_weights)
        global_model.load_state_dict(avg_sd, strict=True)
        global_model.eval()

        round_rec["agg_weights"] = [float(w) for w in client_weights]
        logs["rounds"].append(round_rec)

        # persist every round for crash safety
        (out_dir / "federated_logs.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
        print(f"[round {t}] saved -> {out_dir/'federated_logs.json'}")

    print(f"[done] logs at {out_dir/'federated_logs.json'}")


if __name__ == "__main__":
    main()
