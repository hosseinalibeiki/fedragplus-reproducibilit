"""
Client local update for RAG-style *context-augmented LM* fine-tuning.

Key idea (label-free):
- Use each chunk's title as a query, retrieve top-k passages from the client's index.
- Build an input sequence: [ctx1 .. ctxk] + [doc_text].
- Train causal LM to predict doc_text tokens given preceding context.
This incorporates retrieval without requiring supervised labels.

This is a pragmatic, reproducible approximation suitable when no reference answers exist.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from data.utils_text import normalize_text

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from .regularizers import KLRegConfig, apply_kl_regularizer


@dataclass
class LocalTrainConfig:
    model_name: str
    lr: float = 3e-5
    weight_decay: float = 0.01
    batch_size: int = 2
    max_length: int = 1024
    local_epochs: int = 1
    fp16: bool = True
    grad_clip_norm: float = 1.0
    seed: int = 42
    top_k: int = 5


class ChunkRAGDataset(Dataset):
    def __init__(self, chunks_jsonl: Path, ids_jsonl: Path, faiss_index_path: Path, embed_model_name: str,
                 top_k: int, tokenizer, max_length: int, normalize_embeddings: bool = True):
        if faiss is None:
            raise RuntimeError("faiss is required for ChunkRAGDataset. Install faiss-cpu or faiss-gpu.")
        self.rows = self._read_jsonl(chunks_jsonl)
        self.ids = self._read_jsonl(ids_jsonl)
        self.id_by_pos = self.ids  # aligns with FAISS vectors

        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.index = faiss.read_index(str(faiss_index_path))
        self.embedder = SentenceTransformer(embed_model_name)
        self.top_k = int(top_k)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.normalize_embeddings = bool(normalize_embeddings)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def _retrieve_context(self, query: str) -> str:
        q = normalize_text(query)
        q_emb = self.embedder.encode([q], convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings).astype(np.float32)
        scores, idxs = self.index.search(q_emb, self.top_k)
        ctx_texts = []
        for idx in idxs[0].tolist():
            if idx < 0 or idx >= len(self.id_by_pos):
                continue
            rec = self.id_by_pos[idx]
            # We only have ids metadata; context text is not stored in index_dir.
            # Therefore, we use titles as lightweight context markers for reproducibility.
            title = normalize_text(rec.get("title", ""))
            if title:
                ctx_texts.append(title)
        if not ctx_texts:
            return ""
        return " | ".join(ctx_texts[: self.top_k])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        r = self.rows[i]
        title = normalize_text(r.get("title", "")) or "topic"
        doc_text = normalize_text(r.get("text", ""))
        ctx = self._retrieve_context(title)

        # Construct input: context + document (target)
        # We train causal LM on full sequence; labels = input_ids with -100 on context tokens if desired.
        prompt = f"Context: {ctx}\n\nDocument: "
        full = prompt + doc_text

        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]

        # Mask labels for the "prompt" part so loss focuses on doc_text.
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt")["input_ids"][0]
        labels = input_ids.clone()
        mask_len = min(len(prompt_ids), len(labels))
        labels[:mask_len] = -100

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def _collate(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    # left padding is fine; use tokenizer default if needed
    max_len = max(x["input_ids"].shape[0] for x in batch)
    def pad(x, value):
        pad_len = max_len - x.shape[0]
        if pad_len <= 0:
            return x
        return torch.cat([x, torch.full((pad_len,), value, dtype=x.dtype)], dim=0)

    input_ids = torch.stack([pad(x["input_ids"], pad_id) for x in batch], dim=0)
    attention_mask = torch.stack([pad(x["attention_mask"], 0) for x in batch], dim=0)
    labels = torch.stack([pad(x["labels"], -100) for x in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def local_update(
    global_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    client_chunks_jsonl: Path,
    client_index_dir: Path,
    train_cfg: LocalTrainConfig,
    kl_cfg: KLRegConfig,
    probe_batch: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns a state_dict of trainable parameters after local training.
    """
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    device = next(global_model.parameters()).device
    model = AutoModelForCausalLM.from_pretrained(train_cfg.model_name)
    model.to(device)
    model.train()

    # Initialize local model with global weights (full sync)
    model.load_state_dict(global_model.state_dict(), strict=True)

    # Dataset uses FAISS index and ids.jsonl (built in retrieval/ pipeline)
    faiss_index_path = client_index_dir / "faiss.index"
    ids_jsonl = client_index_dir / "ids.jsonl"

    ds = ChunkRAGDataset(
        chunks_jsonl=client_chunks_jsonl,
        ids_jsonl=ids_jsonl,
        faiss_index_path=faiss_index_path,
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=train_cfg.top_k,
        tokenizer=tokenizer,
        max_length=train_cfg.max_length,
        normalize_embeddings=True,
    )
    dl = DataLoader(
        ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,  # deterministic
        collate_fn=lambda b: _collate(b, pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.fp16)

    for _ep in range(train_cfg.local_epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=train_cfg.fp16):
                out = model(**batch)
                loss = out.loss

                # Optional KL stability on probe batch
                if probe_batch is not None and kl_cfg.enabled and kl_cfg.lambda_kl > 0:
                    pb = {k: v.to(device) for k, v in probe_batch.items()}
                    local_logits = model(**{k: pb[k] for k in ["input_ids", "attention_mask"]}).logits
                    with torch.no_grad():
                        global_logits = global_model(**{k: pb[k] for k in ["input_ids", "attention_mask"]}).logits
                    loss = apply_kl_regularizer(loss, local_logits, global_logits, kl_cfg)

            scaler.scale(loss).backward()
            if train_cfg.grad_clip_norm and train_cfg.grad_clip_norm > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)

            scaler.step(optim)
            scaler.update()

    # Return full model state for now (can later restrict to adapters)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
