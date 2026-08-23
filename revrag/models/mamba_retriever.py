"""
Mamba-2 bi-encoder retriever.

Baseline: reimplementation of Zhang et al. (2024) Mamba Retriever using
Mamba-2 backbone. Encodes full documents without chunking via linear-time
recurrence. Used both as Baseline 4 and as the encoder component of RevRAG.

Architecture:
    - Query encoder: Mamba-2 → mean pool hidden states → L2-normalize
    - Document encoder: same (shared weights, following DPR convention)
    - Training: in-batch negatives + hard negatives, InfoNCE loss
    - Indexing: FAISS flat IP

The key difference from BGE: no chunking. The full document is encoded in a
single forward pass. This is what makes cross-page consistency possible.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attempt to load Mamba-2. Falls back to a simple RNN stub for environments
# where mamba-ssm is not installed (e.g., CPU-only smoke-test machines).
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import MambaLMHeadModel
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning(
        "mamba-ssm not installed. MambaRetriever will use a stub encoder. "
        "Install mamba-ssm (requires CUDA) for real experiments."
    )


class _StubMambaEncoder(nn.Module):
    """Minimal GRU stand-in for smoke tests on CPU-only machines."""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(32000, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        out, _ = self.gru(x)
        return out  # (B, T, H)


class MambaEncoder(nn.Module):
    """
    Mamba-2 encoder wrapper for bi-encoder retrieval.

    Loads a pretrained Mamba-2 language model and extracts hidden states
    for mean-pool encoding. The LM head is discarded.
    """

    def __init__(
        self,
        checkpoint: str = "state-spaces/mamba2-130m",
        hidden_size: int = 768,
        max_length: int = 8192,
        pooling: str = "mean",
    ):
        super().__init__()
        self.max_length = max_length
        self.pooling = pooling
        self.hidden_size = hidden_size

        if MAMBA_AVAILABLE:
            logger.info(f"Loading Mamba-2 from {checkpoint}")
            self.backbone = MambaLMHeadModel.from_pretrained(checkpoint)
            # We only need the backbone, not the LM head
            self.backbone.lm_head = nn.Identity()
        else:
            logger.warning("Using stub encoder (mamba-ssm unavailable)")
            self.backbone = _StubMambaEncoder(hidden_size)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, use_fast=True
            )
        except Exception:
            # Mamba-2 may not have its own tokenizer; fall back to GPT-NeoX
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b", use_fast=True
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass → pooled representation.

        Args:
            input_ids:      (B, T)
            attention_mask: (B, T)

        Returns:
            (B, H) normalized embeddings
        """
        if MAMBA_AVAILABLE:
            outputs = self.backbone(input_ids)
            # MambaLMHeadModel with Identity head returns (B, T, H)
            hidden = outputs.logits if hasattr(outputs, "logits") else outputs
        else:
            hidden = self.backbone(input_ids, attention_mask)

        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(1)
        elif self.pooling == "last":
            pooled = hidden[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return F.normalize(pooled, p=2, dim=-1)

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 8,
        device: str = "cpu",
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts → numpy array (N, H)."""
        self.eval()
        all_embs = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for start in iterator:
                batch = texts[start : start + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc.get("attention_mask", None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)
                embs = self.forward(input_ids, attn_mask)
                all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


class MambaRetriever:
    """
    Full retrieval system: MambaEncoder + FAISS index.

    Usage:
        retriever = MambaRetriever(...)
        retriever.index_documents([(doc_id, text), ...])
        results = retriever.batch_retrieve([(qid, query), ...])
    """

    def __init__(
        self,
        checkpoint: str = "state-spaces/mamba2-130m",
        hidden_size: int = 768,
        max_length: int = 8192,
        pooling: str = "mean",
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.encoder = MambaEncoder(checkpoint, hidden_size, max_length, pooling)
        self.encoder.to(self.device)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.doc_ids: List[str] = []
        self.doc_page_map: Dict[str, int] = {}  # doc_id → approximate page count

    def index_documents(self, documents: List[Tuple[str, str]]) -> None:
        """Encode all documents and build FAISS index."""
        logger.info(f"Indexing {len(documents)} documents with Mamba-2 encoder...")
        self.doc_ids = [d[0] for d in documents]
        texts = [d[1] for d in documents]

        # Store approximate page count (for cross-page analysis)
        for doc_id, text in documents:
            # Approximate: 1 page ≈ 3000 chars
            self.doc_page_map[doc_id] = max(1, len(text) // 3000)

        embeddings = self.encoder.encode_texts(
            texts,
            batch_size=self.batch_size,
            device=self.device,
            show_progress=True,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info(f"FAISS index: {self.index.ntotal} docs, dim={dim}")

    def retrieve(
        self, query: str, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        q_emb = self.encoder.encode_texts(
            [query], batch_size=1, device=self.device, show_progress=False
        )
        scores, indices = self.index.search(q_emb, top_k)
        return [
            (self.doc_ids[i], float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def batch_retrieve(
        self, queries: List[Tuple[str, str]], top_k: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        query_texts = [q[1] for q in queries]
        q_embs = self.encoder.encode_texts(
            query_texts,
            batch_size=self.batch_size,
            device=self.device,
            show_progress=True,
        )
        scores_batch, indices_batch = self.index.search(q_embs, top_k)
        results = {}
        for (qid, _), scores, indices in zip(queries, scores_batch, indices_batch):
            results[qid] = [
                (self.doc_ids[i], float(s))
                for s, i in zip(scores, indices)
                if i >= 0
            ]
        return results


# ---------------------------------------------------------------------------
# InfoNCE training loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """In-batch negative contrastive loss for bi-encoder training."""

    def __init__(self, temperature: float = 0.02):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        q_embs: torch.Tensor,   # (B, H)
        d_embs: torch.Tensor,   # (B, H) — positive docs
    ) -> torch.Tensor:
        # scores: (B, B) — diagonal is positive pair
        scores = torch.matmul(q_embs, d_embs.T) / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)
        return loss
