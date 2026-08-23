"""
src/models/bi_encoder.py
========================
DPR-style bi-encoder built on top of MambaEncoder.

Architecture:
  query_encoder    : MambaEncoder (shared weights with doc encoder by default)
  document_encoder : MambaEncoder
  FAISS index      : flat inner-product (L2-normalised → cosine similarity)

Training:
  InfoNCE contrastive loss with in-batch negatives + hard negatives.
  Temperature τ = 0.02 (standard for dense retrieval).

Inference:
  1. Index all documents (encode + add to FAISS)
  2. Encode query
  3. FAISS search → top-k doc ids + scores
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import MambaEncoder

logger = logging.getLogger(__name__)


class BiEncoder(nn.Module):
    """
    Bi-encoder retriever.

    Parameters
    ----------
    checkpoint    : Mamba-2 or fallback checkpoint
    hidden_size   : embedding dimension
    max_seq_len   : max tokens per sequence
    share_weights : use same encoder for queries and documents (default True)
    device        : "cuda" or "cpu"
    """

    def __init__(
        self,
        checkpoint:    str  = "state-spaces/mamba2-130m",
        hidden_size:   int  = 768,
        max_seq_len:   int  = 4096,
        share_weights: bool = True,
        device:        str  = "cuda",
    ):
        super().__init__()
        self.device       = device
        self.hidden_size  = hidden_size
        self.share_weights = share_weights

        self.query_encoder = MambaEncoder(
            checkpoint=checkpoint,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            device=device,
        )

        if share_weights:
            self.doc_encoder = self.query_encoder
        else:
            self.doc_encoder = MambaEncoder(
                checkpoint=checkpoint,
                hidden_size=hidden_size,
                max_seq_len=max_seq_len,
                device=device,
            )

        # FAISS index (built at index time, not during __init__)
        self._index        = None
        self._doc_ids:     List[str] = []
        self._doc_texts:   List[str] = []

    # ── Forward (used during training) ────────────────────────────────────────

    def forward(
        self,
        q_input_ids:      torch.Tensor,
        q_attention_mask: torch.Tensor,
        d_input_ids:      torch.Tensor,
        d_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (query_embeddings, doc_embeddings), both L2-normalised.
        Shape: (B, hidden_size) each.
        """
        q_emb = self.query_encoder(q_input_ids, q_attention_mask)
        d_emb = self.doc_encoder(d_input_ids, d_attention_mask)
        return q_emb, d_emb

    # ── Indexing (inference) ──────────────────────────────────────────────────

    def build_index(
        self,
        doc_texts: List[str],
        doc_ids:   List[str],
        batch_size: int = 32,
    ):
        """Encode all documents and build a FAISS flat inner-product index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("pip install faiss-gpu  # or faiss-cpu")

        logger.info(f"Building FAISS index for {len(doc_texts)} documents …")
        self._doc_ids   = doc_ids
        self._doc_texts = doc_texts

        embs = self.doc_encoder.encode(doc_texts, batch_size=batch_size)
        embs_np = embs.numpy().astype(np.float32)

        self._index = faiss.IndexFlatIP(self.hidden_size)
        self._index.add(embs_np)
        logger.info(f"  ✓ Index built: {self._index.ntotal} vectors.")

    def retrieve(
        self,
        queries:  List[str],
        top_k:    int = 20,
        batch_size: int = 8,
    ) -> List[List[Dict]]:
        """
        Retrieve top-k documents for each query.

        Returns
        -------
        List of lists, each inner list contains dicts:
          {doc_id, text, score}
        """
        assert self._index is not None, "Call build_index() first."
        q_embs = self.query_encoder.encode(queries, batch_size=batch_size)
        q_np   = q_embs.numpy().astype(np.float32)

        scores, indices = self._index.search(q_np, top_k)

        results = []
        for row_scores, row_idxs in zip(scores, indices):
            hits = []
            for score, idx in zip(row_scores, row_idxs):
                if idx < 0:
                    continue
                hits.append({
                    "doc_id": self._doc_ids[idx],
                    "text":   self._doc_texts[idx],
                    "score":  float(score),
                })
            results.append(hits)
        return results

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.query_encoder.state_dict(), path / "query_encoder.pt")
        if not self.share_weights:
            torch.save(self.doc_encoder.state_dict(), path / "doc_encoder.pt")
        logger.info(f"  ✓ BiEncoder saved → {path}")

    def load(self, path: Path):
        path = Path(path)
        self.query_encoder.load_state_dict(
            torch.load(path / "query_encoder.pt", map_location=self.device)
        )
        if not self.share_weights and (path / "doc_encoder.pt").exists():
            self.doc_encoder.load_state_dict(
                torch.load(path / "doc_encoder.pt", map_location=self.device)
            )
        logger.info(f"  ✓ BiEncoder loaded ← {path}")
