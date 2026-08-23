"""
src/models/encoder.py
=====================
Mamba-2 sequence encoder for dense retrieval.

On GPU (Kaggle T4):
    Uses state-spaces/mamba2-130m via HuggingFace transformers.
    Linear-time O(T) encoding — no attention, no chunking needed.

On CPU (local dev):
    Automatically falls back to bert-base-uncased with identical interface.
    Swap happens transparently — no code changes needed.

Output:
    Fixed-size embedding (hidden_size,) per sequence, L2-normalised.
    Ready for dot-product similarity / FAISS inner-product search.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MAMBA2_CHECKPOINT = "state-spaces/mamba2-130m"
FALLBACK_CHECKPOINT = "bert-base-uncased"


class MambaEncoder(nn.Module):
    """
    Sequence encoder backed by Mamba-2 (GPU) or BERT (CPU fallback).

    Parameters
    ----------
    checkpoint : str
        HuggingFace model ID. Defaults to mamba2-130m.
    hidden_size : int
        Output embedding dimension. Projects to this if model dim differs.
    max_seq_len : int
        Maximum input tokens. Mamba-2 handles this natively; BERT is capped at 512.
    device : str
    """

    def __init__(
        self,
        checkpoint:   str = MAMBA2_CHECKPOINT,
        hidden_size:  int = 768,
        max_seq_len:  int = 4096,
        device:       str = "cuda",
    ):
        super().__init__()
        self.device      = device
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self._is_mamba   = False

        self.backbone, self.tokenizer, actual_dim = self._load_backbone(
            checkpoint, device
        )

        # Project to hidden_size if backbone dim differs
        self.proj = (
            nn.Linear(actual_dim, hidden_size, bias=False)
            if actual_dim != hidden_size
            else nn.Identity()
        )
        self.to(device)

    # ── Backbone loading ──────────────────────────────────────────────────────

    def _load_backbone(self, checkpoint: str, device: str):
        """Try Mamba-2, fall back to BERT on CPU or import failure."""
        if device == "cpu":
            logger.info("CPU device — using BERT fallback encoder.")
            return self._load_bert(FALLBACK_CHECKPOINT)

        try:
            return self._load_mamba2(checkpoint)
        except Exception as e:
            logger.warning(
                f"Mamba-2 load failed ({e}). "
                f"Falling back to {FALLBACK_CHECKPOINT}."
            )
            return self._load_bert(FALLBACK_CHECKPOINT)

    def _load_mamba2(self, checkpoint: str):
        from transformers import AutoTokenizer, Mamba2Model  # type: ignore
        logger.info(f"Loading Mamba-2 encoder: {checkpoint}")
        tok   = AutoTokenizer.from_pretrained(checkpoint)
        model = Mamba2Model.from_pretrained(checkpoint)
        dim   = model.config.hidden_size
        self._is_mamba = True
        logger.info(f"  ✓ Mamba-2 loaded (hidden_size={dim})")
        return model, tok, dim

    def _load_bert(self, checkpoint: str):
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        logger.info(f"Loading BERT fallback encoder: {checkpoint}")
        tok   = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint)
        dim   = model.config.hidden_size
        logger.info(f"  ✓ BERT fallback loaded (hidden_size={dim})")
        return model, tok, dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids      : (B, T)
        attention_mask : (B, T)

        Returns
        -------
        embeddings : (B, hidden_size)  L2-normalised
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask if not self._is_mamba else None,
        )

        # Hidden states: (B, T, D)
        hidden = out.last_hidden_state

        # Mean-pool over valid tokens (respect attention_mask for BERT)
        mask = attention_mask.unsqueeze(-1).float()      # (B, T, 1)
        emb  = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, D)

        emb = self.proj(emb)
        return F.normalize(emb, p=2, dim=-1)

    # ── Encode text ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """
        Encode a list of strings into L2-normalised embeddings.

        Returns
        -------
        torch.Tensor of shape (N, hidden_size)
        """
        self.eval()
        all_embs = []

        max_len = min(self.max_seq_len, 512 if not self._is_mamba else self.max_seq_len)

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = self.tokenizer(
                batch,
                max_length=max_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            emb = self(input_ids, attention_mask)
            all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0)
