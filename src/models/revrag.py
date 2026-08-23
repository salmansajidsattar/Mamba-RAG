"""
src/models/revrag.py
====================
RevRAG: full retrieval pipeline.

  query → BiEncoder → FAISS top-k → CrossPageVerifier → final top-k

This is the complete model as presented in the paper.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.models.bi_encoder import BiEncoder
from src.models.verifier   import CrossPageVerifier

logger = logging.getLogger(__name__)


class RevRAG:
    """
    End-to-end RevRAG retrieval system.

    Parameters
    ----------
    checkpoint            : Mamba-2 checkpoint (or CPU fallback)
    hidden_size           : encoder embedding dim
    max_seq_len           : max tokens
    top_k_retrieval       : candidates from FAISS before verification
    top_k_final           : passages returned after reranking
    page_order_weight     : λ
    contradiction_threshold : τ_c
    use_verifier          : set False for ablation (bi-encoder only baseline)
    share_weights         : shared query/doc encoder
    device                : "cuda" or "cpu"
    """

    def __init__(
        self,
        checkpoint:              str   = "state-spaces/mamba2-130m",
        hidden_size:             int   = 768,
        max_seq_len:             int   = 4096,
        top_k_retrieval:         int   = 20,
        top_k_final:             int   = 5,
        page_order_weight:       float = 0.3,
        contradiction_threshold: float = 0.5,
        use_verifier:            bool  = True,
        share_weights:           bool  = True,
        device:                  str   = "cuda",
        verifier_checkpoint:     str   = "cross-encoder/nli-deberta-v3-small",
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final     = top_k_final
        self.use_verifier    = use_verifier
        self.device          = device

        # ── Bi-encoder ────────────────────────────────────────────────────────
        self.bi_encoder = BiEncoder(
            checkpoint=checkpoint,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            share_weights=share_weights,
            device=device,
        )

        # ── Verifier ──────────────────────────────────────────────────────────
        if use_verifier:
            self.verifier = CrossPageVerifier(
                checkpoint=verifier_checkpoint,
                page_order_weight=page_order_weight,
                contradiction_threshold=contradiction_threshold,
                device=device,
            )
        else:
            self.verifier = None

    # ── Index building ────────────────────────────────────────────────────────

    def index(self, doc_texts: List[str], doc_ids: List[str], batch_size: int = 32):
        """Build FAISS index over document corpus."""
        self.bi_encoder.build_index(doc_texts, doc_ids, batch_size)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        queries:          List[str],
        top_k:            Optional[int]   = None,
        lambda_override:  Optional[float] = None,
        tau_override:     Optional[float] = None,
    ) -> List[List[Dict]]:
        """
        Retrieve and (optionally) rerank passages for each query.

        Returns
        -------
        List[List[Dict]]  — one list per query, each dict has:
          doc_id, text, score (retrieval cosine sim)
          + if verifier: final_score, page_order_boost, nli_contradiction_score
        """
        final_k = top_k or self.top_k_final
        cand_k  = max(self.top_k_retrieval, final_k * 4)

        # Step 1: FAISS retrieval
        candidates = self.bi_encoder.retrieve(queries, top_k=cand_k)

        # Step 2: Cross-page verification + reranking
        if self.verifier is None:
            return [c[:final_k] for c in candidates]

        results = []
        for cands in candidates:
            reranked = self.verifier.rerank(
                cands,
                top_k=final_k,
                lambda_override=lambda_override,
                tau_override=tau_override,
            )
            results.append(reranked)

        return results

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        self.bi_encoder.save(Path(path))

    def load(self, path: Path):
        self.bi_encoder.load(Path(path))
