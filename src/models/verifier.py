"""
src/models/verifier.py
======================
Cross-page consistency verifier using NLI.

Model: cross-encoder/nli-deberta-v3-small
  - DeBERTa-v3 is SOTA on NLI benchmarks (MNLI, SNLI)
  - NOT trained on ContractNLI → no data leakage in evaluation
  - "small" variant: fast enough for inference, fits T4 comfortably

Algorithm (per query):
  1. Score all passage pairs (p_i, p_j) with NLI cross-encoder
  2. If contradiction(p_i, p_j) > τ_c → apply page-order authority rule:
       later page number → higher authority → boost its score
  3. Final passage score = λ × page_order_boost + (1-λ) × retrieval_score

Key metrics this enables:
  CRA@1  : did we rank the authoritative passage above the contradicting one?
  ECS    : Evidence Consistency Score — fraction of non-contradicting pairs
  DPR↓   : Decoy Promotion Rate — did the superseded passage slip into top-1?

Hyperparameters (ablated in paper):
  λ  (page_order_weight)         ∈ {0.0, 0.1, ..., 1.0}
  τ_c (nli_contradiction_threshold) ∈ {0.30, 0.35, ..., 0.70}
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

NLI_LABELS   = ["contradiction", "neutral", "entailment"]
CONTRA_IDX   = 0   # position of 'contradiction' in DeBERTa NLI output


class CrossPageVerifier:
    """
    NLI-based cross-page consistency verifier.

    Parameters
    ----------
    checkpoint            : NLI cross-encoder checkpoint
    page_order_weight     : λ — weight of page-order score in final ranking
    contradiction_threshold : τ_c — NLI contradiction score to flag a pair
    device                : "cuda" or "cpu"
    batch_size            : passage pairs per NLI forward pass
    """

    def __init__(
        self,
        checkpoint:              str   = "cross-encoder/nli-deberta-v3-small",
        page_order_weight:       float = 0.3,
        contradiction_threshold: float = 0.5,
        device:                  str   = "cuda",
        batch_size:              int   = 32,
    ):
        self.checkpoint              = checkpoint
        self.page_order_weight       = page_order_weight
        self.contradiction_threshold = contradiction_threshold
        self.device                  = device
        self.batch_size              = batch_size

        self._pipeline = None   # lazy-loaded

    def _load(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline  # type: ignore
        logger.info(f"Loading NLI verifier: {self.checkpoint}")
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.checkpoint,
            device=0 if self.device == "cuda" else -1,
        )
        logger.info("  ✓ Verifier loaded.")

    # ── Core NLI scoring ──────────────────────────────────────────────────────

    def _nli_score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Returns dict with keys: contradiction, neutral, entailment.
        """
        self._load()
        out = self._pipeline(
            premise,
            candidate_labels=NLI_LABELS,
            hypothesis_template="{}",
        )
        return dict(zip(out["labels"], out["scores"]))

    def score_pair(self, text_a: str, text_b: str) -> float:
        """Contradiction score for a passage pair. Range [0, 1]."""
        scores = self._nli_score(text_a, text_b)
        return scores.get("contradiction", 0.0)

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        passages:         List[Dict],   # each: {text, score, page_num (optional)}
        top_k:            int = 5,
        lambda_override:  Optional[float] = None,
        tau_override:     Optional[float] = None,
    ) -> List[Dict]:
        """
        Rerank passages using cross-page consistency.

        Parameters
        ----------
        passages : list of dicts with keys:
            text      : passage text
            score     : retrieval score (cosine similarity)
            page_num  : (optional) page/section index; inferred from order if absent
        top_k : number of passages to return

        Returns
        -------
        Reranked list of passage dicts with added keys:
            nli_contradiction_score : max contradiction score with any other passage
            page_order_boost        : 1.0 if this passage wins authority, else 0.0
            final_score             : λ × page_order_boost + (1-λ) × retrieval_score
        """
        if not passages:
            return []

        λ   = lambda_override   if lambda_override   is not None else self.page_order_weight
        τ_c = tau_override      if tau_override       is not None else self.contradiction_threshold

        n = len(passages)

        # Assign page numbers (position in retrieval list if not provided)
        for i, p in enumerate(passages):
            if "page_num" not in p:
                p["page_num"] = i

        # Score all passage pairs for contradiction
        contra_scores = [[0.0] * n for _ in range(n)]
        pairs = list(combinations(range(n), 2))

        # Batch NLI calls
        pair_texts = [
            (passages[i]["text"][:512], passages[j]["text"][:512])
            for i, j in pairs
        ]
        for k, (i, j) in enumerate(pairs):
            sc = self.score_pair(pair_texts[k][0], pair_texts[k][1])
            contra_scores[i][j] = sc
            contra_scores[j][i] = sc

        # For each passage: max contradiction with any other passage
        for i, p in enumerate(passages):
            p["nli_contradiction_score"] = max(contra_scores[i]) if n > 1 else 0.0

        # Apply page-order authority rule:
        # If two passages contradict (score > τ_c), the LATER one wins
        page_order_boost = [0.0] * n
        for i, j in pairs:
            if contra_scores[i][j] < τ_c:
                continue
            # Later page = higher page_num = winner
            winner = i if passages[i]["page_num"] > passages[j]["page_num"] else j
            page_order_boost[winner] = max(page_order_boost[winner], contra_scores[i][j])

        # Final score = λ × page_order_boost + (1-λ) × retrieval_score
        for i, p in enumerate(passages):
            p["page_order_boost"] = page_order_boost[i]
            p["final_score"]      = (
                λ * page_order_boost[i] +
                (1 - λ) * p["score"]
            )

        return sorted(passages, key=lambda x: x["final_score"], reverse=True)[:top_k]

    # ── Metrics ───────────────────────────────────────────────────────────────

    def evidence_consistency_score(self, passages: List[Dict]) -> float:
        """
        ECS: fraction of passage pairs with NO contradiction (score < τ_c).
        Range [0, 1]. Higher = more consistent evidence set.
        """
        n = len(passages)
        if n < 2:
            return 1.0
        pairs    = list(combinations(range(n), 2))
        n_clean  = sum(
            1 for i, j in pairs
            if self.score_pair(
                passages[i]["text"][:512],
                passages[j]["text"][:512],
            ) < self.contradiction_threshold
        )
        return n_clean / len(pairs)
