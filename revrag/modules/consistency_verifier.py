"""
Cross-Page Consistency Verification Module — the core contribution of RevRAG.

Given the top-k retrieved passages from a single long document, this module:
    1. Builds a pairwise comparison matrix over passages (NLI cross-encoder)
    2. Detects contradiction/supersession relationships
    3. Reranks passages so that the most authoritative (up-to-date) evidence
       ranks highest, demoting superseded passages

This module is designed to be a clean, separable component so that ablation
(a) [Mamba retriever alone] and (b) [verifier bolted onto BGE] are trivial
to run — just swap the upstream retriever.

Key design choices and tradeoffs:
    - NLI cross-encoder (DeBERTa-v3): lightweight, fast, works on short spans.
      Alternative considered: full LLM-as-judge (GPT-4). Rejected for latency
      and reproducibility reasons in a research pipeline; noted as future work.
    - Document order rule: when NLI detects contradiction, the passage from
      the later section is given higher authority. This is a domain heuristic
      (contracts, filings) — noted honestly as a limitation for documents
      where order does not imply authority (e.g., scientific papers with
      appendices that refine rather than supersede body text).
    - Threshold: contradiction confidence must exceed `nli_threshold` (default
      0.5) to trigger reranking; pairs below threshold are left as-is.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# NLI label indices for cross-encoder/nli-deberta-v3-small
# (entailment=0, neutral=1, contradiction=2 — model-dependent, verify at load)
NLI_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


@dataclass
class PassageInfo:
    """Metadata for a retrieved passage."""
    doc_id: str
    passage_id: str          # unique within the retrieval run
    text: str
    relevance_score: float   # raw retrieval score from bi-encoder
    section_idx: int         # approximate position in document (0-indexed)
    page_approx: int         # approximate page number (1-indexed)
    total_sections: int = 0


@dataclass
class ConsistencyScore:
    """Result of pairwise consistency scoring between two passages."""
    passage_a: str
    passage_b: str
    contradiction_prob: float
    entailment_prob: float
    neutral_prob: float
    is_contradiction: bool
    later_passage: str  # which of a/b comes later in the document


@dataclass
class RerankedResult:
    """Final reranked passage with combined score."""
    passage_info: PassageInfo
    final_score: float
    consistency_adjusted: bool  # True if this passage was moved by verifier
    superseded_by: Optional[str] = None  # passage_id that supersedes this one


class ConsistencyVerifier:
    """
    Cross-page consistency verification and reranking module.

    Typical usage within the RevRAG pipeline:
        verifier = ConsistencyVerifier(...)
        reranked = verifier.rerank(passages, query)
    """

    def __init__(
        self,
        checkpoint: str = "cross-encoder/nli-deberta-v3-small",
        nli_contradiction_threshold: float = 0.5,
        page_order_weight: float = 0.3,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_pair_length: int = 512,
    ):
        """
        Args:
            checkpoint:                  HuggingFace NLI model
            nli_contradiction_threshold: min contradiction prob to trigger reranking
            page_order_weight:           weight for document-order boost (0 = pure NLI)
            batch_size:                  pairs scored per NLI forward pass
            max_pair_length:             max tokens for NLI input (passage_a + passage_b)
        """
        self.threshold = nli_contradiction_threshold
        self.order_weight = page_order_weight
        self.batch_size = batch_size
        self.max_pair_length = max_pair_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading NLI model {checkpoint} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint
            ).to(self.device)
            self.nli_model.eval()
            # Determine label order from model config
            id2label = self.nli_model.config.id2label
            self._contra_idx = next(
                k for k, v in id2label.items() if "contra" in v.lower()
            )
            self._entail_idx = next(
                k for k, v in id2label.items() if "entail" in v.lower()
            )
            self._neutral_idx = next(
                k for k, v in id2label.items()
                if "neutral" in v.lower() or "not" in v.lower()
            )
            self._nli_available = True
        except Exception as e:
            logger.warning(
                f"Could not load NLI model ({e}). "
                "ConsistencyVerifier will use a stub (random scores) for smoke tests."
            )
            self._nli_available = False

    def _score_pairs_nli(
        self, pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """
        Score a list of (text_a, text_b) pairs with the NLI cross-encoder.

        Returns:
            (N, 3) array of softmax probabilities [entail, neutral, contra]
        """
        if not self._nli_available:
            # Stub: return random scores for testing
            n = len(pairs)
            probs = np.random.dirichlet(np.ones(3), size=n)
            return probs

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                premises = [p[0] for p in batch]
                hypotheses = [p[1] for p in batch]
                enc = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_pair_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.nli_model(**enc).logits  # (B, num_labels)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # Re-order to [entail, neutral, contra]
                reordered = np.stack([
                    probs[:, self._entail_idx],
                    probs[:, self._neutral_idx],
                    probs[:, self._contra_idx],
                ], axis=1)
                all_probs.append(reordered)

        return np.vstack(all_probs)

    def score_all_pairs(
        self, passages: List[PassageInfo]
    ) -> List[ConsistencyScore]:
        """
        Score all N*(N-1)/2 passage pairs for consistency.

        For efficiency, only score pairs from *different pages* (same page
        pairs are trivially consistent — they were retrieved together).
        """
        n = len(passages)
        if n < 2:
            return []

        # Build pairs: only cross-page pairs
        pair_indices = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if passages[i].page_approx != passages[j].page_approx
        ]
        if not pair_indices:
            # All from same page — no cross-page check needed
            return []

        text_pairs = [
            (passages[i].text[:256], passages[j].text[:256])
            for i, j in pair_indices
        ]

        probs = self._score_pairs_nli(text_pairs)

        scores = []
        for (i, j), prob in zip(pair_indices, probs):
            a, b = passages[i], passages[j]
            later = b.passage_id if b.section_idx > a.section_idx else a.passage_id
            scores.append(ConsistencyScore(
                passage_a=a.passage_id,
                passage_b=b.passage_id,
                entailment_prob=float(prob[0]),
                neutral_prob=float(prob[1]),
                contradiction_prob=float(prob[2]),
                is_contradiction=float(prob[2]) >= self.threshold,
                later_passage=later,
            ))
        return scores

    def rerank(
        self,
        passages: List[PassageInfo],
        query: str,
        return_scores: bool = False,
    ) -> List[RerankedResult]:
        """
        Main entry point: rerank passages using cross-page consistency.

        Algorithm:
            1. Score all cross-page pairs with NLI
            2. For each contradiction pair, demote the earlier passage and
               boost the later one (later = more authoritative in contracts)
            3. Final score = λ * relevance_score + (1-λ) * consistency_boost
               where λ = 1 - page_order_weight

        Args:
            passages:      list of PassageInfo from bi-encoder retrieval
            query:         original query string (reserved for future use)
            return_scores: if True, include ConsistencyScore objects in output

        Returns:
            List[RerankedResult] sorted descending by final_score
        """
        start_t = time.time()

        if len(passages) <= 1:
            return [
                RerankedResult(
                    passage_info=p,
                    final_score=p.relevance_score,
                    consistency_adjusted=False,
                )
                for p in passages
            ]

        # Step 1: Normalize relevance scores to [0, 1]
        rel_scores = np.array([p.relevance_score for p in passages])
        rel_min, rel_max = rel_scores.min(), rel_scores.max()
        if rel_max > rel_min:
            norm_rel = (rel_scores - rel_min) / (rel_max - rel_min)
        else:
            norm_rel = np.ones_like(rel_scores)

        # Step 2: Score cross-page pairs
        consistency_scores = self.score_all_pairs(passages)

        # Step 3: Build consistency adjustment vector
        # adjustment[i] > 0 means passage[i] is boosted (later/authoritative)
        # adjustment[i] < 0 means passage[i] is demoted (superseded)
        id_to_idx = {p.passage_id: i for i, p in enumerate(passages)}
        adjustments = np.zeros(len(passages))
        superseded_by: Dict[str, str] = {}

        for cs in consistency_scores:
            if not cs.is_contradiction:
                continue
            # Boost later passage, demote earlier one
            later_idx = id_to_idx[cs.later_passage]
            earlier_id = (
                cs.passage_a if cs.later_passage == cs.passage_b else cs.passage_b
            )
            earlier_idx = id_to_idx[earlier_id]

            boost = cs.contradiction_prob  # stronger contradiction → larger adjustment
            adjustments[later_idx] += boost
            adjustments[earlier_idx] -= boost
            superseded_by[earlier_id] = cs.later_passage

        # Step 4: Normalize adjustments to [−1, 1]
        adj_abs_max = np.abs(adjustments).max()
        if adj_abs_max > 0:
            norm_adj = adjustments / adj_abs_max
        else:
            norm_adj = adjustments

        # Step 5: Combine scores
        lam = 1.0 - self.order_weight
        final_scores = lam * norm_rel + self.order_weight * norm_adj

        # Step 6: Build results
        results = []
        for i, passage in enumerate(passages):
            results.append(RerankedResult(
                passage_info=passage,
                final_score=float(final_scores[i]),
                consistency_adjusted=bool(norm_adj[i] != 0),
                superseded_by=superseded_by.get(passage.passage_id),
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)

        elapsed = (time.time() - start_t) * 1000
        logger.debug(
            f"Consistency verification: {len(passages)} passages, "
            f"{len(consistency_scores)} pairs scored, {elapsed:.1f}ms"
        )
        return results

    def compute_evidence_consistency_score(
        self, passages: List[PassageInfo]
    ) -> float:
        """
        Evidence Consistency Score (ECS) — one of our proposed metrics.

        Definition:
            ECS = 1 - (number of contradicting cross-page pairs /
                       total cross-page pairs)

        A score of 1.0 means all retrieved passages are mutually consistent.
        A score of 0.0 means every cross-page pair contradicts each other.

        This is computed BEFORE reranking to measure the quality of the raw
        retrieved set, and AFTER reranking to show improvement.
        """
        cross_page_scores = self.score_all_pairs(passages)
        if not cross_page_scores:
            return 1.0  # no cross-page pairs → trivially consistent
        n_contra = sum(1 for s in cross_page_scores if s.is_contradiction)
        return 1.0 - (n_contra / len(cross_page_scores))
