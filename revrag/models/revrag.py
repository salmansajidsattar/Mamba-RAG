"""
RevRAG: Revision-Aware Retrieval-Augmented Generation.

Proposed method: Mamba-2 bi-encoder + cross-page consistency verifier.

The full pipeline:
    1. Mamba-2 encoder reads the full document (no chunking) → FAISS retrieval
    2. Top-k candidates passed to ConsistencyVerifier
    3. Verifier scores cross-page pairs → reranks by authority
    4. Top-final_k passages returned as generator context
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.consistency_verifier import ConsistencyVerifier, PassageInfo, RerankedResult
from models.mamba_retriever import MambaRetriever

logger = logging.getLogger(__name__)


class RevRAG:
    """
    Full RevRAG pipeline combining MambaRetriever and ConsistencyVerifier.

    The verifier is a separable component — set use_verifier=False to get
    the Mamba-only ablation (Ablation a in the paper).
    """

    def __init__(
        self,
        mamba_checkpoint: str = "state-spaces/mamba2-130m",
        nli_checkpoint: str = "cross-encoder/nli-deberta-v3-small",
        hidden_size: int = 768,
        max_doc_length: int = 8192,
        top_k: int = 20,
        final_k: int = 5,
        nli_threshold: float = 0.5,
        page_order_weight: float = 0.3,
        batch_size: int = 8,
        device: Optional[str] = None,
        use_verifier: bool = True,
    ):
        self.top_k = top_k
        self.final_k = final_k
        self.use_verifier = use_verifier

        self.retriever = MambaRetriever(
            checkpoint=mamba_checkpoint,
            hidden_size=hidden_size,
            max_length=max_doc_length,
            batch_size=batch_size,
            device=device,
        )

        if use_verifier:
            self.verifier = ConsistencyVerifier(
                checkpoint=nli_checkpoint,
                nli_contradiction_threshold=nli_threshold,
                page_order_weight=page_order_weight,
                device=device,
            )
        else:
            self.verifier = None

        # Store document texts for passage extraction after retrieval
        self._doc_store: Dict[str, str] = {}

    def index_documents(
        self, documents: List[Tuple[str, str]]
    ) -> None:
        """
        Index documents for retrieval.
        Also stores full text for passage extraction.
        """
        for doc_id, text in documents:
            self._doc_store[doc_id] = text
        self.retriever.index_documents(documents)

    def _extract_passages(
        self,
        doc_id: str,
        doc_text: str,
        passage_size: int = 512,
        overlap: int = 64,
    ) -> List[PassageInfo]:
        """
        Extract overlapping passages from a document for the verifier.
        Each passage is annotated with its approximate page/section position.

        Unlike the BGE baseline (which uses these chunks as the retrieval unit),
        here passages are used ONLY for the verifier — the retrieval is done
        at the document level by Mamba.
        """
        words = doc_text.split()
        passages = []
        total_words = len(words)

        for i, start in enumerate(range(0, total_words, passage_size - overlap)):
            end = min(start + passage_size, total_words)
            text = " ".join(words[start:end])
            # Approximate page (1 page ≈ 500 words)
            page_approx = (start // 500) + 1
            section_idx = i
            passages.append(PassageInfo(
                doc_id=doc_id,
                passage_id=f"{doc_id}_p{i}",
                text=text,
                relevance_score=0.0,  # will be filled after retrieval scoring
                section_idx=section_idx,
                page_approx=page_approx,
                total_sections=len(list(range(0, total_words, passage_size - overlap))),
            ))
        return passages

    def retrieve(
        self,
        query_id: str,
        query: str,
        tracer=None,
    ) -> List[RerankedResult]:
        """
        Full RevRAG retrieval for a single query.

        Args:
            query_id: for logging/tracing
            query:    query string
            tracer:   optional Langfuse tracer (see tracing/langfuse_tracer.py)

        Returns:
            List[RerankedResult] of length <= final_k, sorted by final_score
        """
        t0 = time.time()
        span = tracer.start_span(f"revrag.retrieve.{query_id}") if tracer else None

        # Step 1: Bi-encoder retrieval
        raw_hits = self.retriever.retrieve(query, top_k=self.top_k)
        if not raw_hits:
            return []

        t1 = time.time()

        if self.verifier is None or not self.use_verifier:
            # Ablation (a): Mamba retriever alone
            results = [
                RerankedResult(
                    passage_info=PassageInfo(
                        doc_id=did,
                        passage_id=f"{did}_raw",
                        text=self._doc_store.get(did, "")[:512],
                        relevance_score=score,
                        section_idx=0,
                        page_approx=1,
                    ),
                    final_score=score,
                    consistency_adjusted=False,
                )
                for did, score in raw_hits[:self.final_k]
            ]
            if span:
                span.end(output={"retrieval_ms": (t1 - t0) * 1000})
            return results

        # Step 2: Extract passages from top-k docs for verifier
        all_passages: List[PassageInfo] = []
        for doc_id, ret_score in raw_hits:
            doc_text = self._doc_store.get(doc_id, "")
            if not doc_text:
                continue
            passages = self._extract_passages(doc_id, doc_text)
            # Score passages by BM25-like term overlap with query
            # (lightweight proxy; full cross-encoder scoring is left for
            # future work to keep latency under 50ms)
            q_terms = set(query.lower().split())
            for p in passages:
                p_terms = set(p.text.lower().split())
                overlap_score = len(q_terms & p_terms) / max(len(q_terms), 1)
                # Combine doc-level retrieval score with passage relevance
                p.relevance_score = ret_score * 0.7 + overlap_score * 0.3
            all_passages.extend(passages)

        # Rank passages by relevance, keep top-k for verification
        all_passages.sort(key=lambda p: p.relevance_score, reverse=True)
        candidate_passages = all_passages[:self.top_k]

        t2 = time.time()

        # Step 3: Consistency verification and reranking
        reranked = self.verifier.rerank(candidate_passages, query)

        t3 = time.time()

        if span:
            span.end(output={
                "retrieval_ms": (t1 - t0) * 1000,
                "passage_extract_ms": (t2 - t1) * 1000,
                "verification_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - t0) * 1000,
                "n_passages_checked": len(candidate_passages),
            })

        return reranked[:self.final_k]

    def batch_retrieve(
        self,
        queries: List[Tuple[str, str]],
        tracer=None,
    ) -> Dict[str, List[RerankedResult]]:
        """Retrieve for a batch of (query_id, query_text) pairs."""
        return {
            qid: self.retrieve(qid, qtext, tracer=tracer)
            for qid, qtext in queries
        }
