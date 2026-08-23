"""
BGE-M3 dense retriever baseline.

BAAI/bge-m3 handles sequences up to 8,192 tokens natively via its hybrid
(dense + sparse + ColBERT) architecture.  We use *dense-only* mode so the
comparison with Mamba-2 and bge-large-en-v1.5 is apples-to-apples: a single
embedding per passage, cosine similarity, FAISS flat IP index.

Key difference from BGERetriever (bge-large-en-v1.5):
  - bge-large-en-v1.5: max 512 tokens → must chunk long documents
  - bge-m3:            max 8192 tokens → encodes entire document in one shot

This makes BGE-M3 the strongest fair baseline for our claim that the
benefit of Mamba-2 is *not* merely avoiding chunking: if BGE-M3 already
matches Mamba-2 on retrieval quality, the SSM advantage is nullified.

Usage:
    retriever = BGEM3Retriever(device="cuda")
    retriever.index(doc_texts, doc_ids)
    results = retriever.batch_retrieve(queries, top_k=20)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT = "BAAI/bge-m3"
_MAX_LENGTH = 8192          # BGE-M3 supported maximum
_DENSE_EMBED_DIM = 1024     # BGE-M3 dense output dimension
_BATCH_SIZE = 4             # Small batch: sequences can be very long


@dataclass
class BGEM3RetrievalResult:
    doc_id: str
    score: float
    text: str
    encode_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# BGEM3Retriever
# ---------------------------------------------------------------------------
class BGEM3Retriever:
    """
    Dense retriever using BAAI/bge-m3 (dense-only mode).

    Parameters
    ----------
    checkpoint : str
        HuggingFace model ID.  Default: BAAI/bge-m3
    max_length : int
        Truncation length in tokens.  BGE-M3 supports up to 8192.
    batch_size : int
        Encoding batch size.  Keep small (2–8) for 8K-token sequences.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        max_length: int = _MAX_LENGTH,
        batch_size: int = _BATCH_SIZE,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        self._model = None          # loaded lazily
        self._tokenizer = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
            # FlagEmbedding provides the official BGE-M3 API
            self._flag_model = BGEM3FlagModel(
                self.checkpoint,
                use_fp16=(self.device == "cuda"),
            )
            self._use_flag = True
            logger.info("Loaded BGE-M3 via FlagEmbedding.")
        except ImportError:
            # Fallback: plain HuggingFace AutoModel (dense only)
            logger.warning(
                "FlagEmbedding not installed; falling back to HuggingFace "
                "AutoModel for BGE-M3 dense mode.  "
                "Install with: pip install FlagEmbedding"
            )
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self._model = AutoModel.from_pretrained(self.checkpoint).to(self.device)
            self._model.eval()
            self._use_flag = False

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised dense embeddings, shape (N, 1024)."""
        self._load_model()

        if self._use_flag:
            # FlagEmbedding handles batching internally
            out = self._flag_model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            embeddings = out["dense_vecs"]          # (N, 1024) numpy array
        else:
            embeddings_list = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self._model(**enc)
                # CLS pooling (standard for BGE)
                emb = out.last_hidden_state[:, 0, :]  # (B, 1024)
                # L2 normalise
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                embeddings_list.append(emb.cpu().numpy())
            embeddings = np.concatenate(embeddings_list, axis=0)

        # Ensure float32 and contiguous for FAISS
        return np.ascontiguousarray(embeddings.astype(np.float32))

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    def index(self, doc_texts: List[str], doc_ids: List[str]) -> None:
        """
        Encode all documents and build a FAISS flat IP index.

        Parameters
        ----------
        doc_texts : list of str
            Full document texts (no chunking — BGE-M3 handles length natively).
        doc_ids : list of str
            Parallel list of document identifiers.
        """
        assert len(doc_texts) == len(doc_ids), "doc_texts and doc_ids must be same length"
        logger.info(f"Indexing {len(doc_texts)} documents with BGE-M3 (max_len={self.max_length})…")

        t0 = time.perf_counter()
        embeddings = self._encode_texts(doc_texts)
        encode_ms = (time.perf_counter() - t0) * 1000

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._doc_ids = list(doc_ids)
        self._doc_texts = list(doc_texts)

        logger.info(
            f"Index built: {self._index.ntotal} vectors, dim={dim}, "
            f"encode time={encode_ms:.0f} ms"
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self, query: str, top_k: int = 20
    ) -> List[BGEM3RetrievalResult]:
        """Retrieve top-k documents for a single query."""
        return self.batch_retrieve([query], top_k=top_k)[0]

    def batch_retrieve(
        self, queries: List[str], top_k: int = 20
    ) -> List[List[BGEM3RetrievalResult]]:
        """
        Retrieve top-k documents for each query in the batch.

        Returns
        -------
        list of list of BGEM3RetrievalResult
            Outer list: one entry per query.
            Inner list: top_k results, sorted by descending score.
        """
        if self._index is None:
            raise RuntimeError("Call index() before retrieve().")

        t0 = time.perf_counter()
        q_embs = self._encode_texts(queries)
        encode_ms = (time.perf_counter() - t0) * 1000

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_embs, k)  # (Q, k)

        all_results = []
        per_query_enc_ms = encode_ms / max(len(queries), 1)
        for q_idx in range(len(queries)):
            results = []
            for rank in range(k):
                doc_idx = int(indices[q_idx, rank])
                if doc_idx < 0:
                    continue
                results.append(
                    BGEM3RetrievalResult(
                        doc_id=self._doc_ids[doc_idx],
                        score=float(scores[q_idx, rank]),
                        text=self._doc_texts[doc_idx],
                        encode_time_ms=per_query_enc_ms,
                    )
                )
            all_results.append(results)
        return all_results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n = self._index.ntotal if self._index else 0
        return (
            f"BGEM3Retriever(checkpoint={self.checkpoint!r}, "
            f"max_length={self.max_length}, indexed={n})"
        )
