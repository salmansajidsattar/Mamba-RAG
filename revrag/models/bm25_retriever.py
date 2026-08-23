"""BM25 sparse retrieval baseline using rank_bm25."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    """
    Standard BM25 baseline.
    Expected by reviewers as the sparse retrieval lower bound.
    """

    def __init__(self):
        self.bm25 = None
        self.doc_ids: List[str] = []

    def index(self, documents: List[Tuple[str, str]]) -> None:
        """
        Build BM25 index.

        Args:
            documents: [(doc_id, doc_text), ...]
        """
        logger.info(f"Building BM25 index over {len(documents)} documents...")
        self.doc_ids = [d[0] for d in documents]
        tokenized = [_tokenize(d[1]) for d in tqdm(documents, desc="Tokenizing")]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a single query.

        Returns:
            [(doc_id, score), ...] sorted descending by score
        """
        if self.bm25 is None:
            raise RuntimeError("Call index() before retrieve().")
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    def batch_retrieve(
        self, queries: List[Tuple[str, str]], top_k: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve for a batch of queries.

        Args:
            queries: [(query_id, query_text), ...]

        Returns:
            {query_id: [(doc_id, score), ...], ...}
        """
        results = {}
        for qid, qtext in tqdm(queries, desc="BM25 retrieval"):
            results[qid] = self.retrieve(qtext, top_k=top_k)
        return results
