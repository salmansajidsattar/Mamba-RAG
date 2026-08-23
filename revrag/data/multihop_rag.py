"""
MultiHop-RAG dataset loader.

MultiHop-RAG (Li et al., 2024, arXiv:2401.15391) is a peer-reviewed
retrieval benchmark built from real news articles and Wikipedia passages.
It requires multi-hop reasoning across multiple retrieved documents to
answer each query.

Why we include it:
    1. It is NOT legal-domain, so it tests whether RevRAG's encoder and
       verifier generalise beyond contracts.
    2. It is a real, peer-reviewed benchmark — not synthetic.
    3. Multi-hop queries stress-test long-context retrieval: answers require
       chaining information from 2–3 passages, which chunked retrievers
       routinely miss.

What we do NOT claim:
    MultiHop-RAG was not designed to test intra-document revision.
    We use it purely to evaluate retrieval quality (Recall@k, NDCG@10, MRR)
    and to show the Mamba-2 encoder is competitive on a general-domain
    multi-hop benchmark, not just legal contracts.

HuggingFace dataset: yixuantt/MultiHopRAG
    pip install datasets

Dataset structure:
    query: str
    answer: str
    question_type: str   (e.g. "inference_query", "comparison_query", ...)
    supporting_passages: [{"id": str, "text": str, "title": str}, ...]
    corpus: the full passage pool is accessed separately via the 'corpus' split

Note on corpus loading:
    The full corpus (~600K passages) is large. For Kaggle we subsample to
    the passages that appear in either supporting or distracting sets of
    the evaluated queries, plus a random sample of negatives.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HF_DATASET_ID = "yixuantt/MultiHopRAG"
_CORPUS_SPLIT = "MultiHopRAG"      # the full corpus split name
_QUERY_SPLIT = "MultiHopRAG"       # queries are in the same split, filtered by type


class MultiHopRAGDataset:
    """
    MultiHop-RAG retrieval benchmark.

    Parameters
    ----------
    split : str
        "train" or "test" (MultiHop-RAG uses a single split; use "test").
    max_queries : int or None
        Cap on number of queries to load. Full set has ~2,500 test queries.
        Use 500 for Kaggle runs.
    neg_ratio : int
        Number of random negative passages to add per query's supporting
        passages, to build a realistic corpus. Default 50.
    question_types : list of str or None
        Filter by question type. Options include:
            "inference_query", "comparison_query", "null_query",
            "temporal_query", "bridge_comparison_query"
        Default: all types.
    seed : int
    cache_dir : str or None

    Attributes
    ----------
    queries : list of dict
        {query_id, query_text, answer, question_type}
    documents : dict
        {doc_id: passage_text}  — supporting + negative passages
    doc_titles : dict
        {doc_id: title}
    gold_passage_ids : dict
        {query_id: [passage_id, ...]}
    qrels : dict
        {query_id: {passage_id: 1 for gold, 0 for negatives (omitted)}}
    """

    def __init__(
        self,
        split: str = "test",
        max_queries: Optional[int] = 500,
        neg_ratio: int = 50,
        question_types: Optional[List[str]] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.max_queries = max_queries
        self.neg_ratio = neg_ratio
        self.question_types = question_types
        self.seed = seed
        self.cache_dir = cache_dir

        self.queries: List[Dict] = []
        self.documents: Dict[str, str] = {}
        self.doc_titles: Dict[str, str] = {}
        self.gold_passage_ids: Dict[str, List[str]] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}

        self._load()

    def _load(self) -> None:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("pip install datasets")

        logger.info(
            f"Loading MultiHop-RAG from HuggingFace [{_HF_DATASET_ID}]…"
        )
        rng = random.Random(self.seed)

        # Load the dataset
        ds = load_dataset(
            _HF_DATASET_ID,
            split=_QUERY_SPLIT,
            cache_dir=self.cache_dir,
        )
        logger.info(f"MultiHop-RAG raw examples: {len(ds)}")

        # Filter by question type if specified
        if self.question_types:
            ds = ds.filter(
                lambda ex: ex.get("question_type", "") in self.question_types
            )
            logger.info(f"After type filter: {len(ds)} examples")

        # Subsample queries
        indices = list(range(len(ds)))
        if self.max_queries and len(indices) > self.max_queries:
            indices = rng.sample(indices, self.max_queries)
        selected = [ds[i] for i in sorted(indices)]

        # Collect all supporting passages to build corpus
        all_supporting: Dict[str, str] = {}  # passage_id → text
        all_titles: Dict[str, str] = {}

        for q_idx, ex in enumerate(selected):
            query_id = str(ex.get("id", f"mhrag_{q_idx}"))
            query_text = ex.get("query", "").strip()
            if not query_text:
                continue

            self.queries.append({
                "query_id": query_id,
                "query_text": query_text,
                "answer": ex.get("answer", ""),
                "question_type": ex.get("question_type", "unknown"),
            })

            # Supporting passages = gold passages for this query
            supporting = ex.get("supporting_passages", []) or []
            gold_ids = []
            for sp in supporting:
                pid = sp.get("id", f"p_{len(all_supporting)}")
                text = sp.get("text", "").strip()
                title = sp.get("title", "")
                if text and pid not in all_supporting:
                    all_supporting[pid] = text
                    all_titles[pid] = title
                gold_ids.append(pid)

            self.gold_passage_ids[query_id] = gold_ids
            self.qrels[query_id] = {pid: 1 for pid in gold_ids}

        # Collect distracting passages from the dataset
        # (passages referenced in other queries act as hard negatives)
        distractor_pool: List[str] = []
        for ex in ds:
            for sp in (ex.get("supporting_passages", []) or []):
                pid = sp.get("id", "")
                text = sp.get("text", "").strip()
                title = sp.get("title", "")
                if pid and pid not in all_supporting and text:
                    all_supporting[pid] = text
                    all_titles[pid] = title
                    distractor_pool.append(pid)

        # Add a random sample of negatives to the corpus to make retrieval
        # non-trivial (otherwise every query has only its gold docs in corpus)
        n_neg = min(len(distractor_pool), len(self.queries) * self.neg_ratio)
        neg_ids = rng.sample(distractor_pool, n_neg)

        # Build final document store: gold + negatives
        for pid in list(all_supporting.keys()):
            is_any_gold = any(pid in gids for gids in self.gold_passage_ids.values())
            is_sampled_neg = pid in neg_ids
            if is_any_gold or is_sampled_neg:
                self.documents[pid] = all_supporting[pid]
                self.doc_titles[pid] = all_titles.get(pid, "")

        type_counts: Dict[str, int] = defaultdict(int)
        for q in self.queries:
            type_counts[q["question_type"]] += 1

        logger.info(
            f"MultiHop-RAG loaded: {len(self.queries)} queries, "
            f"{len(self.documents)} passages in corpus.\n"
            f"  Question types: {dict(type_counts)}"
        )

    # ------------------------------------------------------------------
    # Standard retrieval evaluation helpers
    # ------------------------------------------------------------------

    def recall_at_k(self, query_id: str, retrieved_ids: List[str], k: int) -> float:
        """Fraction of gold passage ids recovered in top-k retrieved ids."""
        gold = set(self.gold_passage_ids.get(query_id, []))
        if not gold:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        return len(gold & retrieved_k) / len(gold)

    def get_qrel_dict(self) -> Dict[str, Dict[str, int]]:
        """pytrec_eval format: {query_id: {passage_id: relevance}}"""
        return dict(self.qrels)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict:
        return self.queries[idx]

    def summary(self) -> str:
        return (
            f"MultiHop-RAG: {len(self.queries)} queries, "
            f"{len(self.documents)} corpus passages"
        )
