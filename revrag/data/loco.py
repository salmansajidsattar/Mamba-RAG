"""
LoCoV1 (Long Context) benchmark data loader.

LoCoV1 is a long-context retrieval benchmark introduced alongside M2-BERT
(Chang et al., arXiv:2402.07440). Documents up to 32K tokens, multi-hop queries.

HuggingFace — two separate datasets:
    hazyresearch/LoCoV1-Documents  — document corpus
    hazyresearch/LoCoV1-Queries    — queries + gold doc IDs

Local cache layout (after download_datasets_local.py):
    data/raw/loco_v1/documents/    ← LoCoV1-Documents cache
    data/raw/loco_v1/queries/      ← LoCoV1-Queries cache
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LoCoV1Dataset:
    """
    LoCoV1 framed as a retrieval benchmark.

    Parameters
    ----------
    cache_dir : str | Path | None
        Root cache dir (contains documents/ and queries/ subdirs).
        If None, HuggingFace downloads to default cache.
    max_queries : int | None
        Truncate queries for quick runs.

    Attributes
    ----------
    queries    : list[dict]  — {query_id, query, gold_doc_ids}
    documents  : dict        — {doc_id: text}
    qrels      : dict        — {query_id: {doc_id: 1}}
    """

    _HF_DOCS    = "hazyresearch/LoCoV1-Documents"
    _HF_QUERIES = "hazyresearch/LoCoV1-Queries"

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        max_queries: Optional[int] = None,
    ) -> None:
        self.cache_dir   = Path(cache_dir) if cache_dir else None
        self.max_queries = max_queries

        self.queries:   List[Dict]        = []
        self.documents: Dict[str, str]    = {}
        self.qrels:     Dict[str, Dict]   = {}
        self.gold_spans: Dict[str, List]  = {}

        self._load()

    def _load(self) -> None:
        from datasets import load_dataset  # type: ignore

        docs_cache    = str(self.cache_dir / "documents") if self.cache_dir else None
        queries_cache = str(self.cache_dir / "queries")   if self.cache_dir else None

        # ── Documents ─────────────────────────────────────────────────────
        logger.info("Loading LoCoV1-Documents …")
        docs_ds = load_dataset(self._HF_DOCS, cache_dir=docs_cache, split="train")
        for row in docs_ds:
            doc_id = str(row.get("id", row.get("doc_id", "")))
            text   = row.get("text", row.get("content", ""))
            if doc_id and text:
                self.documents[doc_id] = text
        logger.info(f"  {len(self.documents)} documents loaded.")

        # ── Queries ───────────────────────────────────────────────────────
        logger.info("Loading LoCoV1-Queries …")
        queries_ds = load_dataset(self._HF_QUERIES, cache_dir=queries_cache, split="train")
        if self.max_queries:
            queries_ds = queries_ds.select(range(min(self.max_queries, len(queries_ds))))

        for row in queries_ds:
            qid       = str(row.get("id", row.get("query_id", "")))
            query_txt = row.get("query", row.get("question", ""))
            gold_ids  = row.get("answer_pids", row.get("gold_doc_ids", []))
            if isinstance(gold_ids, str):
                gold_ids = [gold_ids]

            self.queries.append({
                "query_id":    qid,
                "query":       query_txt,
                "gold_doc_ids": [str(g) for g in gold_ids],
            })
            self.qrels[qid] = {str(g): 1 for g in gold_ids}

            # Gold spans: full document text as span (LoCoV1 uses doc-level relevance)
            self.gold_spans[qid] = [
                (str(g), 0, len(self.documents.get(str(g), "")))
                for g in gold_ids
                if str(g) in self.documents
            ]

        logger.info(f"  {len(self.queries)} queries loaded.")

    # ── Evaluation helpers ─────────────────────────────────────────────────

    def recall_at_k(self, query_id: str, retrieved_ids: List[str], k: int) -> float:
        gold = set(self.qrels.get(query_id, {}).keys())
        if not gold:
            return 0.0
        hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in gold)
        return hits / len(gold)

    def character_precision_recall(
        self,
        query_id: str,
        retrieved_texts: List[str],
        overlap_threshold: float = 0.5,
    ) -> Tuple[float, float]:
        gold_spans = self.gold_spans.get(query_id, [])
        if not gold_spans:
            return 0.0, 0.0

        covered, tps = set(), 0
        for ret_text in retrieved_texts:
            hit = False
            for idx, (doc_id, s, e) in enumerate(gold_spans):
                if idx in covered:
                    continue
                gold_text = self.documents.get(doc_id, "")[s:e]
                if not gold_text:
                    continue
                overlap = sum(c in ret_text for c in gold_text)
                if overlap / max(len(gold_text), 1) >= overlap_threshold:
                    covered.add(idx)
                    hit = True
            if hit:
                tps += 1

        precision = tps / max(len(retrieved_texts), 1)
        recall    = len(covered) / max(len(gold_spans), 1)
        return precision, recall

    def get_qrel_dict(self) -> Dict:
        return dict(self.qrels)

    def __len__(self) -> int:
        return len(self.queries)

    def summary(self) -> str:
        return (
            f"LoCoV1: {len(self.queries)} queries, "
            f"{len(self.documents)} documents"
        )
