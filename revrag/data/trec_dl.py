"""
TREC DL document ranking data loader.

Uses the MS MARCO document collection with TREC DL 2019-2023 query/qrel sets.
Documents are loaded as full text (not chunked) for Mamba; chunking is done
inside the chunked-Transformer baseline only.

Reviewer note: We use the official MS MARCO document collection
(~3.2M documents). For dev/smoke-test runs we subsample to `max_docs`.
Filtering decision: documents with fewer than 50 words are dropped as they
carry insufficient signal for long-document retrieval evaluation.
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TRECDLDataset(Dataset):
    """
    MS MARCO document collection with TREC DL query/qrel pairs.

    Args:
        collection_path: TSV with columns [doc_id, url, title, body]
        queries_path:    TSV with columns [query_id, query_text]
        qrels_path:      TSV with columns [query_id, 0, doc_id, relevance]
        max_doc_length:  Token budget for full-document encoding
        max_docs:        Subsample collection (None = full)
        split:           "train" | "dev" | "test"
    """

    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        qrels_path: str,
        tokenizer=None,
        max_doc_length: int = 4096,
        max_docs: Optional[int] = None,
        split: str = "train",
        min_word_count: int = 50,
    ):
        self.collection_path = collection_path
        self.max_doc_length = max_doc_length
        self.tokenizer = tokenizer
        self.split = split
        self.min_word_count = min_word_count

        self.documents: Dict[str, Dict] = {}
        self.queries: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}

        logger.info(f"Loading TREC DL {split} split...")
        self._load_collection(collection_path, max_docs)
        self._load_queries(queries_path)
        self._load_qrels(qrels_path)
        logger.info(
            f"Loaded {len(self.documents)} docs, {len(self.queries)} queries, "
            f"{sum(len(v) for v in self.qrels.values())} qrels"
        )

    def _load_collection(self, path: str, max_docs: Optional[int]) -> None:
        """Load and filter MS MARCO document collection."""
        count = 0
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in tqdm(reader, desc="Loading collection"):
                if len(row) < 4:
                    continue
                doc_id, url, title, body = row[0], row[1], row[2], row[3]
                text = f"{title} {body}".strip()
                if len(text.split()) < self.min_word_count:
                    continue  # filter: too short, dropped (see module docstring)
                self.documents[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "body": body,
                    "text": text,
                    "url": url,
                }
                count += 1
                if max_docs and count >= max_docs:
                    break

    def _load_queries(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.queries[parts[0]] = parts[1]

    def _load_qrels(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
                    if qid not in self.qrels:
                        self.qrels[qid] = {}
                    self.qrels[qid][did] = rel

    def get_query_list(self) -> List[Tuple[str, str]]:
        """Return [(query_id, query_text), ...] for evaluation."""
        return [
            (qid, text)
            for qid, text in self.queries.items()
            if qid in self.qrels
        ]

    def get_doc_texts(self) -> List[Tuple[str, str]]:
        """Return [(doc_id, doc_text), ...] for indexing."""
        return [(did, d["text"]) for did, d in self.documents.items()]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        qid = list(self.queries.keys())[idx]
        return {
            "query_id": qid,
            "query": self.queries[qid],
            "relevant_docs": self.qrels.get(qid, {}),
        }


def load_trec_dl(cfg: dict, tokenizer=None) -> TRECDLDataset:
    """Convenience loader from config dict."""
    c = cfg["data"]["trec_dl"]
    return TRECDLDataset(
        collection_path=c["collection_path"],
        queries_path=c["queries_path"],
        qrels_path=c["qrels_path"],
        tokenizer=tokenizer,
        max_doc_length=c.get("max_doc_length", 4096),
        max_docs=c.get("max_docs", None),
    )
