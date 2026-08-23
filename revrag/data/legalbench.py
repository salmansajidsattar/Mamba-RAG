"""
LegalBench-RAG data loader.

Loads CUAD, MAUD, ContractNLI, and PrivacyQA subsets from the
LegalBench-RAG benchmark (Pipitone et al., 2024; arXiv:2408.10343).

Evaluation protocol: character-level Precision@k and Recall@k, as specified
by LegalBench-RAG. NOT document-level nDCG — reviewers will check this.

Filtering decisions:
  - Documents shorter than 1,000 characters are excluded; they do not
    represent the multi-page scenario our method targets.
  - Queries with no ground-truth spans are excluded (unanswerable queries
    outside our evaluation scope).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

SUBSET_NAMES = ["cuad", "maud", "contractnli", "privacyqa"]


class LegalBenchRAGDataset(Dataset):
    """
    Multi-subset LegalBench-RAG dataset.

    Each item is a dict with:
        query_id    : str
        query       : str
        doc_id      : str
        document    : str  (full text, not chunked)
        gold_spans  : List[Dict]  each has {start, end, text}
        subset      : str
    """

    def __init__(
        self,
        data_dir: str,
        subsets: Optional[List[str]] = None,
        max_doc_length: int = 8192,
        min_doc_chars: int = 1000,
    ):
        self.data_dir = Path(data_dir)
        self.subsets = subsets or SUBSET_NAMES
        self.max_doc_length = max_doc_length
        self.min_doc_chars = min_doc_chars
        self.items: List[Dict] = []
        self.documents: Dict[str, str] = {}
        self._load_all()

    def _load_all(self) -> None:
        for subset in self.subsets:
            subset_path = self.data_dir / subset
            if not subset_path.exists():
                logger.warning(f"Subset path not found: {subset_path}, skipping.")
                continue
            self._load_subset(subset, subset_path)
        logger.info(
            f"LegalBench-RAG: {len(self.items)} query-doc pairs across "
            f"{len(self.documents)} unique documents"
        )

    def _load_subset(self, subset_name: str, path: Path) -> None:
        """
        Expected directory layout (from LegalBench-RAG repo):
            <subset>/
                documents/    (*.txt files, one per contract)
                queries.jsonl (one query per line)
                qrels.jsonl   (one qrel per line, with char offsets)
        """
        docs_dir = path / "documents"
        queries_file = path / "queries.jsonl"
        qrels_file = path / "qrels.jsonl"

        # Load documents
        doc_texts: Dict[str, str] = {}
        if docs_dir.exists():
            for doc_file in sorted(docs_dir.glob("*.txt")):
                doc_id = doc_file.stem
                text = doc_file.read_text(encoding="utf-8", errors="replace")
                if len(text) < self.min_doc_chars:
                    continue  # filter: too short (see module docstring)
                doc_texts[doc_id] = text
                self.documents[doc_id] = text

        if not queries_file.exists() or not qrels_file.exists():
            logger.warning(f"Missing queries or qrels for subset {subset_name}")
            return

        # Load qrels first (map query_id -> gold spans)
        qrels: Dict[str, List[Dict]] = {}
        with open(qrels_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["query_id"]
                if qid not in qrels:
                    qrels[qid] = []
                qrels[qid].append({
                    "doc_id": item["doc_id"],
                    "start": item["start_char"],
                    "end": item["end_char"],
                    "text": item.get("text", ""),
                })

        # Load queries and pair with docs + gold spans
        with open(queries_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["query_id"]
                if qid not in qrels:
                    continue  # filter: no gold span

                for span_info in qrels[qid]:
                    doc_id = span_info["doc_id"]
                    if doc_id not in doc_texts:
                        continue
                    self.items.append({
                        "query_id": qid,
                        "query": item["query"],
                        "doc_id": doc_id,
                        "document": doc_texts[doc_id],
                        "gold_spans": [span_info],
                        "subset": subset_name,
                    })

    def get_all_documents(self) -> Dict[str, str]:
        return self.documents

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def load_legalbench(cfg: dict) -> LegalBenchRAGDataset:
    c = cfg["data"]["legalbench"]
    return LegalBenchRAGDataset(
        data_dir=c["data_dir"],
        subsets=c.get("datasets", SUBSET_NAMES),
        max_doc_length=c.get("max_doc_length", 8192),
    )
