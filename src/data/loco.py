"""
src/data/loco.py
================
LoCoV1 loader — reads from local JSONL files in the HF cache snapshot.

HuggingFace repos (already cached locally):
  hazyresearch/LoCoV1-Documents  → data/hf_cache/datasets--hazyresearch--LoCoV1-Documents/
  hazyresearch/LoCoV1-Queries    → data/hf_cache/datasets--hazyresearch--LoCoV1-Queries/

Only the "test" split exists on HF; we split it 70/15/15 internally.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DOCS_CACHE    = "datasets--hazyresearch--LoCoV1-Documents"
_QUERIES_CACHE = "datasets--hazyresearch--LoCoV1-Queries"


class LoCoV1Dataset:
    """
    Wraps LoCoV1 for retrieval evaluation.

    Parameters
    ----------
    cache_dir   : HF cache root
    split       : "train" | "val" | "test"
    max_queries : cap on queries
    max_docs    : cap on corpus size
    seed        : for internal 70/15/15 split
    """

    def __init__(
        self,
        cache_dir:   Optional[Path] = None,
        split:       str  = "test",
        max_queries: Optional[int] = None,
        max_docs:    Optional[int] = None,
        seed:        int  = 42,
    ):
        self.split       = split
        self.max_queries = max_queries
        self.max_docs    = max_docs
        self.seed        = seed
        self.cache_dir   = Path(cache_dir) if cache_dir else None

        self.examples: List[Dict] = []
        self.corpus:   Dict[str, str] = {}
        self._load()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _snapshot_dir(self, cache_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        root = self.cache_dir / cache_name
        if not root.exists():
            return None
        # snapshots/<hash>/
        snaps = list((root / "snapshots").iterdir()) if (root / "snapshots").exists() else []
        return snaps[0] if snaps else root

    def _iter_jsonl(self, directory: Path):
        """Yield dicts from all *.jsonl files under directory."""
        for fpath in sorted(directory.rglob("*.jsonl")):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        # ── Documents ─────────────────────────────────────────────────────────
        doc_snap = self._snapshot_dir(_DOCS_CACHE)
        if doc_snap is None:
            raise FileNotFoundError(
                f"LoCoV1-Documents cache not found at {self.cache_dir}/{_DOCS_CACHE}"
            )

        logger.info(f"Loading LoCoV1 documents from {doc_snap}")
        for row in self._iter_jsonl(doc_snap):
            doc_id = str(row.get("pid") or row.get("doc_id") or row.get("id") or "")
            text   = row.get("passage") or row.get("text") or ""
            if doc_id and text:
                self.corpus[doc_id] = text

        if self.max_docs and len(self.corpus) > self.max_docs:
            keys = list(self.corpus.keys())[: self.max_docs]
            self.corpus = {k: self.corpus[k] for k in keys}

        logger.info(f"  LoCoV1 corpus: {len(self.corpus)} documents")

        # ── Queries ───────────────────────────────────────────────────────────
        q_snap = self._snapshot_dir(_QUERIES_CACHE)
        if q_snap is None:
            raise FileNotFoundError(
                f"LoCoV1-Queries cache not found at {self.cache_dir}/{_QUERIES_CACHE}"
            )

        logger.info(f"Loading LoCoV1 queries from {q_snap}")
        all_qs: List[Dict] = []
        for row in self._iter_jsonl(q_snap):
            qid      = str(row.get("qid") or row.get("id") or "")
            query    = row.get("query") or row.get("question") or ""
            gold_ids = row.get("answer_pids") or row.get("gold_doc_ids") or []
            if isinstance(gold_ids, str):
                gold_ids = [gold_ids]
            gold_ids = [str(g) for g in gold_ids]
            if qid and query:
                all_qs.append({"query_id": qid, "query": query, "gold_ids": gold_ids})

        # Internal 70/15/15 split (HF only has "test")
        rng = random.Random(self.seed)
        rng.shuffle(all_qs)
        n     = len(all_qs)
        n_val = max(1, int(0.15 * n))
        n_tst = max(1, int(0.15 * n))
        sp_map = {
            "train": all_qs[n_val + n_tst :],
            "val":   all_qs[: n_val],
            "test":  all_qs[n_val : n_val + n_tst],
        }
        chosen = sp_map.get(self.split, [])
        if self.max_queries:
            chosen = chosen[: self.max_queries]

        self.examples = chosen
        logger.info(f"  LoCoV1 {self.split}: {len(self.examples)} queries")

    # ── Accessors ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> Dict:
        return self.examples[idx]

    def get_queries(self) -> List[str]:
        return [e["query"] for e in self.examples]

    def get_qrels(self) -> Dict[str, set]:
        return {e["query_id"]: set(e["gold_ids"]) for e in self.examples}

    def get_corpus(self) -> Dict[str, str]:
        return dict(self.corpus)

    def get_all_passages(self) -> Tuple[List[str], List[str]]:
        ids   = list(self.corpus.keys())
        texts = [self.corpus[i] for i in ids]
        return ids, texts
