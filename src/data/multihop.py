"""
src/data/multihop.py
====================
MultiHopRAG loader — reads from local JSON file, no HF download needed.

Local file detected automatically in:
  data/hf_cache/datasets--yixuantt--MultiHopRAG/snapshots/**/MultiHopRAG.json
  data/MultiHop-RAG-main/MultiHopRAG.json
  data/MultiHopRAG.json
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MultiHopRAGDataset:
    """
    Wraps MultiHopRAG for retrieval evaluation.

    Parameters
    ----------
    cache_dir   : HF cache root (used to locate the snapshot JSON)
    split       : "train" | "val" | "test"
    max_queries : cap on queries (None = all)
    seed        : random seed for 70/15/15 split
    """

    def __init__(
        self,
        cache_dir:   Optional[Path] = None,
        split:       str  = "test",
        max_queries: Optional[int] = None,
        seed:        int  = 42,
    ):
        self.split       = split
        self.max_queries = max_queries
        self.seed        = seed
        self.cache_dir   = Path(cache_dir) if cache_dir else None

        self.examples: List[Dict] = []
        self.corpus:   Dict[str, str] = {}
        self._load()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_local_json(self) -> Optional[Path]:
        candidates = []
        if self.cache_dir:
            snap_root = self.cache_dir / "datasets--yixuantt--MultiHopRAG"
            if snap_root.exists():
                candidates += list(snap_root.rglob("MultiHopRAG.json"))
            candidates += [
                self.cache_dir.parent / "MultiHop-RAG-main" / "MultiHopRAG.json",
                self.cache_dir.parent / "MultiHopRAG.json",
            ]
        return next(
            (p for p in candidates if p.exists() and p.stat().st_size > 10_000),
            None,
        )

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        local = self._find_local_json()
        if local is None:
            raise FileNotFoundError(
                "MultiHopRAG.json not found.\n"
                "Expected at: data/hf_cache/datasets--yixuantt--MultiHopRAG/"
                "snapshots/**/MultiHopRAG.json\n"
                "Or: data/MultiHop-RAG-main/MultiHopRAG.json"
            )

        logger.info(f"Loading MultiHopRAG from {local}")
        raw = json.load(open(local, encoding="utf-8"))

        all_examples = []
        for i, row in enumerate(raw):
            evidence = row.get("evidence_list", [])
            if not evidence:
                continue

            gold_ids, gold_texts = [], []
            for j, ev in enumerate(evidence):
                fact  = ev.get("fact") or ev.get("body") or ev.get("text") or ""
                title = ev.get("title", "")
                if not fact.strip():
                    continue
                doc_id = f"multihop_{i}_{j}"
                self.corpus[doc_id] = (f"[{title}] {fact}".strip()
                                       if title else fact)
                gold_ids.append(doc_id)
                gold_texts.append(fact)

            if not gold_ids:
                continue

            all_examples.append({
                "query_id":   f"mh_{i}",
                "query":      row["query"],
                "answer":     row.get("answer", ""),
                "gold_ids":   gold_ids,
                "gold_texts": gold_texts,
                "type":       row.get("question_type", ""),
            })

        # Reproducible 70/15/15 split
        rng = random.Random(self.seed)
        shuffled = all_examples[:]
        rng.shuffle(shuffled)
        n     = len(shuffled)
        n_val = max(1, int(0.15 * n))
        n_tst = max(1, int(0.15 * n))

        splits_map = {
            "train": shuffled[: n - n_val - n_tst],
            "val":   shuffled[n - n_val - n_tst : n - n_tst],
            "test":  shuffled[n - n_tst :],
        }
        chosen = splits_map[self.split]
        if self.max_queries:
            chosen = chosen[: self.max_queries]

        self.examples = chosen
        logger.info(
            f"  MultiHopRAG {self.split}: {len(self.examples)} queries | "
            f"corpus: {len(self.corpus)} docs"
        )

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
        relevant = {did for e in self.examples for did in e["gold_ids"]}
        return {k: v for k, v in self.corpus.items() if k in relevant}

    def get_all_passages(self) -> Tuple[List[str], List[str]]:
        corpus = self.get_corpus()
        ids    = list(corpus.keys())
        texts  = [corpus[i] for i in ids]
        return ids, texts
