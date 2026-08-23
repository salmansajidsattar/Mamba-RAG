"""
src/data/legalbench.py
======================
LegalBench-RAG loader.

LegalBench-RAG (Pipitone & Römmele, 2024) is a character-level retrieval
benchmark for legal documents. It uses character-level precision/recall rather
than passage-level Recall@k — matching the ContractNLI evaluation protocol.

HuggingFace: "nguyen-brat/legalbench_rag"
Subsets available: "contractnli", "cuad", "maud", "privacy_qa", "sara"

We load all subsets by default and evaluate using character-level metrics.

Splits: HF provides train/test; we carve a val set from train.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_HF_ID = "nguyen-brat/legalbench_rag"

# Each subset corresponds to a legal domain
_SUBSETS = ["contractnli", "cuad", "maud", "privacy_qa", "sara"]


class LegalBenchRAGDataset:
    """
    Wraps LegalBench-RAG for character-level retrieval evaluation.

    Each example has:
      query      : str
      gold_spans : List[Tuple[int, int, str]]  — (start, end, text)
      context    : str — full document the spans come from

    Parameters
    ----------
    cache_dir  : HF cache directory
    split      : "train" | "val" | "test"
    subsets    : list of subset names to include (None = all)
    max_examples : cap on examples
    seed       : reproducible split
    """

    def __init__(
        self,
        cache_dir:    Optional[Path] = None,
        split:        str = "test",
        subsets:      Optional[List[str]] = None,
        max_examples: Optional[int] = None,
        seed:         int = 42,
    ):
        self.split        = split
        self.subsets      = subsets or _SUBSETS
        self.max_examples = max_examples
        self.seed         = seed
        self.cache_dir    = Path(cache_dir) if cache_dir else None

        self.examples: List[Dict] = []
        self._load()

    def _load(self):
        from datasets import load_dataset

        kw = dict(trust_remote_code=True)
        if self.cache_dir:
            kw["cache_dir"] = str(self.cache_dir)

        all_examples: List[Dict] = []
        hf_split_map = {"train": "train", "val": "train", "test": "test"}
        hf_sp = hf_split_map[self.split]

        for subset in self.subsets:
            try:
                ds = load_dataset(_HF_ID, subset, split=hf_sp, **kw)
            except Exception as e:
                logger.warning(f"  LegalBench-RAG/{subset} ({hf_sp}) unavailable: {e}")
                continue

            for i, row in enumerate(ds):
                # Schema: question/query, answer/gold_answer, context/passage
                query   = (row.get("question") or row.get("query") or
                           row.get("instruction") or "")
                context = (row.get("context") or row.get("passage") or
                           row.get("contract") or "")
                gold_answer = (row.get("answer") or row.get("gold_answer") or
                               row.get("relevant_sentence") or "")

                if not query or not context or not gold_answer:
                    continue

                # Locate gold_answer in context (character-level span)
                start = context.find(gold_answer)
                if start == -1:
                    # Try case-insensitive
                    start = context.lower().find(gold_answer.lower())
                    if start == -1:
                        # Use the whole answer as span [0, len(gold_answer)]
                        start = 0

                end = start + len(gold_answer)
                gold_spans = [(start, end, gold_answer)]

                all_examples.append({
                    "example_id": f"lb_{subset}_{i}",
                    "subset":     subset,
                    "query":      query,
                    "context":    context,
                    "gold_spans": gold_spans,
                    "gold_text":  gold_answer,
                })

        # Val split carved from train
        rng = random.Random(self.seed)
        if self.split == "val":
            rng.shuffle(all_examples)
            n_val = max(1, int(0.15 * len(all_examples)))
            all_examples = all_examples[:n_val]
        elif self.split == "train":
            rng.shuffle(all_examples)
            n_val = max(1, int(0.15 * len(all_examples)))
            all_examples = all_examples[n_val:]

        if self.max_examples:
            all_examples = all_examples[: self.max_examples]

        self.examples = all_examples
        logger.info(
            f"LegalBench-RAG {self.split}: {len(self.examples)} examples "
            f"across subsets {self.subsets}"
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> Dict:
        return self.examples[idx]

    def get_queries(self) -> List[str]:
        return [e["query"] for e in self.examples]

    def get_contexts(self) -> List[str]:
        """Full documents to retrieve from."""
        return [e["context"] for e in self.examples]

    def get_gold_spans(self) -> List[List[Tuple[int, int, str]]]:
        """Character-level gold spans for evaluation."""
        return [e["gold_spans"] for e in self.examples]

    def get_gold_texts(self) -> List[str]:
        return [e["gold_text"] for e in self.examples]

    def as_passages(
        self,
        window:  int = 512,
        stride:  int = 256,
    ) -> Tuple[List[str], List[str]]:
        """
        Chunk contexts into overlapping passages for retrieval indexing.
        Returns (passage_ids, passage_texts).
        """
        ids, texts = [], []
        for ex in self.examples:
            ctx  = ex["context"]
            for start in range(0, max(1, len(ctx) - window + 1), stride):
                end   = min(start + window, len(ctx))
                pid   = f"{ex['example_id']}_p{start}"
                ids.append(pid)
                texts.append(ctx[start:end])
        return ids, texts
