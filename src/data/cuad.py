"""
src/data/cuad.py
================
CUAD amendment pairs extractor.

Reads from local CUAD_v1.json (preferred) or HuggingFace cache as fallback.
Local path checked: data/CUAD_v1/CUAD_v1.json

CUAD (Hendrycks et al., NeurIPS 2021) contains 510 commercial contracts with
expert annotations across 41 categories. We exploit categories that describe
genuine amendment / revision relationships to extract real gold-decoy pairs:

  gold  = LATER span  (authoritative, supersedes earlier)
  decoy = EARLIER span (superseded, should NOT be top-1)

Gives hard negatives for training AND a benchmark metric:
  CRA@1  = fraction where gold is ranked above decoy at k=1
  DPR    = fraction where decoy is top-1 but gold is not (lower = better)
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AMENDMENT_CATEGORIES = [
    "Amendment Exception",
    "Renewal Term",
    "Expiration Date",
    "Anti-Assignment",
    "Termination For Convenience",
    "Notice Period To Terminate Renewal",
]

CATEGORY_QUERIES = {
    "Amendment Exception":              "What are the exceptions to amendment requirements?",
    "Renewal Term":                     "What is the renewal term of this agreement?",
    "Expiration Date":                  "When does this agreement expire?",
    "Anti-Assignment":                  "Can this agreement be assigned to a third party?",
    "Termination For Convenience":      "Can either party terminate this agreement for convenience?",
    "Notice Period To Terminate Renewal": "What notice is required to terminate renewal?",
}


class CUADAmendmentDataset:
    """
    Extracts amendment pairs from CUAD expert annotations.

    Parameters
    ----------
    hf_cache_dir : path to HuggingFace cache (fallback if local JSON missing)
    categories   : which CUAD categories to extract (default: all 6)
    min_char_gap : minimum character distance between two spans
    seed         : for reproducible train/val/test split
    """

    def __init__(
        self,
        hf_cache_dir: Path | str,
        categories: Optional[List[str]] = None,
        min_char_gap: int = 500,
        seed: int = 42,
    ):
        self.hf_cache_dir = str(hf_cache_dir)
        self.categories   = categories or AMENDMENT_CATEGORIES
        self.min_char_gap = min_char_gap
        self.seed         = seed

        self.items:    List[Dict] = []
        self.triplets: List[Tuple[str, str, str]] = []

        self._load()
        self._split()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        rows = self._load_rows()

        grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
        for ex in rows:
            title    = ex.get("title", "")
            question = ex.get("question", "")
            context  = ex.get("context", "")
            answers  = ex.get("answers", {})

            cat = None
            for c in self.categories:
                if c.lower() in question.lower():
                    cat = c
                    break
            if cat is None:
                continue

            texts  = answers.get("text", [])
            starts = answers.get("answer_start", [])
            if len(texts) < 2:
                continue

            spans = sorted(zip(starts, texts), key=lambda x: x[0])
            grouped[(title, cat, context[:50])].append({
                "category": cat,
                "query":    CATEGORY_QUERIES[cat],
                "context":  context,
                "spans":    spans,
            })

        item_id = 0
        for (title, cat, _), entries in grouped.items():
            for entry in entries:
                spans = entry["spans"]
                for i in range(len(spans) - 1):
                    s_early, t_early = spans[i]
                    s_late,  t_late  = spans[i + 1]
                    gap = abs(s_late - s_early)
                    if gap < self.min_char_gap:
                        continue
                    if len(t_early) < 20 or len(t_late) < 20:
                        continue
                    self.items.append({
                        "item_id":    f"cuad_{item_id:04d}",
                        "category":   cat,
                        "query":      entry["query"],
                        "gold_text":  t_late,
                        "decoy_text": t_early,
                        "char_gap":   gap,
                        "contract":   entry["context"],
                    })
                    self.triplets.append((entry["query"], t_late, t_early))
                    item_id += 1

        logger.info(f"  {len(self.items)} CUAD amendment pairs extracted.")

    def _load_rows(self) -> List[Dict]:
        """Try local CUAD_v1.json first, then HF cache."""
        hf_parent = Path(self.hf_cache_dir).parent
        local_candidates = [
            hf_parent / "CUAD_v1" / "CUAD_v1.json",
            hf_parent / "cuad_v1.json",
            hf_parent / "CUAD_v1.json",
        ]
        local_path = next((p for p in local_candidates if p.exists()), None)

        if local_path:
            logger.info(f"Loading CUAD from local file: {local_path}")
            raw = json.load(open(local_path, encoding="utf-8"))
            rows = []
            for contract in raw.get("data", []):
                title = contract.get("title", "")
                for para in contract.get("paragraphs", []):
                    context = para.get("context", "")
                    for qa in para.get("qas", []):
                        ans_list = qa.get("answers", [])
                        rows.append({
                            "title":    title,
                            "question": qa.get("question", ""),
                            "context":  context,
                            "answers": {
                                "text":         [a["text"] for a in ans_list],
                                "answer_start": [a["answer_start"] for a in ans_list],
                            },
                        })
            logger.info(f"  {len(rows)} CUAD QA pairs loaded from local JSON.")
            return rows

        logger.info("Loading CUAD from HuggingFace cache …")
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(
            "theatticusproject/cuad",
            cache_dir=self.hf_cache_dir,
            split="train",
            verification_mode="no_checks",
        )
        logger.info(f"  {len(ds)} CUAD examples loaded from HF cache.")
        return list(ds)

    # ── Split ─────────────────────────────────────────────────────────────────

    def _split(self):
        rng  = random.Random(self.seed)
        idxs = list(range(len(self.items)))
        rng.shuffle(idxs)
        n     = len(idxs)
        n_tr  = int(n * 0.70)
        n_val = int(n * 0.15)
        self.train_idx = idxs[:n_tr]
        self.val_idx   = idxs[n_tr: n_tr + n_val]
        self.test_idx  = idxs[n_tr + n_val:]
        logger.info(
            f"  CUAD splits — train:{len(self.train_idx)} "
            f"val:{len(self.val_idx)} test:{len(self.test_idx)}"
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_split(self, split: str) -> List[Dict]:
        idx = {"train": self.train_idx, "val": self.val_idx,
               "test": self.test_idx}[split]
        return [self.items[i] for i in idx]

    def __len__(self) -> int:
        return len(self.items)

    def summary(self) -> str:
        return (
            f"CUAD Amendments | {len(self.items)} pairs | "
            f"train:{len(self.train_idx)} "
            f"val:{len(self.val_idx)} test:{len(self.test_idx)}"
        )

    # ── Eval helpers ──────────────────────────────────────────────────────────

    def cra_at_k(self, item_id: str, retrieved_texts: List[str], k: int = 1) -> float:
        """Conflict Resolution Accuracy @ k."""
        item = next((x for x in self.items if x["item_id"] == item_id), None)
        if item is None:
            return 0.0
        gold_lower  = item["gold_text"].lower()
        decoy_lower = item["decoy_text"].lower()
        gold_pos  = next(
            (i for i, t in enumerate(retrieved_texts[:k])
             if _overlap(t.lower(), gold_lower) > 0.5), None
        )
        decoy_pos = next(
            (i for i, t in enumerate(retrieved_texts[:k])
             if _overlap(t.lower(), decoy_lower) > 0.5), None
        )
        if gold_pos is None:
            return 0.0
        if decoy_pos is not None and decoy_pos < gold_pos:
            return 0.0
        return 1.0

    def decoy_promotion_rate(self, retrieved_texts: List[str], k: int = 1) -> float:
        """DPR: fraction where decoy is in top-k but gold is not."""
        if not self.items:
            return 0.0
        promoted = 0
        for item in self.items:
            gold_in  = any(_overlap(t.lower(), item["gold_text"].lower()) > 0.5
                           for t in retrieved_texts[:k])
            decoy_in = any(_overlap(t.lower(), item["decoy_text"].lower()) > 0.5
                           for t in retrieved_texts[:k])
            if decoy_in and not gold_in:
                promoted += 1
        return promoted / len(self.items)


def _overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))
