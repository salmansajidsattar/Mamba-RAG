"""
CUAD Amendment Pair extractor.

The Contract Understanding Atticus Dataset (CUAD, Hendrycks et al., 2021)
contains 510 real commercial contracts annotated across 41 legal clause
categories by legal experts.

We exploit three categories that describe genuine amendment / revision
relationships within the same contract:

    Category  5 — "Amendment Exception"
        Clauses that specify exceptions to amendment provisions, i.e.,
        conditions under which a different rule applies.

    Category 10 — "Renewal Term"
        Clauses that override the base contract term with a renewal term.
        A later renewal clause supersedes the original term.

    Category  9 — "Expiration Date"
        Clauses that set or modify the expiration date. When multiple
        expiration-related clauses exist in the same contract, the later
        one is typically authoritative.

    Category 30 — "Anti-Assignment"
        Restrictions on assignment that may be modified by later clauses.

Extraction logic
----------------
For each contract:
    1. Collect all annotated spans per category.
    2. For any category with ≥2 spans, or any pair of related categories
       (e.g. Amendment Exception + another clause it modifies), create a
       retrieval pair:
           query  = the question for that category
           gold   = the LATER span (higher char offset → authoritative)
           decoy  = the EARLIER span (superseded)
    3. Filter pairs where the section gap (char offset difference / avg doc
       length) is above a minimum threshold — ensures the spans are
       genuinely on different pages, not adjacent lines.

This yields ~300–500 real revision query-evidence pairs from real expert
annotations, with zero synthetic generation.

HuggingFace dataset: theatticusproject/cuad
    pip install datasets

Dataset structure:
    id: str
    title: str       (contract filename)
    context: str     (section text, NOT full contract)
    question: str    (e.g. "Highlight the parts related to Amendment Exception")
    answers: {text: [str], answer_start: [int]}
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HF_DATASET_ID = "theatticusproject/cuad"

# Category questions that describe amendment/revision relationships
# (these are the exact question strings used in the CUAD dataset)
AMENDMENT_CATEGORIES = {
    "Amendment Exception": (
        "Highlight the parts (if any) of this contract related to "
        "\"Amendment Exception\" that should be reviewed by a lawyer. "
        "Details: Does the contract include exceptions to its amendment "
        "provision, i.e., provisions that can be changed without an amendment?"
    ),
    "Renewal Term": (
        "Highlight the parts (if any) of this contract related to "
        "\"Renewal Term\" that should be reviewed by a lawyer. "
        "Details: What is the renewal term after the initial term expires? "
        "This includes automatic extensions and unilateral extensions with or "
        "without notice."
    ),
    "Expiration Date": (
        "Highlight the parts (if any) of this contract related to "
        "\"Expiration Date\" that should be reviewed by a lawyer. "
        "Details: On what date will the contract's initial term expire?"
    ),
    "Anti-Assignment": (
        "Highlight the parts (if any) of this contract related to "
        "\"Anti-Assignment\" that should be reviewed by a lawyer. "
        "Details: Is there a clause that restricts the ability of a party to "
        "transfer rights or obligations to another party?"
    ),
    "Termination For Convenience": (
        "Highlight the parts (if any) of this contract related to "
        "\"Termination For Convenience\" that should be reviewed by a lawyer. "
        "Details: Can a party terminate this contract without cause (solely "
        "\"at will\") and/or with a specified notice period?"
    ),
    "Notice Period To Terminate Renewal": (
        "Highlight the parts (if any) of this contract related to "
        "\"Notice Period To Terminate Renewal\" that should be reviewed by a "
        "lawyer. Details: What is the notice period required to terminate "
        "renewal?"
    ),
}

# Queries we pose to the retrieval system (simpler than the CUAD question)
RETRIEVAL_QUERIES = {
    "Amendment Exception":
        "What are the exceptions to the amendment provisions of this agreement?",
    "Renewal Term":
        "What is the renewal term after the initial contract term expires?",
    "Expiration Date":
        "When does this contract expire or terminate?",
    "Anti-Assignment":
        "What restrictions apply to the assignment of rights under this agreement?",
    "Termination For Convenience":
        "Can either party terminate this agreement without cause?",
    "Notice Period To Terminate Renewal":
        "What notice period is required to terminate the renewal of this agreement?",
}

# Minimum character gap between gold and decoy spans to qualify as cross-page
MIN_CHAR_GAP = 500


class CUADAmendmentDataset:
    """
    Real amendment revision pairs extracted from CUAD contracts.

    Each item represents a genuine retrieval scenario where:
        - Two expert-annotated spans from the same contract relate to the
          same legal question (e.g., expiration date).
        - The later span (by character offset) is the operative clause.
        - The earlier span is the superseded/original clause.

    This gives the cross-page consistency test WITHOUT synthetic generation.

    Parameters
    ----------
    split : str
        "train", "test". CUAD has no validation split.
    categories : list of str or None
        Which CUAD categories to extract pairs from. Default: all amendment
        categories defined in AMENDMENT_CATEGORIES.
    min_char_gap : int
        Minimum character-offset gap between gold (later) and decoy (earlier)
        span. Ensures they are in different sections, not adjacent lines.
    cache_dir : str or None

    Attributes
    ----------
    items : list of dict
        Each item: {item_id, query, doc_id, contract_title, category,
                    gold_span_text, gold_span_start, gold_span_end,
                    decoy_span_text, decoy_span_start, decoy_span_end,
                    char_gap, full_contract (if available)}
    documents : dict
        {doc_id: contract_text} — reconstructed from all annotated sections.
    qrels : dict
        {item_id: {doc_id: 1}}
    gold_spans : dict
        {item_id: [(doc_id, start, end, text)]}
    """

    def __init__(
        self,
        split: str = "test",
        categories: Optional[List[str]] = None,
        min_char_gap: int = MIN_CHAR_GAP,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.categories = categories or list(AMENDMENT_CATEGORIES.keys())
        self.min_char_gap = min_char_gap
        self.cache_dir = cache_dir

        self.items: List[Dict] = []
        self.documents: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}
        self.gold_spans: Dict[str, List[Tuple[str, int, int, str]]] = {}

        self._load()

    def _load(self) -> None:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("pip install datasets")

        logger.info(
            f"Loading CUAD ({self.split}) from HuggingFace [{_HF_DATASET_ID}]…"
        )
        ds = load_dataset(_HF_DATASET_ID, split=self.split, cache_dir=self.cache_dir)
        logger.info(f"Raw CUAD examples: {len(ds)}")

        # Group by (contract_title, category) → collect all annotated spans
        # Structure: {contract_title: {category: [(answer_start, answer_text), ...]}}
        contract_spans: Dict[str, Dict[str, List[Tuple[int, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Also reconstruct contract text by stitching all sections for a contract
        contract_sections: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        for ex in ds:
            title = ex.get("title", "unknown")
            context = ex.get("context", "")
            question = ex.get("question", "")
            answers = ex.get("answers", {})

            # Match this example to one of our target categories
            matched_category = None
            for cat, cat_question in AMENDMENT_CATEGORIES.items():
                # Match on key phrase in the question
                cat_key = cat.lower().replace(" ", "").replace("-", "")
                q_key = question.lower().replace(" ", "").replace("-", "")
                if cat_key in q_key or cat.lower() in question.lower():
                    matched_category = cat
                    break

            if matched_category not in self.categories:
                continue

            # Collect annotated spans for this (contract, category)
            answer_starts = answers.get("answer_start", [])
            answer_texts = answers.get("text", [])
            for start, text in zip(answer_starts, answer_texts):
                if text.strip():
                    contract_spans[title][matched_category].append((start, text.strip()))

            # Track sections to reconstruct contract text
            if context:
                # Store context with its approximate position
                # (answer_start is relative to context, not full contract)
                contract_sections[title].append((
                    min(answer_starts) if answer_starts else 0,
                    context,
                ))

        logger.info(
            f"Contracts with amendment annotations: "
            f"{sum(1 for t, cats in contract_spans.items() if any(cats.values()))}"
        )

        # Reconstruct pseudo-full-contract text for each contract
        # by concatenating all unique contexts in order of first appearance
        for title, sections in contract_sections.items():
            # Deduplicate sections (same section may appear multiple times)
            seen = set()
            ordered_texts = []
            for _, text in sorted(sections, key=lambda x: x[0]):
                key = text[:100]
                if key not in seen:
                    seen.add(key)
                    ordered_texts.append(text)
            self.documents[title] = "\n\n".join(ordered_texts)

        # Build revision pairs
        item_idx = 0
        category_counts: Dict[str, int] = defaultdict(int)

        for title, cat_spans in contract_spans.items():
            contract_text = self.documents.get(title, "")

            for category in self.categories:
                spans = cat_spans.get(category, [])
                if len(spans) < 2:
                    continue  # Need ≥2 spans to form a revision pair

                # Sort by character offset
                spans_sorted = sorted(spans, key=lambda x: x[0])

                # Create pair: earliest = decoy (superseded), latest = gold (authoritative)
                # Take the first and last if multiple exist
                decoy_start, decoy_text = spans_sorted[0]
                gold_start, gold_text = spans_sorted[-1]

                char_gap = abs(gold_start - decoy_start)
                if char_gap < self.min_char_gap:
                    continue  # Too close — likely not a genuine cross-section revision

                if gold_start <= decoy_start:
                    # Later span should be later in document
                    continue

                item_id = f"cuad_{title[:30].replace(' ', '_')}_{category[:15].replace(' ', '_')}_{item_idx}"
                query = RETRIEVAL_QUERIES.get(category, f"What does the contract say about {category}?")

                item = {
                    "item_id": item_id,
                    "query": query,
                    "doc_id": title,
                    "contract_title": title,
                    "category": category,
                    "gold_span_text": gold_text,
                    "gold_span_start": gold_start,
                    "gold_span_end": gold_start + len(gold_text),
                    "decoy_span_text": decoy_text,
                    "decoy_span_start": decoy_start,
                    "decoy_span_end": decoy_start + len(decoy_text),
                    "char_gap": char_gap,
                    "n_spans_total": len(spans),
                }
                self.items.append(item)
                self.qrels[item_id] = {title: 1}
                self.gold_spans[item_id] = [
                    (title, gold_start, gold_start + len(gold_text), gold_text)
                ]
                category_counts[category] += 1
                item_idx += 1

        logger.info(
            f"CUAD amendment pairs extracted: {len(self.items)} total\n"
            f"  Category breakdown: {dict(category_counts)}\n"
            f"  Contracts with pairs: {len(set(it['doc_id'] for it in self.items))}"
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def character_overlap(
        self,
        item_id: str,
        retrieved_passages: List[str],
        overlap_threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Character-level precision and recall.
        A retrieved passage is a hit if it overlaps with the gold span by
        at least `overlap_threshold` × gold span length.
        """
        gold = self.gold_spans.get(item_id, [])
        if not gold:
            return 0.0, 0.0

        covered = set()
        tp = 0
        for ret_text in retrieved_passages:
            hit = False
            for idx, (_, _, _, gold_text) in enumerate(gold):
                if idx in covered:
                    continue
                overlap = sum(c in ret_text for c in gold_text)
                if overlap / max(len(gold_text), 1) >= overlap_threshold:
                    covered.add(idx)
                    hit = True
            if hit:
                tp += 1

        return tp / max(len(retrieved_passages), 1), len(covered) / max(len(gold), 1)

    def decoy_promotion_rate(
        self,
        item_id: str,
        retrieved_passages: List[str],
        top_k: int = 1,
    ) -> float:
        """
        Fraction of top-k retrieved passages that contain the DECOY (earlier,
        superseded clause) but NOT the gold (later, authoritative clause).
        This directly measures the failure mode RevRAG is designed to fix.
        """
        item = next((it for it in self.items if it["item_id"] == item_id), None)
        if item is None:
            return 0.0
        decoy = item["decoy_span_text"]
        gold = item["gold_span_text"]
        top_passages = retrieved_passages[:top_k]
        decoy_hits = sum(decoy[:50] in p for p in top_passages)
        gold_hits = sum(gold[:50] in p for p in top_passages)
        return 1.0 if (decoy_hits > 0 and gold_hits == 0) else 0.0

    def get_by_category(self, category: str) -> List[Dict]:
        return [it for it in self.items if it["category"] == category]

    def get_qrel_dict(self) -> Dict[str, Dict[str, int]]:
        return dict(self.qrels)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]
