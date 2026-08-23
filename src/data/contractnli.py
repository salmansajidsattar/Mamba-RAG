"""
src/data/contractnli.py
=======================
ContractNLI dataset loader with official train/dev/test splits.

Source: https://github.com/stanfordnlp/contract-nli
        JSON files sit directly in the contract-nli/ folder (train.json, dev.json, test.json)
        OR inside a data/ sub-folder — both layouts are handled automatically.

For bi-encoder training we produce (query, positive, hard_negative) triples:
  query         = hypothesis text
  positive      = Entailment evidence span (gold passage)
  hard_negative = Contradiction span (same doc, different meaning) OR
                  NotMentioned span (same doc, no evidence)

The Contradiction subset is also used as the cross-page consistency test:
  RevRAG's verifier must rank the authoritative clause above contradicting ones.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContractNLIDataset:
    """
    Loads ContractNLI from official JSON files and exposes:
      - .queries    : list of query dicts
      - .documents  : {doc_id: full_text}
      - .gold_spans : {query_id: [(doc_id, start, end, text)]}
      - .qrels      : {query_id: {doc_id: 1}}
      - .triplets   : list of (query, positive_text, hard_neg_text) for training

    Parameters
    ----------
    data_dir : Path
        Root of cloned contract-nli repo (JSON files may be here directly
        OR inside a data/ sub-folder).
    split : "train" | "dev" | "test"
    label_filter : which NLI labels to load (default: Entailment + Contradiction)
    min_span_len : minimum char length for a span to be valid
    seed : random seed for hard-negative sampling
    """

    LABEL_MAP = {"Entailment": 1, "Contradiction": -1, "NotMentioned": 0}

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        label_filter: Optional[List[str]] = None,
        min_span_len: int = 20,
        seed: int = 42,
    ):
        self.data_dir     = Path(data_dir)
        self.split        = split
        self.label_filter = label_filter or ["Entailment", "Contradiction"]
        self.min_span_len = min_span_len
        self.seed         = seed

        self.queries:    List[Dict]              = []
        self.documents:  Dict[str, str]          = {}
        self.gold_spans: Dict[str, List[Tuple]]  = {}
        self.qrels:      Dict[str, Dict[str, int]] = {}
        self.triplets:   List[Tuple[str, str, str]] = []
        self._label_counts: Dict[str, int]       = defaultdict(int)

        self._load()
        if split == "train":
            self._build_triplets()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        # JSON files may sit directly in data_dir OR inside a data/ sub-folder.
        # Try both layouts so the loader works regardless of how the repo was cloned.
        for candidate in [
            self.data_dir / f"{self.split}.json",
            self.data_dir / "data" / f"{self.split}.json",
        ]:
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"ContractNLI {self.split}.json not found.\n"
                f"Tried:\n"
                f"  {self.data_dir / (self.split + '.json')}\n"
                f"  {self.data_dir / 'data' / (self.split + '.json')}\n"
                f"Clone: git clone https://github.com/stanfordnlp/contract-nli\n"
                f"Then point CFG.data.contractnli_dir at the contract-nli/ folder."
            )

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        hypotheses = raw.get("labels", {})   # hyp_id -> {hypothesis, ...}

        for doc in raw.get("documents", []):
            doc_id = str(doc["id"])
            text   = doc.get("text", "")
            if not text:
                continue

            self.documents[doc_id] = text
            char_spans = doc.get("spans", [])   # [{start, end}, ...]

            for ann_set in doc.get("annotation_sets", []):
                for hyp_id, ann in ann_set.get("annotations", {}).items():
                    choice = ann.get("choice", "")
                    if choice not in self.label_filter:
                        continue

                    hyp_text = hypotheses.get(hyp_id, {}).get("hypothesis", "").strip()
                    if not hyp_text:
                        continue

                    # ann["spans"] = indices into doc["spans"]
                    valid_spans = []
                    for idx in ann.get("spans", []):
                        if idx >= len(char_spans):
                            continue
                        span = char_spans[idx]
                        s, e = (span[0], span[1]) if isinstance(span, list) else (span["start"], span["end"])
                        if (e - s) >= self.min_span_len:
                            valid_spans.append((doc_id, s, e, text[s:e]))

                    query_id = f"{doc_id}__{hyp_id}"
                    self.queries.append({
                        "query_id":      query_id,
                        "hypothesis":    hyp_text,
                        "label":         choice,
                        "doc_id":        doc_id,
                        "has_gold_span": len(valid_spans) > 0,
                    })
                    self.gold_spans[query_id] = valid_spans
                    self.qrels[query_id]      = {doc_id: 1}
                    self._label_counts[choice] += 1

        logger.info(
            f"ContractNLI [{self.split}]: {len(self.queries)} queries, "
            f"{len(self.documents)} contracts | {dict(self._label_counts)}"
        )

    # ── Training triplet construction ─────────────────────────────────────────

    def _build_triplets(self):
        """
        Build (query, positive_span, hard_negative_span) triplets for InfoNCE.

        Positive  = Entailment span for the same hypothesis
        Hard neg  = Contradiction span for same hypothesis (same doc, different meaning)
                    OR a random Entailment span from a DIFFERENT hypothesis (in-doc)
        """
        rng = random.Random(self.seed)

        # Group by doc_id for in-document negative mining
        doc_to_entailments: Dict[str, List[str]] = defaultdict(list)
        for q in self.queries:
            if q["label"] == "Entailment" and q["has_gold_span"]:
                doc_to_entailments[q["doc_id"]].append(q["query_id"])

        for q in self.queries:
            if q["label"] != "Entailment" or not q["has_gold_span"]:
                continue

            qid      = q["query_id"]
            hyp_text = q["hypothesis"]
            doc_id   = q["doc_id"]
            pos_spans = self.gold_spans[qid]

            for _, ps, pe, pos_text in pos_spans:
                hard_neg = None

                # Hard neg 1: contradiction span for same hypothesis in same doc
                for other_q in self.queries:
                    if (other_q["doc_id"] == doc_id
                            and other_q["label"] == "Contradiction"
                            and other_q["has_gold_span"]
                            and other_q["query_id"].split("__")[1]
                            == qid.split("__")[1]):
                        neg_spans = self.gold_spans[other_q["query_id"]]
                        if neg_spans:
                            hard_neg = rng.choice(neg_spans)[3]
                        break

                # Hard neg 2: random entailment from different hypothesis, same doc
                if hard_neg is None:
                    same_doc_ents = [
                        qid2 for qid2 in doc_to_entailments[doc_id]
                        if qid2 != qid
                    ]
                    if same_doc_ents:
                        neg_qid   = rng.choice(same_doc_ents)
                        neg_spans = self.gold_spans[neg_qid]
                        if neg_spans:
                            hard_neg = rng.choice(neg_spans)[3]

                # Hard neg 3: random span from same document
                if hard_neg is None:
                    doc_text   = self.documents[doc_id]
                    rand_start = rng.randint(0, max(0, len(doc_text) - 200))
                    hard_neg   = doc_text[rand_start: rand_start + 200]

                self.triplets.append((hyp_text, pos_text, hard_neg))

        logger.info(f"ContractNLI training triplets: {len(self.triplets)}")

    # ── Retrieval eval helpers ────────────────────────────────────────────────

    def get_by_label(self, label: str) -> List[Dict]:
        return [q for q in self.queries if q["label"] == label]

    def get_contradiction_queries(self) -> List[Dict]:
        """Cross-page consistency test — Contradiction examples only."""
        return self.get_by_label("Contradiction")

    def character_overlap(
        self,
        query_id: str,
        retrieved_texts: List[str],
        threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """Character-level precision and recall (LegalBench-RAG protocol)."""
        gold = self.gold_spans.get(query_id, [])
        if not gold:
            return 0.0, 0.0
        covered, tps = set(), 0
        for ret in retrieved_texts:
            hit = False
            for i, (_, _, _, g_text) in enumerate(gold):
                if i in covered:
                    continue
                ov = sum(c in ret for c in g_text) / max(len(g_text), 1)
                if ov >= threshold:
                    covered.add(i)
                    hit = True
            if hit:
                tps += 1
        return tps / max(len(retrieved_texts), 1), len(covered) / max(len(gold), 1)

    def get_triplets(self) -> List[Tuple[str, str, str]]:
        """Return (query, positive_text, hard_neg_text) for training."""
        return self.triplets

    def get_passages(
        self,
        window: int = 512,
        stride: int = 256,
    ) -> Tuple[List[str], List[str]]:
        """
        Chunk all contracts into overlapping passages.
        Returns (passage_ids, passage_texts) for building a retrieval index.
        """
        ids, texts = [], []
        for doc_id, text in self.documents.items():
            for i, start in enumerate(range(0, max(1, len(text) - window + 1), stride)):
                end = min(start + window, len(text))
                ids.append(f"{doc_id}_p{i}")
                texts.append(text[start:end])
        return ids, texts

    def get_doc_passages(
        self,
        doc_id: str,
        window: int = 512,
        stride: int = 256,
    ) -> List[Dict]:
        """Chunk a single contract into overlapping passages (with metadata)."""
        text     = self.documents.get(doc_id, "")
        passages = []
        for i, start in enumerate(range(0, max(1, len(text) - window + 1), stride)):
            end = min(start + window, len(text))
            passages.append({
                "passage_id": f"{doc_id}_p{i}",
                "doc_id":     doc_id,
                "text":       text[start:end],
                "char_start": start,
                "char_end":   end,
            })
        return passages

    def __len__(self) -> int:
        return len(self.queries)

    def summary(self) -> str:
        return (
            f"ContractNLI [{self.split}] | "
            f"{len(self.queries)} queries | "
            f"{len(self.documents)} contracts | "
            f"{len(self.triplets)} training triplets | "
            f"labels={dict(self._label_counts)}"
        )
