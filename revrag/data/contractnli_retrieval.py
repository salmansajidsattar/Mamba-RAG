"""
ContractNLI as a retrieval benchmark.

ContractNLI (Koreeda & Manning, EMNLP Findings 2021) contains 607 real
Non-Disclosure Agreements annotated by legal experts with hypothesis-span
pairs and NLI labels (Entailment / Contradiction / NotMentioned).

Data source: https://github.com/stanfordnlp/contract-nli
  File layout:
    contract-nli/data/train.json
    contract-nli/data/dev.json
    contract-nli/data/test.json

JSON structure:
  {
    "documents": [
      {
        "id": "nda-1",
        "text": "<full contract text>",
        "spans": [{"start": 0, "end": 120}, ...],   ← char offsets
        "annotation_sets": [
          {
            "annotations": {
              "nda-1": {
                "choice": "Entailment",
                "spans": [0, 3]   ← indices into document["spans"]
              }
            }
          }
        ]
      }
    ],
    "labels": {
      "nda-1": { "hypothesis": "The agreement is perpetual." }
    }
  }

Framing as retrieval:
  Query  = hypothesis text
  Corpus = passages (sentence-chunked from full contract text)
  Gold   = passages that overlap with expert-annotated evidence spans

Cross-page consistency test:
  Contradiction examples = cases where a clause contradicts the hypothesis.
  RevRAG's verifier should rank the authoritative clause above confounders.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContractNLIRetrievalDataset:
    """
    ContractNLI framed as a passage-retrieval benchmark.

    Parameters
    ----------
    data_dir : str | Path
        Root of the cloned contract-nli repo
        (contains data/train.json, data/dev.json, data/test.json).
    split : str
        "train", "dev", or "test". Use "test" for final evaluation.
    label_filter : list of str or None
        Only load examples with these NLI labels.
        Default: ["Entailment", "Contradiction"] (skip NotMentioned).
    min_span_length : int
        Minimum character length of a gold span.

    Attributes
    ----------
    queries     : list[dict]  — {query_id, hypothesis, label, doc_id}
    documents   : dict        — {doc_id: full_contract_text}
    gold_spans  : dict        — {query_id: [(doc_id, char_start, char_end, text)]}
    qrels       : dict        — {query_id: {doc_id: 1}}
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "test",
        label_filter: Optional[List[str]] = None,
        min_span_length: int = 20,
    ) -> None:
        self.data_dir         = Path(data_dir)
        self.split            = split
        self.label_filter     = label_filter or ["Entailment", "Contradiction"]
        self.min_span_length  = min_span_length

        self.queries:    List[Dict]  = []
        self.documents:  Dict        = {}
        self.gold_spans: Dict        = {}
        self.qrels:      Dict        = {}
        self._label_counts: Dict     = defaultdict(int)

        self._load()

    # ── Internal loader ────────────────────────────────────────────────────

    def _load(self) -> None:
        # "dev" split → dev.json, else train.json / test.json
        filename  = f"{self.split}.json"
        json_path = self.data_dir / "data" / filename

        if not json_path.exists():
            raise FileNotFoundError(
                f"ContractNLI data file not found: {json_path}\n"
                f"Run step 3 to clone https://github.com/stanfordnlp/contract-nli"
            )

        logger.info(f"Loading ContractNLI from {json_path} …")
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        documents_raw = raw.get("documents", [])
        hypotheses    = raw.get("labels", {})   # hypothesis_id → {"hypothesis": str}

        for doc in documents_raw:
            doc_id       = str(doc["id"])
            contract_txt = doc.get("text", "")
            if not contract_txt:
                continue

            self.documents[doc_id] = contract_txt

            # doc["spans"] = list of {"start": int, "end": int} char offsets
            char_spans = doc.get("spans", [])

            for ann_set in doc.get("annotation_sets", []):
                for hyp_id, ann in ann_set.get("annotations", {}).items():
                    choice = ann.get("choice", "")
                    if choice not in self.label_filter:
                        continue

                    hyp_entry  = hypotheses.get(hyp_id, {})
                    hypothesis = hyp_entry.get("hypothesis", "").strip()
                    if not hypothesis:
                        continue

                    # ann["spans"] = indices into doc["spans"]
                    span_indices = ann.get("spans", [])
                    valid_spans  = []
                    for idx in span_indices:
                        if idx >= len(char_spans):
                            continue
                        s = char_spans[idx]["start"]
                        e = char_spans[idx]["end"]
                        if (e - s) < self.min_span_length:
                            continue
                        span_text = contract_txt[s:e]
                        valid_spans.append((doc_id, s, e, span_text))

                    query_id = f"{doc_id}__{hyp_id}"
                    self.queries.append({
                        "query_id":      query_id,
                        "hypothesis":    hypothesis,
                        "label":         choice,
                        "doc_id":        doc_id,
                        "has_gold_span": len(valid_spans) > 0,
                    })
                    self.gold_spans[query_id] = valid_spans
                    self.qrels[query_id]      = {doc_id: 1}
                    self._label_counts[choice] += 1

        logger.info(
            f"ContractNLI [{self.split}]: {len(self.queries)} queries, "
            f"{len(self.documents)} contracts, labels={dict(self._label_counts)}"
        )

    # ── Evaluation helpers ─────────────────────────────────────────────────

    def character_overlap(
        self,
        query_id: str,
        retrieved_passages: List[str],
        overlap_threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Character-level precision and recall.

        A retrieved passage is a hit if it shares >= overlap_threshold
        fraction of characters with any gold span.
        Returns (precision, recall).
        """
        gold = self.gold_spans.get(query_id, [])
        if not gold:
            return 0.0, 0.0

        covered       = set()
        true_positives = 0

        for ret_text in retrieved_passages:
            hit = False
            for idx, (_, start, end, gold_text) in enumerate(gold):
                if idx in covered:
                    continue
                overlap = sum(c in ret_text for c in gold_text)
                if overlap / max(len(gold_text), 1) >= overlap_threshold:
                    covered.add(idx)
                    hit = True
            if hit:
                true_positives += 1

        precision = true_positives / max(len(retrieved_passages), 1)
        recall    = len(covered)   / max(len(gold), 1)
        return precision, recall

    def get_by_label(self, label: str) -> List[Dict]:
        return [q for q in self.queries if q["label"] == label]

    def get_contradiction_queries(self) -> List[Dict]:
        """Return only Contradiction examples — the cross-page consistency test."""
        return self.get_by_label("Contradiction")

    def get_qrel_dict(self) -> Dict:
        return dict(self.qrels)

    def summary(self) -> str:
        return (
            f"ContractNLI [{self.split}]: "
            f"{len(self.queries)} queries, "
            f"{len(self.documents)} contracts, "
            f"labels={dict(self._label_counts)}"
        )

    def __len__(self) -> int:
        return len(self.queries)
