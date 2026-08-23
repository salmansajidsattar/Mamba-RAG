"""
src/data/processed_loader.py
=============================
Thin loader that reads the pre-computed JSON files produced by
preprocess_local.py.  Use this on Kaggle where raw datasets are not
available — just attach the revrag-processed dataset.

Usage:
    from src.data.processed_loader import ProcessedDataLoader
    loader = ProcessedDataLoader(processed_dir=Path("/kaggle/working/data/processed"))
    train_trips = loader.contractnli_triplets("train")
    ids, texts  = loader.contractnli_passages()
    cuad_items  = loader.cuad_queries("test")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {path}\n"
            "Run preprocess_local.py locally first, then upload revrag_processed.zip."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"  Loaded {path.name}: {len(data)} items")
    return data


class ProcessedDataLoader:
    """
    Loads pre-computed JSON files from revrag_processed/.

    Parameters
    ----------
    processed_dir : path to the revrag_processed/ directory
    """

    def __init__(self, processed_dir: Path | str):
        self.root = Path(processed_dir)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Processed data dir not found: {self.root}\n"
                "Run preprocess_local.py, zip revrag_processed/, upload to Kaggle,\n"
                "then run kaggle_load_processed.py."
            )

    # ── ContractNLI ───────────────────────────────────────────────────────────

    def contractnli_triplets(self, split: str) -> List[Dict]:
        """Returns [{query, positive, hard_neg}] for split in (train, val)."""
        return _load(self.root / "contractnli" / f"{split}_triplets.json")

    def contractnli_queries(self, split: str = "test") -> List[Dict]:
        """Returns [{query_id, query, label, doc_id, gold_texts}]."""
        return _load(self.root / "contractnli" / f"{split}_queries.json")

    def contractnli_passages(self) -> Tuple[List[str], List[str]]:
        """Returns (passage_ids, passage_texts) for index building."""
        data = _load(self.root / "contractnli" / "passages.json")
        ids   = list(data.keys())
        texts = [data[k] for k in ids]
        return ids, texts

    # ── CUAD ─────────────────────────────────────────────────────────────────

    def cuad_triplets(self, split: str) -> List[Dict]:
        """Returns [{query, positive, hard_neg}] for split in (train, val, test)."""
        return _load(self.root / "cuad" / f"{split}_triplets.json")

    def cuad_queries(self, split: str = "test") -> List[Dict]:
        """Returns [{query_id, query, category, gold_text, decoy_text, ...}]."""
        return _load(self.root / "cuad" / f"{split}_queries.json")

    def cuad_passages(self) -> Tuple[List[str], List[str]]:
        """Returns (passage_ids, passage_texts). Each item has gold + decoy passage."""
        data = _load(self.root / "cuad" / "passages.json")
        ids   = list(data.keys())
        texts = [data[k] for k in ids]
        return ids, texts

    # ── MultiHopRAG ──────────────────────────────────────────────────────────

    def multihop_queries(self, split: str = "test") -> List[Dict]:
        """Returns [{query_id, query, gold_ids}]."""
        return _load(self.root / "multihop" / f"{split}_queries.json")

    def multihop_passages(self) -> Tuple[List[str], List[str]]:
        """Returns (passage_ids, passage_texts)."""
        data = _load(self.root / "multihop" / "passages.json")
        ids   = list(data.keys())
        texts = [data[k] for k in ids]
        return ids, texts

    def multihop_qrels(self, split: str = "test") -> Dict[str, set]:
        """Returns {query_id: {gold_passage_id, ...}}."""
        queries = self.multihop_queries(split)
        return {q["query_id"]: set(q["gold_ids"]) for q in queries}

    # ── LoCoV1 ───────────────────────────────────────────────────────────────

    def loco_queries(self) -> List[Dict]:
        """Returns [{query_id, query, gold_ids}]."""
        return _load(self.root / "loco" / "test_queries.json")

    def loco_passages(self) -> Tuple[List[str], List[str]]:
        """Returns (passage_ids, passage_texts)."""
        data = _load(self.root / "loco" / "passages.json")
        ids   = list(data.keys())
        texts = [data[k] for k in ids]
        return ids, texts

    def loco_qrels(self) -> Dict[str, set]:
        """Returns {query_id: {gold_doc_id, ...}}."""
        queries = self.loco_queries()
        return {q["query_id"]: set(q["gold_ids"]) for q in queries}

    # ── Combined train data ───────────────────────────────────────────────────

    def all_train_triplets(self, include_cuad: bool = True) -> List[Dict]:
        """Merge ContractNLI + CUAD training triplets."""
        trips = self.contractnli_triplets("train")
        if include_cuad:
            try:
                trips = trips + self.cuad_triplets("train")
            except FileNotFoundError:
                logger.warning("CUAD train triplets not found — skipping CUAD.")
        return trips

    def all_val_triplets(self, include_cuad: bool = True) -> List[Dict]:
        """Merge ContractNLI + CUAD val triplets."""
        trips = self.contractnli_triplets("val")
        if include_cuad:
            try:
                trips = trips + self.cuad_triplets("val")
            except FileNotFoundError:
                logger.warning("CUAD val triplets not found — skipping CUAD.")
        return trips
