"""
src/evaluation/metrics.py
=========================
All retrieval metrics needed for the RevRAG paper.

Standard IR metrics:
  Recall@k, Precision@k, NDCG@k, MRR

Legal / cross-page specific:
  CRA@k   — Conflict Resolution Accuracy
  ECS     — Evidence Consistency Score
  DPR↓    — Decoy Promotion Rate
  Char-P, Char-R — character-level overlap (LegalBench-RAG protocol)

Efficiency:
  encoding_time_per_token — for the scaling curve figure
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standard IR metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def recall_at_k(
    gold_ids: Set[str],
    retrieved_ids: List[str],
    k: int,
) -> float:
    """Recall@k: fraction of gold docs retrieved in top-k."""
    if not gold_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in gold_ids)
    return hits / len(gold_ids)


def precision_at_k(
    gold_ids: Set[str],
    retrieved_ids: List[str],
    k: int,
) -> float:
    """Precision@k: fraction of top-k that are relevant."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in gold_ids)
    return hits / min(k, len(retrieved_ids))


def ndcg_at_k(
    gold_ids: Set[str],
    retrieved_ids: List[str],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain @ k."""
    def dcg(ids):
        return sum(
            (1 / math.log2(i + 2)) for i, doc_id in enumerate(ids[:k])
            if doc_id in gold_ids
        )
    actual  = dcg(retrieved_ids)
    ideal   = dcg(list(gold_ids)[:k])   # ideal: gold docs ranked first
    return actual / ideal if ideal > 0 else 0.0


def mrr(
    gold_ids: Set[str],
    retrieved_ids: List[str],
) -> float:
    """Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in gold_ids:
            return 1.0 / i
    return 0.0


def average_precision(
    gold_ids: Set[str],
    retrieved_ids: List[str],
) -> float:
    """Average Precision (for MAP computation)."""
    if not gold_ids:
        return 0.0
    hits, ap = 0, 0.0
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in gold_ids:
            hits += 1
            ap   += hits / i
    return ap / len(gold_ids)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Character-level metrics (LegalBench-RAG / ContractNLI protocol)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def character_precision_recall(
    gold_text:         str,
    retrieved_texts:   List[str],
    overlap_threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Character-level precision and recall.

    A retrieved passage is a hit if it shares ≥ overlap_threshold fraction
    of the gold text's characters.

    Returns (precision, recall). Note: single gold span → recall is 0 or 1.
    """
    if not gold_text or not retrieved_texts:
        return 0.0, 0.0

    gold_chars = set(gold_text)
    tps        = 0

    for ret in retrieved_texts:
        overlap = sum(c in ret for c in gold_text) / max(len(gold_text), 1)
        if overlap >= overlap_threshold:
            tps += 1

    precision = tps / max(len(retrieved_texts), 1)
    recall    = 1.0 if tps > 0 else 0.0
    return precision, recall


def character_recall_at_k(
    gold_spans:        List[Tuple[int, int, str]],   # [(start, end, text)]
    retrieved_texts:   List[str],
    k:                 int,
    overlap_threshold: float = 0.5,
) -> float:
    """
    Multi-span character recall @ k.
    Fraction of gold spans covered by at least one of the top-k retrieved passages.
    """
    if not gold_spans:
        return 0.0
    covered = set()
    for ret in retrieved_texts[:k]:
        for i, (_, _, g_text) in enumerate(gold_spans):
            if i in covered:
                continue
            ov = sum(c in ret for c in g_text) / max(len(g_text), 1)
            if ov >= overlap_threshold:
                covered.add(i)
    return len(covered) / len(gold_spans)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RevRAG-specific metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cra_at_k(
    retrieved_passages: List[Dict],
    k: int = 1,
) -> float:
    """
    Conflict Resolution Accuracy @ k.

    Each passage dict must have:
      is_gold  : bool  — is this the authoritative (later) passage?
      is_decoy : bool  — is this the superseded (earlier) passage?

    CRA@1 = 1 if gold appears in top-k AND no decoy is ranked above gold.
    """
    if not retrieved_passages:
        return 0.0
    top_k = retrieved_passages[:k]
    gold_pos  = next((i for i, p in enumerate(top_k) if p.get("is_gold")),  None)
    decoy_pos = next((i for i, p in enumerate(top_k) if p.get("is_decoy")), None)

    if gold_pos is None:
        return 0.0   # gold not in top-k
    if decoy_pos is not None and decoy_pos < gold_pos:
        return 0.0   # decoy ranked above gold
    return 1.0


def decoy_promotion_rate(
    results_per_query: List[List[Dict]],
    k: int = 1,
) -> float:
    """
    DPR↓: fraction of queries where decoy is in top-k but gold is not.
    Lower is better.
    """
    if not results_per_query:
        return 0.0
    promoted = 0
    for passages in results_per_query:
        top_k     = passages[:k]
        gold_in   = any(p.get("is_gold")  for p in top_k)
        decoy_in  = any(p.get("is_decoy") for p in top_k)
        if decoy_in and not gold_in:
            promoted += 1
    return promoted / len(results_per_query)


def evidence_consistency_score(
    nli_scores_per_query: List[float],
) -> float:
    """
    ECS: mean fraction of non-contradicting passage pairs across queries.
    Passed as a list of per-query ECS values (computed by CrossPageVerifier).
    """
    if not nli_scores_per_query:
        return 0.0
    return sum(nli_scores_per_query) / len(nli_scores_per_query)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Aggregate: compute all metrics from run results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_all_metrics(
    run_results: Dict[str, List[str]],     # {query_id: [doc_id, ...]}
    qrels:       Dict[str, Set[str]],      # {query_id: {gold_doc_id}}
    k_values:    List[int] = (1, 5, 20),
) -> Dict[str, float]:
    """
    Compute Recall@k, Precision@k, NDCG@k, MRR, MAP for all queries.

    Parameters
    ----------
    run_results : {query_id: ranked list of retrieved doc IDs}
    qrels       : {query_id: set of gold doc IDs}
    k_values    : list of k values to evaluate at

    Returns
    -------
    dict of metric_name → mean value across all queries
    """
    per_query: Dict[str, List[float]] = defaultdict(list)

    for qid, ranked in run_results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue

        for k in k_values:
            per_query[f"recall@{k}"].append(recall_at_k(gold, ranked, k))
            per_query[f"precision@{k}"].append(precision_at_k(gold, ranked, k))
            per_query[f"ndcg@{k}"].append(ndcg_at_k(gold, ranked, k))

        per_query["mrr"].append(mrr(gold, ranked))
        per_query["map"].append(average_precision(gold, ranked))

    return {
        metric: sum(vals) / len(vals)
        for metric, vals in per_query.items()
        if vals
    }
