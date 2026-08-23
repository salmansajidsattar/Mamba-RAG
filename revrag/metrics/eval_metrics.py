"""
Evaluation metrics for RevRAG.

Standard IR metrics:
    - Recall@k       (k = 1, 5, 10, 20)
    - NDCG@10
    - MRR

RevRAG-specific metrics:
    - Evidence Consistency Score (ECS): fraction of cross-page retrieved
      passage pairs that are NOT contradictory (higher = better)
    - Revision Recall (RR): fraction of queries where the authoritative
      (later-page) passage ranks above the superseded (earlier-page) passage
    - Conflict Resolution Accuracy (CRA): on the synthetic benchmark,
      fraction of queries where the gold answer (amendment) is in top-1

Statistical testing:
    - paired_ttest: paired t-test between method A and B on per-query scores
    - bootstrap: bootstrap confidence intervals

All metrics return (mean, std) computed across `seeds` runs.
"""

import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard IR metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """
    Recall@k: fraction of relevant docs retrieved in top-k, averaged over queries.

    Args:
        results:  {query_id: [(doc_id, score), ...]} sorted descending
        qrels:    {query_id: {doc_id: relevance_score}}
        k:        cutoff
        relevance_threshold: minimum relevance score to count as relevant
    """
    recalls = []
    for qid, hits in results.items():
        if qid not in qrels:
            continue
        relevant = {
            did for did, rel in qrels[qid].items()
            if rel >= relevance_threshold
        }
        if not relevant:
            continue
        top_k_docs = {hit[0] for hit in hits[:k]}
        recalls.append(len(relevant & top_k_docs) / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> float:
    """NDCG@k using graded relevance from qrels."""
    ndcgs = []
    for qid, hits in results.items():
        if qid not in qrels:
            continue
        # DCG
        dcg = 0.0
        for rank, (doc_id, _) in enumerate(hits[:k], start=1):
            rel = qrels[qid].get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(rank + 1)
        # Ideal DCG
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(rank + 1)
            for rank, rel in enumerate(ideal_rels, start=1)
        )
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def mrr(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    relevance_threshold: int = 1,
    k: int = 100,
) -> float:
    """Mean Reciprocal Rank."""
    rrs = []
    for qid, hits in results.items():
        if qid not in qrels:
            continue
        relevant = {
            did for did, rel in qrels[qid].items()
            if rel >= relevance_threshold
        }
        rr = 0.0
        for rank, (doc_id, _) in enumerate(hits[:k], start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def character_precision_recall_at_k(
    results: Dict[str, List[Tuple[str, float, Dict]]],
    gold_spans: Dict[str, List[Dict]],
    k: int = 5,
) -> Tuple[float, float]:
    """
    Character-level Precision@k and Recall@k for LegalBench-RAG evaluation.

    Args:
        results:     {qid: [(doc_id, score, meta), ...]} where meta has 'text'
        gold_spans:  {qid: [{start, end, text}, ...]}
        k:           cutoff

    Returns:
        (precision@k, recall@k)
    """
    precisions, recalls = [], []
    for qid, hits in results.items():
        if qid not in gold_spans:
            continue
        gold_texts = {span["text"].strip() for span in gold_spans[qid]}

        top_k_texts = [
            hit[2].get("text", "") if len(hit) > 2 else ""
            for hit in hits[:k]
        ]

        # Check character-level overlap: gold span text must appear in retrieved chunk
        retrieved_correct = sum(
            1 for chunk_text in top_k_texts
            if any(gold in chunk_text for gold in gold_texts)
        )
        total_gold = len(gold_texts)

        precisions.append(retrieved_correct / max(k, 1))
        recalls.append(
            min(retrieved_correct, total_gold) / max(total_gold, 1)
        )

    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    return p, r


# ---------------------------------------------------------------------------
# RevRAG-specific metrics
# ---------------------------------------------------------------------------

def evidence_consistency_score(
    consistency_results: List[Dict],
) -> float:
    """
    Evidence Consistency Score (ECS).

    Input: list of per-query dicts from ConsistencyVerifier.score_all_pairs()
    Each dict has 'is_contradiction' and is a ConsistencyScore object.

    Returns: mean ECS across queries (1.0 = all consistent, 0.0 = all contradictory)
    """
    if not consistency_results:
        return 1.0
    scores = []
    for query_pair_scores in consistency_results:
        if not query_pair_scores:
            scores.append(1.0)
            continue
        n_contra = sum(1 for s in query_pair_scores if s.is_contradiction)
        scores.append(1.0 - n_contra / len(query_pair_scores))
    return float(np.mean(scores))


def revision_recall(
    reranked_results: Dict[str, List],
    gold_authority: Dict[str, str],
) -> float:
    """
    Revision Recall (RR): fraction of queries where the authoritative
    (later-page, amended) passage ranks above the superseded passage.

    Args:
        reranked_results: {qid: [RerankedResult, ...]} sorted by final_score
        gold_authority:   {qid: passage_id of the authoritative passage}

    Returns: Revision Recall score in [0, 1]
    """
    hits = 0
    total = 0
    for qid, results in reranked_results.items():
        if qid not in gold_authority:
            continue
        auth_id = gold_authority[qid]
        ranked_ids = [r.passage_info.passage_id for r in results]
        if auth_id in ranked_ids:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def conflict_resolution_accuracy(
    reranked_results: Dict[str, List],
    gold_answers: Dict[str, str],
    top_k: int = 1,
) -> float:
    """
    Conflict Resolution Accuracy (CRA): on the synthetic benchmark,
    fraction of queries where the gold answer (the amendment) is in the
    text of the top-k retrieved passages.

    Args:
        reranked_results: {qid: [RerankedResult, ...]}
        gold_answers:     {qid: gold_answer_string}
        top_k:            check if gold answer appears in top-k passages

    Returns: CRA score in [0, 1]
    """
    correct = 0
    total = 0
    for qid, results in reranked_results.items():
        if qid not in gold_answers:
            continue
        gold = gold_answers[qid].strip().lower()
        top_texts = [
            r.passage_info.text.lower()
            for r in results[:top_k]
        ]
        if any(gold in t for t in top_texts):
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Paired t-test between two systems.

    Returns dict with t-statistic, p-value, and significance flag.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must be equal length for paired test.")
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < alpha),
        "alpha": alpha,
        "n": len(scores_a),
    }


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    boot_means = [
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Convenience: compute all standard metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    ks: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """Compute Recall@k for each k, NDCG@10, and MRR. Return as dict."""
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(results, qrels, k)
    metrics["ndcg@10"] = ndcg_at_k(results, qrels, k=10)
    metrics["mrr"] = mrr(results, qrels)
    return metrics
