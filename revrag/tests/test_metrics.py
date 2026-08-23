"""Unit tests for eval_metrics.py."""

import sys
sys.path.insert(0, ".")

import pytest
import numpy as np

from metrics.eval_metrics import (
    recall_at_k,
    ndcg_at_k,
    mrr,
    evidence_consistency_score,
    conflict_resolution_accuracy,
    paired_ttest,
    bootstrap_ci,
)


@pytest.fixture
def simple_results():
    return {
        "q1": [("d1", 0.9), ("d2", 0.8), ("d3", 0.7), ("d4", 0.6)],
        "q2": [("d5", 0.9), ("d2", 0.8), ("d6", 0.7)],
    }


@pytest.fixture
def simple_qrels():
    return {
        "q1": {"d1": 2, "d3": 1},
        "q2": {"d5": 2, "d6": 1},
    }


def test_recall_at_1(simple_results, simple_qrels):
    r = recall_at_k(simple_results, simple_qrels, k=1)
    # q1: d1 in top-1 → 1/2 relevant; q2: d5 in top-1 → 2/2 relevant
    assert 0 < r <= 1.0


def test_recall_at_k_perfect(simple_results, simple_qrels):
    r = recall_at_k(simple_results, simple_qrels, k=10)
    assert r == pytest.approx(1.0)


def test_recall_at_k_zero():
    results = {"q1": [("d99", 0.9)]}
    qrels = {"q1": {"d1": 1}}
    r = recall_at_k(results, qrels, k=5)
    assert r == 0.0


def test_ndcg_at_k(simple_results, simple_qrels):
    score = ndcg_at_k(simple_results, simple_qrels, k=10)
    assert 0.0 <= score <= 1.0


def test_ndcg_perfect():
    results = {"q1": [("d1", 0.9), ("d2", 0.8)]}
    qrels = {"q1": {"d1": 2, "d2": 1}}
    score = ndcg_at_k(results, qrels, k=2)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_mrr(simple_results, simple_qrels):
    score = mrr(simple_results, simple_qrels)
    assert 0.0 < score <= 1.0


def test_mrr_first_rank():
    results = {"q1": [("d1", 0.9)]}
    qrels = {"q1": {"d1": 1}}
    assert mrr(results, qrels) == pytest.approx(1.0)


def test_evidence_consistency_score_all_consistent():
    # No contradictions → ECS = 1.0
    from unittest.mock import MagicMock
    mock_scores = [MagicMock(is_contradiction=False) for _ in range(5)]
    ecs = evidence_consistency_score([mock_scores])
    assert ecs == pytest.approx(1.0)


def test_evidence_consistency_score_all_contradictory():
    from unittest.mock import MagicMock
    mock_scores = [MagicMock(is_contradiction=True) for _ in range(4)]
    ecs = evidence_consistency_score([mock_scores])
    assert ecs == pytest.approx(0.0)


def test_evidence_consistency_score_empty():
    ecs = evidence_consistency_score([[]])
    assert ecs == pytest.approx(1.0)


def test_conflict_resolution_accuracy():
    from unittest.mock import MagicMock
    results = {
        "q1": [MagicMock(passage_info=MagicMock(text="payment is $15,000"))],
        "q2": [MagicMock(passage_info=MagicMock(text="term is 2 years"))],
    }
    gold = {"q1": "$15,000", "q2": "5 years"}
    cra = conflict_resolution_accuracy(results, gold, top_k=1)
    assert cra == pytest.approx(0.5)


def test_paired_ttest_significant():
    a = [0.5, 0.6, 0.55, 0.65, 0.7]
    b = [0.3, 0.35, 0.3, 0.28, 0.32]
    result = paired_ttest(a, b)
    assert result["significant"] is True
    assert "p_value" in result
    assert "t_statistic" in result


def test_paired_ttest_not_significant():
    a = [0.5, 0.5, 0.5]
    b = [0.5, 0.5, 0.5]
    result = paired_ttest(a, b)
    # p-value undefined for identical arrays; scipy returns nan
    assert "p_value" in result


def test_bootstrap_ci():
    scores = [0.7, 0.72, 0.68, 0.71, 0.73]
    lo, hi = bootstrap_ci(scores)
    assert lo < np.mean(scores) < hi
    assert lo >= 0.0
    assert hi <= 1.0
