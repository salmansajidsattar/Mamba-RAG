"""Unit tests for the ConsistencyVerifier module."""

import sys
sys.path.insert(0, ".")

import pytest
import numpy as np

from modules.consistency_verifier import (
    ConsistencyVerifier,
    PassageInfo,
    RerankedResult,
)


def make_passage(pid, text, section_idx, page_approx, rel_score=0.5):
    return PassageInfo(
        doc_id="doc1",
        passage_id=pid,
        text=text,
        relevance_score=rel_score,
        section_idx=section_idx,
        page_approx=page_approx,
    )


@pytest.fixture
def verifier():
    """Create a verifier with NLI model unavailable (stub mode)."""
    v = ConsistencyVerifier(
        checkpoint="cross-encoder/nli-deberta-v3-small",
        nli_contradiction_threshold=0.5,
        page_order_weight=0.3,
    )
    return v


def test_verifier_single_passage(verifier):
    p = make_passage("p0", "Payment is $10,000 per month.", 0, 1, rel_score=0.8)
    results = verifier.rerank([p], query="What is the payment?")
    assert len(results) == 1
    assert results[0].passage_info.passage_id == "p0"
    assert not results[0].consistency_adjusted


def test_verifier_same_page_no_pairs(verifier):
    """Passages from same page → no cross-page pairs → no adjustments."""
    passages = [
        make_passage("p0", "Payment is $10,000.", 0, 1, rel_score=0.9),
        make_passage("p1", "Term is 12 months.", 1, 1, rel_score=0.8),
    ]
    scores = verifier.score_all_pairs(passages)
    assert scores == []  # no cross-page pairs


def test_verifier_cross_page_pairs_generated(verifier):
    """Passages from different pages → cross-page pairs generated."""
    passages = [
        make_passage("p0", "Payment is $10,000 per month.", 0, 1, rel_score=0.9),
        make_passage("p1", "Payment is hereby revised to $15,000.", 5, 2, rel_score=0.7),
    ]
    scores = verifier.score_all_pairs(passages)
    assert len(scores) == 1
    assert scores[0].passage_a == "p0"
    assert scores[0].passage_b == "p1"


def test_verifier_later_passage_identified(verifier):
    passages = [
        make_passage("p0", "Payment is $10,000.", 0, 1, rel_score=0.9),
        make_passage("p1", "Revised: payment is $15,000.", 5, 2, rel_score=0.7),
    ]
    scores = verifier.score_all_pairs(passages)
    assert scores[0].later_passage == "p1"  # section 5 > section 0


def test_verifier_rerank_output_length(verifier):
    passages = [
        make_passage(f"p{i}", f"Clause {i} text here.", i, i + 1, rel_score=0.9 - i * 0.1)
        for i in range(5)
    ]
    results = verifier.rerank(passages, "query")
    assert len(results) == 5


def test_verifier_rerank_sorted_by_score(verifier):
    passages = [
        make_passage("p0", "Original clause: $10,000.", 0, 1, rel_score=0.9),
        make_passage("p1", "Amended clause: $15,000.", 5, 2, rel_score=0.6),
    ]
    results = verifier.rerank(passages, "What is the payment?")
    # Results should be sorted descending by final_score
    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ecs_all_same_page(verifier):
    passages = [
        make_passage("p0", "Text A.", 0, 1),
        make_passage("p1", "Text B.", 1, 1),
    ]
    ecs = verifier.compute_evidence_consistency_score(passages)
    assert ecs == pytest.approx(1.0)  # no cross-page pairs → trivially consistent


def test_ecs_range(verifier):
    passages = [
        make_passage("p0", "Payment is $10,000.", 0, 1, rel_score=0.9),
        make_passage("p1", "Payment revised to $15,000.", 5, 2, rel_score=0.7),
        make_passage("p2", "Term is 12 months.", 10, 3, rel_score=0.6),
    ]
    ecs = verifier.compute_evidence_consistency_score(passages)
    assert 0.0 <= ecs <= 1.0
