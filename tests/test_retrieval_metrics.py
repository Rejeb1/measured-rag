from __future__ import annotations

import math

import pytest

from ragmed.eval.retrieval_metrics import (
    breakdown_by_type,
    evaluate_retrieval,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragmed.types import GoldenItem


def item(qid: str, gold: list[str], qtype: str = "factoid") -> GoldenItem:
    return GoldenItem(qid=qid, question="q?", question_type=qtype, gold_chunk_ids=gold)


# --- recall / hit rate / precision ---------------------------------------------


def test_recall_is_fraction_of_gold_found():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, 3) == pytest.approx(0.5)
    assert recall_at_k(["a", "d", "c"], {"a", "d"}, 3) == pytest.approx(1.0)


def test_recall_respects_the_cutoff():
    assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert recall_at_k(["x", "y", "a"], {"a"}, 3) == 1.0


def test_hit_rate_is_binary():
    assert hit_rate_at_k(["a", "b"], {"a", "z"}, 2) == 1.0
    assert hit_rate_at_k(["b", "c"], {"a"}, 2) == 0.0


def test_precision_divides_by_results_returned_not_by_k():
    """At k=10 with 2 results, precision must not be penalised for the 8 absent slots."""
    assert precision_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
    assert precision_at_k(["a", "z"], {"a"}, 10) == pytest.approx(0.5)


def test_precision_of_empty_results_is_zero_not_undefined():
    assert precision_at_k([], {"a"}, 10) == 0.0


# --- MRR ------------------------------------------------------------------------


def test_reciprocal_rank_uses_the_first_hit():
    rr, rank = reciprocal_rank(["x", "y", "a", "b"], {"a", "b"})
    assert rr == pytest.approx(1 / 3)
    assert rank == 3


def test_reciprocal_rank_is_zero_when_nothing_is_found():
    rr, rank = reciprocal_rank(["x", "y"], {"a"})
    assert rr == 0.0 and rank is None


# --- NDCG -----------------------------------------------------------------------


def test_perfect_ranking_scores_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_ndcg_ideal_is_capped_at_k():
    """3 gold chunks evaluated at k=2 must still be able to reach 1.0."""
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, 2) == pytest.approx(1.0)


def test_ndcg_rewards_higher_positions():
    early = ndcg_at_k(["a", "x", "y"], {"a"}, 10)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, 10)
    assert early > late
    assert early == pytest.approx(1.0)
    assert late == pytest.approx((1 / math.log2(4)) / 1.0)


def test_ndcg_is_zero_with_no_hits():
    assert ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_ndcg_with_no_gold_is_zero_not_nan():
    assert ndcg_at_k(["a"], set(), 10) == 0.0


# --- aggregation ----------------------------------------------------------------


def test_unanswerable_items_are_skipped_not_scored_zero():
    """The composition of the golden set must not move the retrieval numbers."""
    items = [
        item("q1", ["a"]),
        GoldenItem(qid="q2", question="q?", question_type="unanswerable", gold_chunk_ids=[]),
    ]
    m = evaluate_retrieval(items, {"q1": ["a"], "q2": ["whatever"]}, k_values=[10])
    assert m.n_evaluated == 1
    assert m.n_skipped == 1
    assert m.recall_at_k[10] == pytest.approx(1.0)


def test_adding_unanswerable_items_does_not_change_retrieval_scores():
    answerable = [item("q1", ["a"]), item("q2", ["b"])]
    retrieved = {"q1": ["a"], "q2": ["x"]}
    base = evaluate_retrieval(answerable, retrieved, k_values=[10])

    padded = answerable + [
        GoldenItem(qid=f"u{i}", question="q?", question_type="unanswerable", gold_chunk_ids=[])
        for i in range(20)
    ]
    with_unanswerable = evaluate_retrieval(padded, retrieved, k_values=[10])
    assert base.recall_at_k[10] == pytest.approx(with_unanswerable.recall_at_k[10])


def test_a_question_with_no_recorded_results_scores_zero():
    """A crashed query is a failure, not missing data."""
    m = evaluate_retrieval([item("q1", ["a"])], {}, k_values=[10])
    assert m.n_evaluated == 1
    assert m.recall_at_k[10] == 0.0


def test_macro_average_weights_questions_equally():
    items = [item("q1", ["a"]), item("q2", ["b", "c", "d"])]
    m = evaluate_retrieval(items, {"q1": ["a"], "q2": ["b"]}, k_values=[10])
    # q1 recall 1.0, q2 recall 1/3 -> mean 2/3, not weighted by gold count.
    assert m.recall_at_k[10] == pytest.approx((1.0 + 1 / 3) / 2)


def test_empty_golden_set_is_safe():
    m = evaluate_retrieval([], {}, k_values=[10])
    assert m.n_evaluated == 0 and m.mrr == 0.0


def test_breakdown_separates_question_types():
    items = [
        item("f1", ["a"], "factoid"),
        item("m1", ["b", "c"], "multi_hop"),
    ]
    m = evaluate_retrieval(items, {"f1": ["a"], "m1": ["b"]}, k_values=[10], ndcg_k=10)
    bd = breakdown_by_type(m, k=10)
    assert bd["factoid"]["recall@10"] == pytest.approx(1.0)
    assert bd["multi_hop"]["recall@10"] == pytest.approx(0.5)
    assert bd["multi_hop"]["hit_rate@10"] == pytest.approx(1.0)


def test_to_dict_is_serialisable():
    m = evaluate_retrieval([item("q1", ["a"])], {"q1": ["a"]}, k_values=[1, 10])
    d = m.to_dict()
    assert d["recall_at_k"]["10"] == 1.0
    assert d["ndcg_at_10"] == 1.0
