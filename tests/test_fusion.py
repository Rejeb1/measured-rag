from __future__ import annotations

import math

import pytest

from ragmed.retrieve.fusion import fuse, normalized_sum_fusion, reciprocal_rank_fusion


def test_rrf_matches_the_formula():
    rankings = {"bm25": [(7, 9.9), (3, 4.4)], "dense": [(3, 0.91), (7, 0.88)]}
    hits = {h.doc_index: h.score for h in reciprocal_rank_fusion(rankings, k=60)}
    # doc 7: rank 1 in bm25, rank 2 in dense. doc 3: the mirror image.
    expected = 1 / 61 + 1 / 62
    assert hits[7] == pytest.approx(expected)
    assert hits[3] == pytest.approx(expected)


def test_agreement_between_retrievers_outranks_a_single_strong_hit():
    """The core claim of hybrid retrieval: two mediocre agreeing votes beat one loud one."""
    rankings = {
        "bm25": [(1, 50.0), (2, 10.0)],
        "dense": [(2, 0.95), (5, 0.94)],
    }
    hits = reciprocal_rank_fusion(rankings, k=60)
    top = hits[0]
    # doc 2 appears in both lists (ranks 2 and 1); doc 1 tops only one.
    assert top.doc_index == 2
    assert top.score > [h for h in hits if h.doc_index == 1][0].score


def test_rank_based_fusion_ignores_score_scale():
    """BM25 scores are unbounded and cosines are not; RRF must not care."""
    small = {"bm25": [(1, 0.001), (2, 0.0005)], "dense": [(2, 0.9), (1, 0.8)]}
    huge = {"bm25": [(1, 10_000.0), (2, 5_000.0)], "dense": [(2, 0.9), (1, 0.8)]}
    assert [h.doc_index for h in reciprocal_rank_fusion(small)] == [
        h.doc_index for h in reciprocal_rank_fusion(huge)
    ]


def test_components_record_each_retrievers_view():
    rankings = {"bm25": [(4, 8.0)], "dense": [(4, 0.7)]}
    hit = reciprocal_rank_fusion(rankings)[0]
    assert hit.components["bm25_rank"] == 1.0
    assert hit.components["bm25_score"] == 8.0
    assert hit.components["dense_score"] == 0.7
    assert hit.components["fused_rank"] == 1.0


def test_ties_break_deterministically():
    """Two configs differing only in dict order must not produce different tables."""
    a = reciprocal_rank_fusion({"x": [(9, 1.0)], "y": [(2, 1.0)]})
    b = reciprocal_rank_fusion({"y": [(2, 1.0)], "x": [(9, 1.0)]})
    assert [h.doc_index for h in a] == [h.doc_index for h in b] == [2, 9]


def test_smaller_k_sharpens_the_top_of_the_ranking():
    rankings = {"bm25": [(1, 1.0), (2, 1.0)], "dense": [(1, 1.0), (2, 1.0)]}
    sharp = reciprocal_rank_fusion(rankings, k=1)
    flat = reciprocal_rank_fusion(rankings, k=1000)
    sharp_gap = sharp[0].score - sharp[1].score
    flat_gap = flat[0].score - flat[1].score
    assert sharp_gap > flat_gap


# --- single-list and empty behaviour ------------------------------------------


def test_single_retriever_passes_through_with_raw_scores():
    """The dense-only ablation row must report dense scores, not RRF scores."""
    hits = fuse({"dense": [(3, 0.91), (1, 0.72)], "bm25": []}, method="rrf")
    assert [h.doc_index for h in hits] == [3, 1]
    assert hits[0].score == pytest.approx(0.91)


def test_empty_rankings_produce_no_hits():
    assert fuse({"bm25": [], "dense": []}) == []
    assert fuse({}) == []


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown fusion method"):
        fuse({"a": [(1, 1.0)], "b": [(2, 1.0)]}, method="magic")


# --- normalized_sum, the control -----------------------------------------------


def test_normalized_sum_does_not_divide_by_zero_on_flat_scores():
    hits = normalized_sum_fusion({"bm25": [(1, 5.0), (2, 5.0)], "dense": [(1, 0.5), (2, 0.5)]})
    assert all(math.isfinite(h.score) for h in hits)


def test_normalized_sum_is_sensitive_to_the_candidate_set():
    """The instability that motivates using RRF instead - asserted, not assumed.

    doc 1 and doc 2 keep identical raw scores in both runs; only an unrelated third
    candidate appears. Under min-max normalisation that is enough to change doc 2's
    contribution, because the scale is derived from whatever else got retrieved.
    """
    without = {h.doc_index: h.score for h in normalized_sum_fusion({"bm25": [(1, 10.0), (2, 8.0)]})}
    with_extra = {
        h.doc_index: h.score
        for h in normalized_sum_fusion({"bm25": [(1, 10.0), (2, 8.0), (3, 0.0)]})
    }
    assert without[2] != pytest.approx(with_extra[2])


def test_rrf_is_stable_under_the_same_perturbation():
    without = {h.doc_index: h.score for h in fuse({"bm25": [(1, 10.0), (2, 8.0)], "dense": [(1, 0.9)]})}
    with_extra = {
        h.doc_index: h.score
        for h in fuse({"bm25": [(1, 10.0), (2, 8.0), (3, 0.0)], "dense": [(1, 0.9)]})
    }
    assert without[2] == pytest.approx(with_extra[2])
