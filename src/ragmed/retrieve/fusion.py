"""Rank fusion.

Reciprocal Rank Fusion scores a document as ``sum(1 / (k + rank_i))`` over the lists
it appears in. Because it consumes *ranks*, not scores, it sidesteps the problem that
makes score-based fusion fragile: a BM25 score is an unbounded sum of IDF terms and a
cosine similarity is bounded in [-1, 1], so any attempt to add them directly requires
a normalisation that is itself a tuning parameter, and one whose correct value drifts
with corpus size and query length.

``normalized_sum`` implements exactly that fragile alternative, on purpose - it is the
control that shows what RRF is worth, rather than asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Ranked list: (doc_index, raw_score), best first.
Ranking = list[tuple[int, float]]


@dataclass(slots=True)
class FusedHit:
    doc_index: int
    score: float
    # Per-retriever rank and raw score, kept for failure analysis: "BM25 had it at
    # rank 2, dense never returned it" is a diagnosis; a fused score alone is not.
    components: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: dict[str, Ranking], k: int = 60) -> list[FusedHit]:
    fused: dict[int, float] = {}
    components: dict[int, dict[str, float]] = {}

    for source, ranking in rankings.items():
        for rank, (doc_index, raw) in enumerate(ranking, start=1):
            fused[doc_index] = fused.get(doc_index, 0.0) + 1.0 / (k + rank)
            comp = components.setdefault(doc_index, {})
            comp[f"{source}_rank"] = float(rank)
            comp[f"{source}_score"] = float(raw)

    hits = [FusedHit(doc_index=i, score=s, components=components.get(i, {})) for i, s in fused.items()]
    # Tie-break on doc_index so fusion is deterministic; without it, two configs that
    # differ only in dict iteration order could produce different ablation numbers.
    hits.sort(key=lambda h: (-h.score, h.doc_index))
    for rank, hit in enumerate(hits, start=1):
        hit.components["fused_rank"] = float(rank)
    return hits


def _min_max(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        # Every score identical: normalising would divide by ~0. Treat them as
        # equally relevant rather than emitting NaN.
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def normalized_sum_fusion(rankings: dict[str, Ranking]) -> list[FusedHit]:
    """Min-max normalise each retriever's scores, then sum.

    Note the asymmetry this introduces: normalisation is computed over each list's
    *returned* candidates, so a document's normalised score depends on which other
    documents happened to be retrieved alongside it. That instability is the point of
    including this method as a comparison.
    """
    fused: dict[int, float] = {}
    components: dict[int, dict[str, float]] = {}

    for source, ranking in rankings.items():
        if not ranking:
            continue
        idxs = np.array([i for i, _ in ranking])
        raw = np.array([s for _, s in ranking], dtype=np.float64)
        norm = _min_max(raw)
        for rank, (doc_index, raw_score, norm_score) in enumerate(
            zip(idxs, raw, norm, strict=True), start=1
        ):
            di = int(doc_index)
            fused[di] = fused.get(di, 0.0) + float(norm_score)
            comp = components.setdefault(di, {})
            comp[f"{source}_rank"] = float(rank)
            comp[f"{source}_score"] = float(raw_score)
            comp[f"{source}_norm"] = float(norm_score)

    hits = [FusedHit(doc_index=i, score=s, components=components.get(i, {})) for i, s in fused.items()]
    hits.sort(key=lambda h: (-h.score, h.doc_index))
    for rank, hit in enumerate(hits, start=1):
        hit.components["fused_rank"] = float(rank)
    return hits


def fuse(rankings: dict[str, Ranking], method: str = "rrf", k: int = 60) -> list[FusedHit]:
    """Fuse ranked lists. A single non-empty list passes through with its own order."""
    populated = {name: r for name, r in rankings.items() if r}
    if not populated:
        return []

    if len(populated) == 1:
        # One retriever: fusion is a no-op. Preserving the original order here (rather
        # than running it through RRF, which is monotonic anyway) keeps the raw scores
        # meaningful for the dense-only and BM25-only ablation rows.
        (source, ranking), = populated.items()
        hits = []
        for rank, (doc_index, raw) in enumerate(ranking, start=1):
            hits.append(
                FusedHit(
                    doc_index=doc_index,
                    score=float(raw),
                    components={
                        f"{source}_rank": float(rank),
                        f"{source}_score": float(raw),
                        "fused_rank": float(rank),
                    },
                )
            )
        return hits

    if method == "rrf":
        return reciprocal_rank_fusion(populated, k=k)
    if method == "normalized_sum":
        return normalized_sum_fusion(populated)
    raise ValueError(f"unknown fusion method {method!r}")
