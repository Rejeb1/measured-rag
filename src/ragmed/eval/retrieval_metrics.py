"""Retrieval metrics: cheap, deterministic, and free of any LLM.

This is the half of the eval layer that answers "was the right chunk in the context at
all". It runs in milliseconds, costs nothing, and gives the same answer every time -
which is why it, not the judge, is the build gate in CI.

Keeping it strictly separate from generation metrics is the whole point of the
project. "recall@10 is 0.91 but faithfulness is 0.68" is a diagnosis: the retriever
is fine and the generator is ignoring its context. A single blended score cannot say
that, and a system that only reports a blended score cannot be debugged.

Relevance is binary here. A chunk is either one of the sources the question was
written from or it is not - there are no graded judgements to average, and inventing
a graded scale over labels that were collected as binary would add precision the data
does not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ragmed.types import GoldenItem


@dataclass(slots=True)
class ItemRetrievalMetrics:
    qid: str
    question_type: str
    recall_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0
    first_hit_rank: int | None = None
    n_gold: int = 0


@dataclass(slots=True)
class RetrievalMetrics:
    recall_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0
    ndcg_k: int = 10
    n_evaluated: int = 0
    n_skipped: int = 0
    per_item: list[ItemRetrievalMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "hit_rate_at_k": {str(k): round(v, 4) for k, v in self.hit_rate_at_k.items()},
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "mrr": round(self.mrr, 4),
            f"ndcg_at_{self.ndcg_k}": round(self.ndcg, 4),
            "n_evaluated": self.n_evaluated,
            "n_skipped_unanswerable": self.n_skipped,
        }


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def hit_rate_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    # Denominator is len(top), not k: at k=10 with only 3 results retrieved,
    # dividing by 10 would report a precision penalty for results that were never
    # returned, conflating precision with recall.
    return len(set(top) & gold) / len(top)


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> tuple[float, int | None]:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank, rank
    return 0.0, None


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Binary-relevance NDCG.

    The ideal ranking places every gold chunk in the top positions, so IDCG is capped
    at ``min(len(gold), k)``. Without that cap, a question with 3 gold chunks
    evaluated at k=10 could never reach 1.0 even with a perfect ranking.
    """
    if not gold:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_item(
    item: GoldenItem,
    retrieved: list[str],
    k_values: list[int],
    ndcg_k: int = 10,
) -> ItemRetrievalMetrics:
    gold = set(item.gold_chunk_ids)
    rr, first = reciprocal_rank(retrieved, gold)
    return ItemRetrievalMetrics(
        qid=item.qid,
        question_type=item.question_type,
        recall_at_k={k: recall_at_k(retrieved, gold, k) for k in k_values},
        hit_rate_at_k={k: hit_rate_at_k(retrieved, gold, k) for k in k_values},
        precision_at_k={k: precision_at_k(retrieved, gold, k) for k in k_values},
        mrr=rr,
        ndcg=ndcg_at_k(retrieved, gold, ndcg_k),
        first_hit_rank=first,
        n_gold=len(gold),
    )


def evaluate_retrieval(
    items: list[GoldenItem],
    retrieved_by_qid: dict[str, list[str]],
    k_values: list[int] | None = None,
    ndcg_k: int = 10,
) -> RetrievalMetrics:
    """Macro-average over answerable questions.

    Unanswerable items are *skipped*, not scored zero. There is no chunk to retrieve,
    so any retrieval score for them is meaningless - counting them as zeros would drag
    every metric down in proportion to how many unanswerable questions the golden set
    happens to contain, making sets of different composition incomparable. Whether the
    system correctly refuses them is measured in generation metrics instead.
    """
    k_values = k_values or [1, 3, 5, 10, 20]
    per_item: list[ItemRetrievalMetrics] = []
    skipped = 0

    for item in items:
        if not item.is_answerable or not item.gold_chunk_ids:
            skipped += 1
            continue
        # A question with no retrieval result recorded scores zero rather than being
        # dropped - a crashed or empty query is a failure, not an absence of data.
        per_item.append(score_item(item, retrieved_by_qid.get(item.qid, []), k_values, ndcg_k))

    n = len(per_item)
    if n == 0:
        return RetrievalMetrics(ndcg_k=ndcg_k, n_evaluated=0, n_skipped=skipped)

    return RetrievalMetrics(
        recall_at_k={k: sum(m.recall_at_k[k] for m in per_item) / n for k in k_values},
        hit_rate_at_k={k: sum(m.hit_rate_at_k[k] for m in per_item) / n for k in k_values},
        precision_at_k={k: sum(m.precision_at_k[k] for m in per_item) / n for k in k_values},
        mrr=sum(m.mrr for m in per_item) / n,
        ndcg=sum(m.ndcg for m in per_item) / n,
        ndcg_k=ndcg_k,
        n_evaluated=n,
        n_skipped=skipped,
        per_item=per_item,
    )


def breakdown_by_type(metrics: RetrievalMetrics, k: int = 10) -> dict[str, dict[str, float]]:
    """Per-question-type recall.

    Aggregate recall hides the interesting result. Multi-hop questions need every gold
    chunk and typically score far below factoids; reporting only the mean lets a
    system look uniformly decent while being unusable for the hard half of the set.
    """
    groups: dict[str, list[ItemRetrievalMetrics]] = {}
    for m in metrics.per_item:
        groups.setdefault(m.question_type, []).append(m)

    out: dict[str, dict[str, float]] = {}
    for qtype, rows in sorted(groups.items()):
        n = len(rows)
        out[qtype] = {
            "n": n,
            f"recall@{k}": round(sum(r.recall_at_k.get(k, 0.0) for r in rows) / n, 4),
            f"hit_rate@{k}": round(sum(r.hit_rate_at_k.get(k, 0.0) for r in rows) / n, 4),
            "mrr": round(sum(r.mrr for r in rows) / n, 4),
            "ndcg": round(sum(r.ndcg for r in rows) / n, 4),
        }
    return out
