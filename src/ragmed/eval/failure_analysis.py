"""Failure analysis.

The single most useful section of a RAG README, and the one that requires the least
machinery: take the worst-scoring questions and sort them by *why* they failed.

The categories are ordered by where in the pipeline the failure occurred, and they are
mutually exclusive by construction - each item is attributed to the earliest stage
that went wrong, because a generation failure downstream of a retrieval failure is not
a generation failure at all:

``retrieval``   the gold chunk never appeared anywhere in the ranked list. Fixing the
                prompt cannot help; this is an indexing, chunking or recall problem.
``ranking``     the gold chunk was retrieved but ranked below the context cutoff. The
                candidate pool was right and the ordering was wrong - a reranker
                problem, not a retriever problem.
``abstention``  the system refused an answerable question, or answered an unanswerable
                one. The second is the dangerous direction.
``generation``  the gold chunk was in the context and the model still got it wrong.
                This is the only category a prompt change can fix.

Counting these separately is what stops "our RAG needs work" from being the whole
diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragmed.eval.generation_metrics import ItemGenerationScores
from ragmed.eval.retrieval_metrics import ItemRetrievalMetrics
from ragmed.eval.runner import EvalRun, ItemRecord

CATEGORIES = ("retrieval", "ranking", "abstention", "generation", "error")


@dataclass(slots=True)
class Failure:
    qid: str
    question: str
    question_type: str
    category: str
    detail: str
    recall: float | None = None
    faithfulness: float | None = None
    first_hit_rank: int | None = None
    answer_excerpt: str | None = None
    gold_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "question_type": self.question_type,
            "category": self.category,
            "detail": self.detail,
            "recall": self.recall,
            "faithfulness": self.faithfulness,
            "first_hit_rank": self.first_hit_rank,
            "answer_excerpt": self.answer_excerpt,
        }


@dataclass
class FailureReport:
    counts: dict[str, int] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    n_items: int = 0
    n_clean: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_clean": self.n_clean,
            "n_failures": len(self.failures),
            "counts": self.counts,
            "failures": [f.to_dict() for f in self.failures],
        }


def classify(
    record: ItemRecord,
    retrieval: ItemRetrievalMetrics | None,
    generation: ItemGenerationScores | None,
    faithfulness_floor: float = 0.8,
    recall_floor: float = 0.5,
) -> Failure | None:
    """Attribute one item to the earliest stage that failed, or None if it is clean."""
    if record.error:
        return Failure(
            qid=record.qid,
            question=record.question,
            question_type=record.question_type,
            category="error",
            detail=record.error,
            gold_chunk_ids=record.gold_chunk_ids,
        )

    is_answerable = bool(record.gold_chunk_ids)

    # Unanswerable items have no retrieval or ranking stage to fail; the only thing
    # that can go wrong is answering when the system should have refused.
    if not is_answerable:
        if not record.abstained and record.answer is not None:
            return Failure(
                qid=record.qid,
                question=record.question,
                question_type=record.question_type,
                category="abstention",
                detail="answered a question the corpus cannot support (hallucination risk)",
                answer_excerpt=(record.answer or "")[:300],
            )
        return None

    recall = retrieval.recall_at_k.get(10) if retrieval else None

    if not record.gold_in_ranked:
        return Failure(
            qid=record.qid,
            question=record.question,
            question_type=record.question_type,
            category="retrieval",
            detail="no gold chunk appeared anywhere in the ranked list",
            recall=recall,
            gold_chunk_ids=record.gold_chunk_ids,
        )

    if not record.gold_in_context:
        return Failure(
            qid=record.qid,
            question=record.question,
            question_type=record.question_type,
            category="ranking",
            detail=(
                f"gold chunk was retrieved at rank {retrieval.first_hit_rank} but fell "
                f"below the context cutoff" if retrieval and retrieval.first_hit_rank
                else "gold chunk was retrieved but did not reach the context"
            ),
            recall=recall,
            first_hit_rank=retrieval.first_hit_rank if retrieval else None,
            gold_chunk_ids=record.gold_chunk_ids,
        )

    if record.abstained:
        return Failure(
            qid=record.qid,
            question=record.question,
            question_type=record.question_type,
            category="abstention",
            detail="refused despite the gold chunk being present in the context",
            recall=recall,
            answer_excerpt=(record.answer or "")[:300],
        )

    if generation is not None and generation.faithfulness is not None:
        if generation.faithfulness < faithfulness_floor:
            return Failure(
                qid=record.qid,
                question=record.question,
                question_type=record.question_type,
                category="generation",
                detail=(
                    f"{generation.n_unsupported}/{generation.n_claims} claims unsupported "
                    f"despite correct context"
                ),
                recall=recall,
                faithfulness=generation.faithfulness,
                answer_excerpt=(record.answer or "")[:300],
            )

    # Retrieval succeeded but only barely; worth surfacing as a near-miss.
    if recall is not None and recall < recall_floor:
        return Failure(
            qid=record.qid,
            question=record.question,
            question_type=record.question_type,
            category="retrieval",
            detail=f"partial recall ({recall:.2f}): some gold chunks were never retrieved",
            recall=recall,
            first_hit_rank=retrieval.first_hit_rank if retrieval else None,
            gold_chunk_ids=record.gold_chunk_ids,
        )

    return None


def analyse(run: EvalRun, top_n: int = 20) -> FailureReport:
    by_qid_retrieval = {m.qid: m for m in run.retrieval.per_item}
    by_qid_generation = (
        {g.qid: g for g in run.generation.per_item} if run.generation else {}
    )

    failures: list[Failure] = []
    for record in run.records:
        f = classify(
            record,
            by_qid_retrieval.get(record.qid),
            by_qid_generation.get(record.qid),
        )
        if f is not None:
            failures.append(f)

    counts = {c: 0 for c in CATEGORIES}
    for f in failures:
        counts[f.category] = counts.get(f.category, 0) + 1

    # Worst first: lowest recall, then lowest faithfulness. Errors sort to the top.
    def sort_key(f: Failure) -> tuple[int, float, float]:
        return (
            0 if f.category == "error" else 1,
            f.recall if f.recall is not None else 0.0,
            f.faithfulness if f.faithfulness is not None else 0.0,
        )

    failures.sort(key=sort_key)

    return FailureReport(
        counts=counts,
        failures=failures[:top_n],
        n_items=len(run.records),
        n_clean=len(run.records) - len(failures),
    )


def render_failure_table(report: FailureReport) -> str:
    total = report.n_items or 1
    lines = ["| Failure mode | n | % of set | What it means |", "|---|---|---|---|"]
    meaning = {
        "retrieval": "gold chunk never retrieved — chunking/recall problem",
        "ranking": "retrieved but ranked too low — reranker problem",
        "abstention": "refused when it shouldn't, or answered when it shouldn't",
        "generation": "correct context, wrong answer — prompt/model problem",
        "error": "the request failed outright",
    }
    for cat in CATEGORIES:
        n = report.counts.get(cat, 0)
        if n:
            lines.append(f"| {cat} | {n} | {100 * n / total:.1f}% | {meaning[cat]} |")
    lines.append(
        f"| **clean** | {report.n_clean} | {100 * report.n_clean / total:.1f}% | — |"
    )
    return "\n".join(lines)
