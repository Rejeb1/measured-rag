"""Run one configuration over the golden set and collect everything measurable.

A single pass produces retrieval metrics, generation metrics, per-stage latency and a
full per-item record. The per-item records are what make failure analysis possible
afterwards without re-running anything - which matters, because on CPU a generation
pass over 180 questions is not something you want to repeat to answer a follow-up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ragmed.config import Config
from ragmed.eval.generation_metrics import (
    GenerationMetrics,
    ItemGenerationScores,
    aggregate_generation,
    score_generation,
)
from ragmed.eval.retrieval_metrics import (
    RetrievalMetrics,
    breakdown_by_type,
    evaluate_retrieval,
)
from ragmed.generate import answer_question
from ragmed.index.dense import Embedder
from ragmed.index.store import CorpusIndex
from ragmed.llm import LLM, LLMError, NullLLM
from ragmed.retrieve.pipeline import RetrievalPipeline
from ragmed.retrieve.rerank import CrossEncoderReranker
from ragmed.telemetry import Trace, aggregate_latency
from ragmed.types import GoldenItem

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ItemRecord:
    """Everything needed to explain one question's outcome after the fact."""

    qid: str
    question: str
    question_type: str
    gold_chunk_ids: list[str]
    ranked_ids: list[str]
    context_ids: list[str]
    reference_answer: str | None = None
    answer: str | None = None
    abstained: bool = False
    citations: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def gold_in_context(self) -> bool:
        return bool(set(self.gold_chunk_ids) & set(self.context_ids))

    @property
    def gold_in_ranked(self) -> bool:
        return bool(set(self.gold_chunk_ids) & set(self.ranked_ids))


@dataclass
class EvalRun:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics | None = None
    latency: dict[str, dict[str, float]] = field(default_factory=dict)
    by_question_type: dict[str, dict[str, float]] = field(default_factory=dict)
    records: list[ItemRecord] = field(default_factory=list)
    n_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "overrides": self.overrides,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict() if self.generation else None,
            "latency": self.latency,
            "by_question_type": self.by_question_type,
            "n_items": len(self.records),
            "n_errors": self.n_errors,
        }


def run_eval(
    cfg: Config,
    index: CorpusIndex,
    golden: list[GoldenItem],
    label: str = "default",
    overrides: dict[str, Any] | None = None,
    with_generation: bool = False,
    llm: LLM | None = None,
    judge: LLM | None = None,
    embedder: Embedder | None = None,
    reranker: CrossEncoderReranker | None = None,
    progress: bool = True,
) -> EvalRun:
    llm = llm or NullLLM()
    judge = judge or NullLLM()

    # A shared embedder is only valid while the embedding model is unchanged; an
    # ablation row that swaps models must not silently reuse the previous encoder.
    if embedder is not None and embedder.cfg.model != cfg.retrieval.dense.model:
        log.info("embedding model changed for %r; building a fresh encoder", label)
        embedder = None
    if reranker is not None and reranker.cfg.model != cfg.retrieval.rerank.model:
        reranker = None

    pipeline = RetrievalPipeline(cfg, index, embedder=embedder, reranker=reranker, llm=llm)

    if with_generation and not llm.available():
        log.warning(
            "generation metrics requested but no LLM is available; "
            "running retrieval-only for %r", label
        )
        with_generation = False
    if with_generation and not judge.available():
        log.warning("no judge available; generation will run but will not be scored")

    items = list(golden)
    iterator = items
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(items, desc=f"eval[{label}]", unit="q")
        except ImportError:
            pass

    records: list[ItemRecord] = []
    traces: list[Trace] = []
    gen_scores: list[ItemGenerationScores] = []
    n_errors = 0

    for item in iterator:
        record = ItemRecord(
            qid=item.qid,
            question=item.question,
            question_type=item.question_type,
            gold_chunk_ids=list(item.gold_chunk_ids),
            ranked_ids=[],
            context_ids=[],
            reference_answer=item.answer,
        )
        try:
            result = pipeline.retrieve(item.question)
        except Exception as exc:  # noqa: BLE001
            # One bad question must not abort a 20-minute ablation sweep. It is
            # recorded as an error and scores zero on retrieval.
            log.exception("retrieval failed for %s", item.qid)
            record.error = f"retrieval: {exc}"
            records.append(record)
            n_errors += 1
            continue

        record.ranked_ids = result.ranked_ids
        record.context_ids = result.context_ids
        record.trace = result.trace.to_dict()
        traces.append(result.trace)

        if with_generation:
            try:
                with result.trace.stage("generate"):
                    answer = answer_question(
                        llm, item.question, result.context_text, result.contexts, cfg.generation
                    )
                record.answer = answer.text
                record.abstained = answer.abstained
                record.citations = answer.citations
            except LLMError as exc:
                log.warning("generation failed for %s: %s", item.qid, exc)
                record.error = f"generation: {exc}"
                n_errors += 1

            if judge.available() and record.answer is not None:
                gen_scores.append(
                    score_generation(
                        judge,
                        qid=item.qid,
                        question_type=item.question_type,
                        question=item.question,
                        answer=record.answer,
                        context=result.context_text,
                        chunks=[s.chunk for s in result.contexts],
                        is_answerable=item.is_answerable,
                    )
                )

        records.append(record)

    retrieval = evaluate_retrieval(
        items,
        {r.qid: r.ranked_ids for r in records},
        k_values=cfg.eval.k_values,
        ndcg_k=cfg.eval.ndcg_k,
    )

    return EvalRun(
        label=label,
        overrides=overrides or {},
        retrieval=retrieval,
        generation=aggregate_generation(gen_scores) if gen_scores else None,
        latency=aggregate_latency(traces),
        by_question_type=breakdown_by_type(retrieval, k=cfg.eval.ndcg_k),
        records=records,
        n_errors=n_errors,
    )


def check_gates(run: EvalRun, cfg: Config) -> tuple[bool, list[str]]:
    """Compare a run against the configured build gates.

    This is what CI calls. A prompt tweak or a config change that quietly drops
    recall should fail the build rather than merge and be discovered later.
    """
    failures: list[str] = []
    e = cfg.eval

    recall = run.retrieval.recall_at_k.get(10)
    if recall is not None and recall < e.min_recall_at_10:
        failures.append(f"recall@10 {recall:.3f} < {e.min_recall_at_10:.3f}")

    if run.retrieval.ndcg < e.min_ndcg_at_10:
        failures.append(f"ndcg@{run.retrieval.ndcg_k} {run.retrieval.ndcg:.3f} < {e.min_ndcg_at_10:.3f}")

    if run.generation and run.generation.faithfulness is not None:
        if run.generation.faithfulness < e.min_faithfulness:
            failures.append(
                f"faithfulness {run.generation.faithfulness:.3f} < {e.min_faithfulness:.3f}"
            )

    total = run.latency.get("TOTAL_RETRIEVAL", {}).get("p95_ms")
    if total is not None and total > e.max_p95_latency_ms:
        failures.append(f"p95 retrieval latency {total:.0f}ms > {e.max_p95_latency_ms:.0f}ms")

    return (not failures), failures
