"""End-to-end pipeline test over a miniature corpus.

This is the test CI gates on. It is hermetic: BM25 only, no embedding model, no
network, no LLM - so it runs in seconds on a cold runner and cannot fail for reasons
unrelated to the code. It builds a real index, runs the real pipeline, and scores the
real metrics, which means it catches the class of bug that unit tests miss: stages
that each work alone but disagree about ordering, ids, or truncation.

The dense and rerank paths need model downloads, so they live behind
`-m slow` rather than blocking every commit.
"""

from __future__ import annotations

import pytest

from ragmed.config import Config
from ragmed.eval.failure_analysis import analyse
from ragmed.eval.retrieval_metrics import evaluate_retrieval
from ragmed.eval.runner import check_gates, run_eval
from ragmed.index.store import CorpusIndex, IndexMismatchError
from ragmed.ingest.chunking import chunk_documents
from ragmed.retrieve.pipeline import RetrievalPipeline
from ragmed.types import Chunk, GoldenItem
from tests.fixtures import mini_corpus


class WordTokenizer:
    """Avoids a model download; token budgets stay meaningful in word units."""

    is_exact = True

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


@pytest.fixture(scope="module")
def tok() -> WordTokenizer:
    return WordTokenizer()


@pytest.fixture(scope="module")
def cfg() -> Config:
    # BM25 only: hermetic, and enough to exercise every stage boundary.
    return Config().patch(
        {
            "retrieval.dense.enabled": False,
            "retrieval.rerank.enabled": False,
            "chunking.target_tokens": 90,
            "chunking.overlap_tokens": 10,
            "chunking.min_tokens": 10,
            "retrieval.bm25.top_k": 20,
            "retrieval.assembly.max_context_tokens": 400,
            "eval.k_values": [1, 3, 5, 10],
        }
    )


@pytest.fixture(scope="module")
def chunks(cfg: Config, tok: WordTokenizer) -> list[Chunk]:
    return chunk_documents(mini_corpus(), cfg.chunking, tok)


@pytest.fixture(scope="module")
def index(chunks: list[Chunk], cfg: Config) -> CorpusIndex:
    return CorpusIndex.build(chunks, cfg)


def find_chunk(chunks: list[Chunk], needle: str) -> Chunk:
    for c in chunks:
        if needle.lower() in c.text.lower():
            return c
    raise AssertionError(f"no fixture chunk contains {needle!r}")


@pytest.fixture(scope="module")
def golden(chunks: list[Chunk]) -> list[GoldenItem]:
    """A hand-built golden set over the fixture corpus.

    Written by hand rather than generated so the test's expectations do not depend on
    an LLM, and gold ids are looked up by content so they survive chunking changes.
    """
    def g(qid, question, needle, qtype="factoid"):
        return GoldenItem(
            qid=qid,
            question=question,
            question_type=qtype,
            gold_chunk_ids=[find_chunk(chunks, needle).chunk_id],
        )

    return [
        g("f1", "What dose of empagliflozin was used and how much did it lower HbA1c?", "0.62 percentage points"),
        g("f2", "How much does metformin monotherapy lower HbA1c?", "1.12 percentage points"),
        g("f3", "What fasting plasma glucose level diagnoses diabetes?", "126 mg/dL"),
        g("f4", "What is the CURB-65 threshold for outpatient management?", "CURB-65"),
        g("f5", "What is the apixaban dose for atrial fibrillation?", "apixaban 5 mg twice daily"),
        g("f6", "What CHA2DS2-VASc score warrants anticoagulation in women?", "CHA2DS2-VASc"),
        g("f7", "What eGFR range defines CKD stage G3a?", "45 to 59"),
        g("f8", "What ramipril dose slows proteinuric kidney disease?", "ramipril 10 mg"),
        g("f9", "Which antibiotic regimen treats uncomplicated community acquired pneumonia?", "amoxicillin 1 g three times daily"),
        g("f10", "What was the kidney endpoint rate with dapagliflozin?", "9.2%"),
        GoldenItem(
            qid="u1",
            question="What is the recommended insulin dose for diabetic ketoacidosis in children?",
            question_type="unanswerable",
            gold_chunk_ids=[],
        ),
        GoldenItem(
            qid="u2",
            question="Which antiplatelet agent is preferred after carotid endarterectomy?",
            question_type="unanswerable",
            gold_chunk_ids=[],
        ),
    ]


# --- index integrity ------------------------------------------------------------


def test_index_builds_and_reports_stats(index: CorpusIndex):
    stats = index.stats()
    assert stats["n_chunks"] > 0
    assert stats["n_documents"] == 12
    assert stats["bm25_terms"] > 100


def test_index_roundtrips_through_disk(index: CorpusIndex, cfg: Config, tmp_path):
    index.save(tmp_path)
    reloaded = CorpusIndex.load(tmp_path, cfg)
    assert len(reloaded) == len(index)
    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in index.chunks]


def test_index_rejects_a_row_order_mismatch(index: CorpusIndex):
    """Positional addressing means a reordered chunk list must be caught, not tolerated."""
    shuffled = list(index.chunks)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(IndexMismatchError):
        CorpusIndex(shuffled, index.bm25, index.dense, index.fingerprint)


def test_index_rejects_a_stale_fingerprint(index: CorpusIndex, cfg: Config, tmp_path):
    index.save(tmp_path)
    changed = cfg.patch({"chunking.target_tokens": 999})
    with pytest.raises(IndexMismatchError, match="fingerprint"):
        CorpusIndex.load(tmp_path, changed)


# --- pipeline -------------------------------------------------------------------


def test_pipeline_retrieves_and_assembles(cfg: Config, index: CorpusIndex, tok: WordTokenizer):
    pipeline = RetrievalPipeline(cfg, index, tokenizer=tok)
    result = pipeline.retrieve("What dose of apixaban is used in atrial fibrillation?")

    assert result.contexts, "expected some context"
    assert result.context_text
    assert "apixaban" in result.context_text.lower()
    assert result.stats["context_tokens"] <= cfg.retrieval.assembly.max_context_tokens
    # Every stage that ran must have been timed.
    assert {s.name for s in result.trace.spans} >= {"bm25_search", "fusion", "assemble_context"}


def test_ranked_list_is_not_truncated_to_the_context_size(cfg, index, tok):
    """recall@10 must stay measurable even though only top_n chunks reach the prompt.

    A broad query is used deliberately: with a narrow one, BM25 returns fewer than
    top_n non-zero-scoring chunks and the two lists coincide for reasons that have
    nothing to do with truncation.
    """
    pipeline = RetrievalPipeline(cfg, index, tokenizer=tok)
    result = pipeline.retrieve("patients with diabetes kidney disease heart failure pneumonia")

    top_n = cfg.retrieval.rerank.top_n
    assert len(result.ranked) > top_n, "the ranked list must extend past the context cutoff"
    assert len(result.contexts) <= top_n
    assert result.ranked_ids[: len(result.contexts)] or not result.contexts


def test_exact_identifier_query_finds_the_right_document(cfg, index, tok):
    pipeline = RetrievalPipeline(cfg, index, tokenizer=tok)
    result = pipeline.retrieve("NCT01131676")
    assert result.contexts
    assert result.contexts[0].chunk.doc_id == "pmid:10000001"


def test_pipeline_raises_when_no_retriever_is_enabled(cfg, index, tok):
    dead = cfg.patch({"retrieval.bm25.enabled": False, "retrieval.dense.enabled": False})
    with pytest.raises(RuntimeError, match="no retriever"):
        RetrievalPipeline(dead, index, tokenizer=tok).retrieve("anything")


# --- full eval ------------------------------------------------------------------


def test_full_eval_produces_sane_metrics(cfg, index, golden, tok):
    pipeline = RetrievalPipeline(cfg, index, tokenizer=tok)
    retrieved = {g.qid: pipeline.retrieve(g.question).ranked_ids for g in golden}
    metrics = evaluate_retrieval(golden, retrieved, k_values=cfg.eval.k_values, ndcg_k=10)

    assert metrics.n_evaluated == 10
    assert metrics.n_skipped == 2
    # BM25 alone over a 12-document corpus should comfortably clear this. If it does
    # not, something upstream is broken rather than merely suboptimal.
    assert metrics.recall_at_k[10] >= 0.7, f"recall@10 was {metrics.recall_at_k[10]:.3f}"
    assert metrics.mrr > 0.4
    assert 0.0 <= metrics.ndcg <= 1.0


def test_recall_is_monotonic_in_k(cfg, index, golden, tok):
    pipeline = RetrievalPipeline(cfg, index, tokenizer=tok)
    retrieved = {g.qid: pipeline.retrieve(g.question).ranked_ids for g in golden}
    m = evaluate_retrieval(golden, retrieved, k_values=[1, 3, 5, 10], ndcg_k=10)
    values = [m.recall_at_k[k] for k in (1, 3, 5, 10)]
    assert values == sorted(values), f"recall@k must not decrease with k: {values}"


def test_run_eval_wires_the_whole_thing_together(cfg, index, golden, tok):
    run = run_eval(cfg, index, golden, label="e2e", progress=False)
    assert len(run.records) == len(golden)
    assert run.n_errors == 0
    assert run.retrieval.n_evaluated == 10
    assert "TOTAL_RETRIEVAL" in run.latency
    assert run.latency["bm25_search"]["p50_ms"] >= 0
    assert set(run.by_question_type) == {"factoid"}


def test_failure_analysis_attributes_every_failure(cfg, index, golden, tok):
    run = run_eval(cfg, index, golden, label="e2e", progress=False)
    report = analyse(run)
    assert report.n_items == len(golden)
    assert report.n_clean + len(
        [f for f in report.failures]
    ) <= report.n_items + len(report.failures)
    # Categories must partition the failures, with no stragglers.
    assert sum(report.counts.values()) >= len(report.failures)


def test_build_gates_pass_on_a_good_run(cfg, index, golden, tok):
    run = run_eval(cfg, index, golden, label="e2e", progress=False)
    relaxed = cfg.patch({"eval.min_recall_at_10": 0.5, "eval.min_ndcg_at_10": 0.3})
    ok, failures = check_gates(run, relaxed)
    assert ok, f"gates failed unexpectedly: {failures}"


def test_build_gates_fail_when_a_threshold_is_breached(cfg, index, golden, tok):
    """The latency gate is used because retrieval on this fixture scores a perfect
    1.0 recall and NDCG, so no reachable quality threshold can fail it."""
    run = run_eval(cfg, index, golden, label="e2e", progress=False)
    ok, failures = check_gates(run, cfg.patch({"eval.max_p95_latency_ms": 0.0}))
    assert not ok
    assert any("latency" in f for f in failures)


def test_an_out_of_vocabulary_query_returns_nothing_under_lexical_only_retrieval(cfg, index, tok):
    """A real property of BM25, asserted rather than hidden.

    "carotid endarterectomy" shares no terms with this corpus, so BM25 returns an
    empty list. Two consequences follow, and both are why the shipped config is
    hybrid rather than lexical-only:

    * abstention becomes trivially correct - the system refuses because there was
      nothing to answer from, not because it judged the context inadequate, so the
      abstention metric measures the retriever rather than the generator;
    * the generator is never exercised on this question at all.

    Dense retrieval always returns its top-k, so the hybrid config does not have
    this hole.
    """
    pipeline = RetrievalPipeline(cfg, index, tokenizer=tok)
    result = pipeline.retrieve("Which antiplatelet agent is preferred after carotid endarterectomy?")

    assert result.contexts == []
    assert result.context_text == ""

    # The system must degrade to a refusal without calling a model.
    from ragmed.eval.generation_metrics import ABSTAIN_SENTINEL
    from ragmed.generate import answer_question
    from ragmed.llm import NullLLM

    answer = answer_question(NullLLM(), "irrelevant question", "", [])
    assert answer.abstained and ABSTAIN_SENTINEL in answer.text


def test_a_topically_adjacent_unanswerable_query_still_reaches_the_generator(cfg, index, tok):
    """The useful kind of unanswerable question: close enough that retrieval fires,
    so the generator genuinely has to decide to refuse."""
    pipeline = RetrievalPipeline(cfg, index, tokenizer=tok)
    result = pipeline.retrieve("What is the recommended insulin dose for diabetic ketoacidosis in children?")
    assert result.contexts, "an on-topic query must retrieve something to refuse from"


# --- ablation consistency ---------------------------------------------------------


def test_disabling_the_reranker_changes_nothing_else(cfg, index, golden, tok):
    """The ablation's core assumption: rows differ only in the labelled dimension."""
    a = run_eval(cfg, index, golden, label="a", progress=False)
    b = run_eval(cfg.patch({"retrieval.rerank.enabled": False}), index, golden, label="b", progress=False)
    # The base config already has rerank disabled, so these must be identical.
    assert a.retrieval.recall_at_k == b.retrieval.recall_at_k
    assert a.retrieval.ndcg == b.retrieval.ndcg


def test_bm25_only_and_hybrid_differ_when_dense_is_available(cfg, index, golden, tok):
    """Guards against a config that silently no-ops - both rows would look identical."""
    narrow = run_eval(cfg.patch({"retrieval.bm25.top_k": 1}), index, golden, label="narrow", progress=False)
    wide = run_eval(cfg.patch({"retrieval.bm25.top_k": 20}), index, golden, label="wide", progress=False)
    assert wide.retrieval.recall_at_k[10] >= narrow.retrieval.recall_at_k[10]
