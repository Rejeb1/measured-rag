"""Query rewriting must never silently no-op during an ablation.

This is the most dangerous bug class the project hit: rewriting without a usable LLM
falls back to the original query, which is *correct* at request time but produces an
ablation row identical to the baseline. That row then reads as a clean, publishable
finding - "query rewriting does not help" - when nothing was rewritten at all.

It happened twice, at two different call sites, before being caught. These tests pin
the guard rather than the call sites.
"""

from __future__ import annotations

import logging

import pytest

from ragmed.config import Config
from ragmed.index.store import CorpusIndex
from ragmed.ingest.chunking import chunk_documents
from ragmed.llm import NullLLM
from ragmed.retrieve.pipeline import RetrievalPipeline
from ragmed.retrieve.rewrite import rewrite_query
from tests.fixtures import mini_corpus


class WordTokenizer:
    is_exact = True

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


class StubLLM:
    """Returns fixed rewrites, and records whether it was consulted."""

    name = "stub"

    def __init__(self, available: bool = True):
        self._available = available
        self.calls = 0

    def available(self) -> bool:
        return self._available

    def complete(self, prompt, system=None, **kw):
        raise AssertionError("rewrite should use complete_json")

    def complete_json(self, prompt, system=None, **kw):
        self.calls += 1
        return {"queries": ["formal clinical phrasing", "plain spoken phrasing"]}

    def stream(self, prompt, system=None, **kw):
        yield ""


@pytest.fixture(scope="module")
def tok():
    return WordTokenizer()


@pytest.fixture(scope="module")
def base_cfg():
    return Config().patch(
        {
            "retrieval.dense.enabled": False,
            "retrieval.rerank.enabled": False,
            "chunking.target_tokens": 90,
            "chunking.min_tokens": 10,
            "retrieval.bm25.top_k": 10,
        }
    )


@pytest.fixture(scope="module")
def index(base_cfg, tok):
    return CorpusIndex.build(chunk_documents(mini_corpus(), base_cfg.chunking, tok), base_cfg)


# --- the guard -------------------------------------------------------------------


def test_pipeline_warns_when_rewrite_is_enabled_without_an_llm(base_cfg, index, tok, caplog):
    cfg = base_cfg.patch({"retrieval.rewrite.enabled": True})
    with caplog.at_level(logging.WARNING):
        pipeline = RetrievalPipeline(cfg, index, llm=NullLLM(), tokenizer=tok)

    assert pipeline.rewrite_active is False
    assert any("will NOT be rewritten" in r.message for r in caplog.records), (
        "an unusable rewrite config must warn loudly, not fail silently"
    )


def test_rewrite_active_is_true_when_an_llm_is_present(base_cfg, index, tok):
    cfg = base_cfg.patch({"retrieval.rewrite.enabled": True})
    pipeline = RetrievalPipeline(cfg, index, llm=StubLLM(), tokenizer=tok)
    assert pipeline.rewrite_active is True


def test_no_warning_when_rewrite_is_disabled(base_cfg, index, tok, caplog):
    with caplog.at_level(logging.WARNING):
        pipeline = RetrievalPipeline(base_cfg, index, llm=NullLLM(), tokenizer=tok)
    assert pipeline.rewrite_active is False
    assert not any("will NOT be rewritten" in r.message for r in caplog.records)


def test_trace_records_whether_rewriting_actually_happened(base_cfg, index, tok):
    """A 0.0ms stage timing is too easy to misread; the trace states it outright."""
    cfg = base_cfg.patch({"retrieval.rewrite.enabled": True})

    dead = RetrievalPipeline(cfg, index, llm=NullLLM(), tokenizer=tok)
    span = next(s for s in dead.retrieve("apixaban dose").trace.spans if s.name == "query_rewrite")
    assert span.meta["rewrite_active"] is False
    assert span.meta["n_queries"] == 1

    live = RetrievalPipeline(cfg, index, llm=StubLLM(), tokenizer=tok)
    span = next(s for s in live.retrieve("apixaban dose").trace.spans if s.name == "query_rewrite")
    assert span.meta["rewrite_active"] is True
    assert span.meta["n_queries"] > 1


def test_a_working_rewrite_actually_changes_the_query_set(base_cfg, index, tok):
    cfg = base_cfg.patch({"retrieval.rewrite.enabled": True, "retrieval.rewrite.max_queries": 3})
    llm = StubLLM()
    result = RetrievalPipeline(cfg, index, llm=llm, tokenizer=tok).retrieve("apixaban dose")
    assert llm.calls == 1
    assert result.queries[0] == "apixaban dose", "the original query must always survive"
    assert len(result.queries) == 3


# --- rewrite_query itself ----------------------------------------------------------


def test_rewrite_keeps_the_original_query_first():
    """A rewrite that drifts off-topic must never be able to lose the answer."""
    out = rewrite_query(StubLLM(), "original question", max_queries=3)
    assert out[0] == "original question"


def test_rewrite_falls_back_cleanly_without_an_llm():
    assert rewrite_query(NullLLM(), "original question", max_queries=3) == ["original question"]


def test_rewrite_is_skipped_when_max_queries_is_one():
    llm = StubLLM()
    assert rewrite_query(llm, "q", max_queries=1) == ["q"]
    assert llm.calls == 0, "no LLM round trip should be spent to produce zero variants"


def test_rewrite_deduplicates_variants():
    class Dupes(StubLLM):
        def complete_json(self, prompt, system=None, **kw):
            self.calls += 1
            return {"queries": ["q", "Q", "other"]}

    out = rewrite_query(Dupes(), "q", max_queries=5)
    assert out == ["q", "other"]


def test_rewrite_survives_a_malformed_response():
    class Junk(StubLLM):
        def complete_json(self, prompt, system=None, **kw):
            self.calls += 1
            return {"queries": "not a list"}

    assert rewrite_query(Junk(), "q", max_queries=3) == ["q"]
