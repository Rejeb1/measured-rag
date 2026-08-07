"""Tests for the eval layer's judgement calls.

These cover the logic that decides what a number *means* - which failure a question is
attributed to, whether a gold label survives re-chunking, whether a refusal counts as
a hallucination. Getting these wrong produces a table that renders perfectly and says
something false, which is the worst failure mode this project has.
"""

from __future__ import annotations

import json

import pytest

from ragmed.eval.ablation import remap_golden
from ragmed.eval.failure_analysis import analyse, classify
from ragmed.eval.generation_metrics import (
    ABSTAIN_SENTINEL,
    ItemGenerationScores,
    aggregate_generation,
    detect_abstention,
    judge_context_precision,
    judge_faithfulness,
    score_generation,
)
from ragmed.eval.judge_validation import HumanLabel, validate_judge
from ragmed.eval.retrieval_metrics import ItemRetrievalMetrics
from ragmed.eval.runner import EvalRun, ItemRecord
from ragmed.llm import LLMError, LLMParseError, extract_json
from ragmed.types import Chunk, GoldenItem

# --- helpers -------------------------------------------------------------------


class FakeLLM:
    """Returns canned payloads keyed by a substring of the prompt."""

    name = "fake"

    def __init__(self, responses: dict[str, object], available: bool = True):
        self.responses = responses
        self._available = available
        self.calls: list[str] = []

    def available(self) -> bool:
        return self._available

    def complete(self, prompt: str, system: str | None = None, **kw) -> str:
        self.calls.append(prompt)
        for key, value in self.responses.items():
            if key in prompt:
                return value if isinstance(value, str) else json.dumps(value)
        raise LLMError("no canned response matched")

    def complete_json(self, prompt: str, system: str | None = None, **kw):
        return extract_json(self.complete(prompt, system, **kw))

    def stream(self, prompt: str, system: str | None = None, **kw):
        yield self.complete(prompt, system, **kw)


def chunk(cid: str, doc: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        ordinal=0,
        text=text,
        token_count=len(text.split()),
        title="T",
        source_type="pubmed",
        meta={"pmid": cid},
    )


def record(**kw) -> ItemRecord:
    base = dict(
        qid="q1",
        question="What dose?",
        question_type="factoid",
        gold_chunk_ids=["g1"],
        ranked_ids=["g1"],
        context_ids=["g1"],
    )
    base.update(kw)
    return ItemRecord(**base)


# --- JSON extraction -----------------------------------------------------------


def test_extract_json_handles_fenced_blocks():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_handles_surrounding_prose():
    assert extract_json('Sure! Here you go: {"a": 1} Hope that helps.') == {"a": 1}


def test_extract_json_raises_rather_than_inventing_a_value():
    with pytest.raises(LLMParseError):
        extract_json("I'm afraid I can't do that.")


# --- faithfulness ---------------------------------------------------------------


def test_faithfulness_is_the_supported_claim_ratio():
    llm = FakeLLM({
        "atomic factual claims": {
            "claims": [
                {"claim": "A", "verdict": "supported"},
                {"claim": "B", "verdict": "unsupported"},
                {"claim": "C", "verdict": "supported"},
                {"claim": "D", "verdict": "contradicted"},
            ]
        }
    })
    score, n, unsupported = judge_faithfulness(llm, "ctx", "ans")
    assert score == pytest.approx(0.5)
    assert n == 4 and unsupported == 2


def test_contradicted_claims_count_against_the_score():
    llm = FakeLLM({"atomic factual claims": {"claims": [{"claim": "A", "verdict": "contradicted"}]}})
    score, _, unsupported = judge_faithfulness(llm, "ctx", "ans")
    assert score == 0.0 and unsupported == 1


def test_an_answer_with_no_claims_is_not_penalised():
    """Otherwise a correct refusal scores worse than a confident fabrication."""
    llm = FakeLLM({"atomic factual claims": {"claims": []}})
    score, n, _ = judge_faithfulness(llm, "ctx", "ans")
    assert score == 1.0 and n == 0


def test_malformed_judge_output_raises_instead_of_defaulting():
    llm = FakeLLM({"atomic factual claims": {"claims": "not a list"}})
    with pytest.raises(LLMError):
        judge_faithfulness(llm, "ctx", "ans")


# --- context precision -----------------------------------------------------------


def test_context_precision_counts_useful_passages():
    llm = FakeLLM({
        "CONTEXT PASSAGES": {
            "verdicts": [
                {"index": 1, "useful": True},
                {"index": 2, "useful": False},
                {"index": 3, "useful": True},
                {"index": 4, "useful": False},
            ]
        }
    })
    chunks = [chunk(str(i), "d", f"text {i}") for i in range(1, 5)]
    assert judge_context_precision(llm, "q?", chunks) == pytest.approx(0.5)


def test_context_precision_denominator_is_passages_actually_judged():
    """A truncated judge response must not look like a precision failure."""
    llm = FakeLLM({"CONTEXT PASSAGES": {"verdicts": [{"index": 1, "useful": True}]}})
    chunks = [chunk(str(i), "d", f"text {i}") for i in range(1, 5)]
    assert judge_context_precision(llm, "q?", chunks) == pytest.approx(1.0)


def test_context_precision_ignores_out_of_range_and_duplicate_indices():
    llm = FakeLLM({
        "CONTEXT PASSAGES": {
            "verdicts": [
                {"index": 1, "useful": True},
                {"index": 1, "useful": False},  # duplicate
                {"index": 99, "useful": True},  # out of range
            ]
        }
    })
    chunks = [chunk("1", "d", "text")]
    assert judge_context_precision(llm, "q?", chunks) == pytest.approx(1.0)


# --- abstention ------------------------------------------------------------------


def test_abstention_detection_is_a_string_comparison():
    assert detect_abstention(f"{ABSTAIN_SENTINEL}: nothing about dosing here.")
    assert not detect_abstention("The dose is 500 mg daily [PMID:1].")


def test_refusing_an_unanswerable_question_is_correct():
    llm = FakeLLM({})
    s = score_generation(
        llm, "q1", "unanswerable", "q?", f"{ABSTAIN_SENTINEL}: not covered.",
        "ctx", [], is_answerable=False,
    )
    assert s.abstained is True
    assert s.abstention_correct is True
    assert s.faithfulness == 1.0
    assert llm.calls == [], "a refusal must not be sent to the judge"


def test_refusing_an_answerable_question_is_incorrect():
    s = score_generation(
        FakeLLM({}), "q1", "factoid", "q?", f"{ABSTAIN_SENTINEL}: unsure.",
        "ctx", [], is_answerable=True,
    )
    assert s.abstention_correct is False


def test_answering_an_unanswerable_question_is_incorrect():
    llm = FakeLLM({
        "atomic factual claims": {"claims": [{"claim": "X", "verdict": "unsupported"}]},
        "Rate how well": {"score": 0.9},
        "CONTEXT PASSAGES": {"verdicts": [{"index": 1, "useful": False}]},
    })
    s = score_generation(
        llm, "q1", "unanswerable", "q?", "The dose is 500 mg.", "ctx",
        [chunk("1", "d", "t")], is_answerable=False,
    )
    assert s.abstained is False
    assert s.abstention_correct is False


def test_judge_errors_are_recorded_not_scored():
    llm = FakeLLM({"Rate how well": {"score": 0.8}})  # faithfulness/precision will fail
    s = score_generation(
        llm, "q1", "factoid", "q?", "an answer", "ctx", [chunk("1", "d", "t")],
        is_answerable=True,
    )
    assert s.faithfulness is None
    assert s.answer_relevance == pytest.approx(0.8)
    assert len(s.errors) == 2


# --- aggregation -----------------------------------------------------------------


def test_errored_items_are_excluded_from_means_not_zeroed():
    items = [
        ItemGenerationScores(qid="a", question_type="factoid", faithfulness=1.0, n_claims=2),
        ItemGenerationScores(qid="b", question_type="factoid", faithfulness=None, errors=["boom"]),
    ]
    agg = aggregate_generation(items)
    assert agg.faithfulness == pytest.approx(1.0)
    assert agg.n_errors == 1


def test_hallucination_rate_excludes_refusals():
    """A system that refuses everything must not score a perfect hallucination rate."""
    items = [
        ItemGenerationScores(qid="a", question_type="factoid", abstained=True, n_claims=0),
        ItemGenerationScores(
            qid="b", question_type="factoid", n_claims=4, n_unsupported=2, faithfulness=0.5
        ),
    ]
    agg = aggregate_generation(items)
    assert agg.hallucination_rate == pytest.approx(0.5)


# --- failure attribution ----------------------------------------------------------


def test_gold_never_retrieved_is_a_retrieval_failure():
    f = classify(record(ranked_ids=["x", "y"], context_ids=["x"]), None, None)
    assert f is not None and f.category == "retrieval"


def test_gold_retrieved_but_below_the_cutoff_is_a_ranking_failure():
    rm = ItemRetrievalMetrics(qid="q1", question_type="factoid", first_hit_rank=17)
    f = classify(record(ranked_ids=["x", "g1"], context_ids=["x"]), rm, None)
    assert f is not None and f.category == "ranking"
    assert "17" in f.detail


def test_correct_context_but_unfaithful_answer_is_a_generation_failure():
    rm = ItemRetrievalMetrics(qid="q1", question_type="factoid", recall_at_k={10: 1.0}, first_hit_rank=1)
    gs = ItemGenerationScores(qid="q1", question_type="factoid", faithfulness=0.3, n_claims=4, n_unsupported=3)
    f = classify(record(answer="something"), rm, gs)
    assert f is not None and f.category == "generation"


def test_a_retrieval_failure_is_not_reattributed_to_generation():
    """Attribution goes to the earliest broken stage, or the counts double-count."""
    gs = ItemGenerationScores(qid="q1", question_type="factoid", faithfulness=0.0, n_claims=3, n_unsupported=3)
    f = classify(record(ranked_ids=["x"], context_ids=["x"], answer="wrong"), None, gs)
    assert f is not None and f.category == "retrieval"


def test_answering_an_unanswerable_question_is_flagged():
    r = record(qid="u1", question_type="unanswerable", gold_chunk_ids=[], ranked_ids=["x"],
               context_ids=["x"], answer="Confidently wrong.", abstained=False)
    f = classify(r, None, None)
    assert f is not None and f.category == "abstention"


def test_correctly_refusing_an_unanswerable_question_is_clean():
    r = record(qid="u1", question_type="unanswerable", gold_chunk_ids=[], ranked_ids=["x"],
               context_ids=["x"], answer=ABSTAIN_SENTINEL, abstained=True)
    assert classify(r, None, None) is None


def test_a_fully_correct_item_is_clean():
    rm = ItemRetrievalMetrics(qid="q1", question_type="factoid", recall_at_k={10: 1.0}, first_hit_rank=1)
    gs = ItemGenerationScores(qid="q1", question_type="factoid", faithfulness=1.0, n_claims=3)
    assert classify(record(answer="good"), rm, gs) is None


def test_analyse_counts_and_ranks_failures():
    run = EvalRun(label="t")
    run.records = [
        record(qid="ok"),
        record(qid="bad", ranked_ids=["x"], context_ids=["x"]),
    ]
    run.retrieval.per_item = [
        ItemRetrievalMetrics(qid="ok", question_type="factoid", recall_at_k={10: 1.0}, first_hit_rank=1),
        ItemRetrievalMetrics(qid="bad", question_type="factoid", recall_at_k={10: 0.0}),
    ]
    report = analyse(run)
    assert report.n_items == 2
    assert report.counts["retrieval"] == 1
    assert report.n_clean == 1


# --- golden set remapping ---------------------------------------------------------


def test_remap_projects_gold_labels_onto_new_chunking():
    old = [chunk("old1", "pmid:1", "metformin reduces hba1c by one percent in adults with diabetes")]
    # Same document re-chunked smaller: the first piece still contains the gold text.
    new = [
        chunk("new1", "pmid:1", "metformin reduces hba1c by one percent in adults with diabetes"),
        chunk("new2", "pmid:1", "amoxicillin treats pneumonia in adults"),
    ]
    golden = [GoldenItem(qid="q1", question="q?", question_type="factoid", gold_chunk_ids=["old1"])]
    remapped, stats = remap_golden(golden, old, new)
    assert stats["remapped"] == 1 and stats["dropped"] == 0
    assert remapped[0].gold_chunk_ids == ["new1"]


def test_remap_can_expand_one_gold_label_into_several():
    """A 1024-token gold chunk legitimately becomes two 512-token ones."""
    text = " ".join(f"word{i}" for i in range(100))
    old = [chunk("old1", "pmid:1", text)]
    new = [chunk("new1", "pmid:1", text), chunk("new2", "pmid:1", text)]
    golden = [GoldenItem(qid="q1", question="q?", question_type="factoid", gold_chunk_ids=["old1"])]
    remapped, stats = remap_golden(golden, old, new)
    assert set(remapped[0].gold_chunk_ids) == {"new1", "new2"}
    assert stats["expanded"] == 1


def test_remap_drops_items_whose_gold_text_vanished():
    old = [chunk("old1", "pmid:1", "alpha beta gamma delta epsilon zeta")]
    new = [chunk("new1", "pmid:1", "completely unrelated replacement content here")]
    golden = [GoldenItem(qid="q1", question="q?", question_type="factoid", gold_chunk_ids=["old1"])]
    remapped, stats = remap_golden(golden, old, new)
    assert remapped == [] and stats["dropped"] == 1


def test_remap_passes_unanswerable_items_through_untouched():
    golden = [GoldenItem(qid="u1", question="q?", question_type="unanswerable", gold_chunk_ids=[])]
    remapped, stats = remap_golden(golden, [], [])
    assert len(remapped) == 1 and stats["unanswerable_passthrough"] == 1


# --- judge validation --------------------------------------------------------------


def test_validation_reports_agreement_and_bias_direction():
    judged = [
        ItemGenerationScores(qid="a", question_type="factoid", faithfulness=1.0),
        ItemGenerationScores(qid="b", question_type="factoid", faithfulness=1.0),
        ItemGenerationScores(qid="c", question_type="factoid", faithfulness=0.9),
    ]
    humans = [
        HumanLabel("a", "faithfulness", 1.0),
        HumanLabel("b", "faithfulness", 0.4),  # judge was generous here
        HumanLabel("c", "faithfulness", 1.0),
    ]
    result = validate_judge(judged, humans)["faithfulness"]
    assert result.n == 3
    assert result.agreement == pytest.approx(2 / 3)
    assert result.bias > 0, "judge scored higher than humans overall"


def test_a_tiny_label_set_is_reported_as_unvalidated():
    judged = [ItemGenerationScores(qid="a", question_type="factoid", faithfulness=1.0)]
    result = validate_judge(judged, [HumanLabel("a", "faithfulness", 1.0)])["faithfulness"]
    assert "too few" in result.verdict()


def test_confidence_interval_is_wide_at_small_n():
    judged = [ItemGenerationScores(qid=str(i), question_type="f", faithfulness=1.0) for i in range(50)]
    humans = [HumanLabel(str(i), "faithfulness", 1.0 if i < 40 else 0.0) for i in range(50)]
    r = validate_judge(judged, humans)["faithfulness"]
    lo, hi = r.agreement_ci95
    assert lo < r.agreement < hi
    assert hi - lo > 0.15, "50 labels should not produce a tight interval"
