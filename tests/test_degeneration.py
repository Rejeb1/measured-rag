"""Collapsed generation must be detected, never scored.

Small quantised models on a memory-constrained GPU intermittently collapse into a
repeated token:

    {"question": "What is@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

In the baseline run this hit **16 of 57 answers (28%)**, starting at question 9 and
clustering in the final third while per-question latency drifted 270s -> 386s. That
pattern is thermal/sustained-load degradation, not prompt content.

The danger is not the garbage itself - it is that an LLM-judge will cheerfully return a
faithfulness score for "@@@@@@@", and that score then enters the reported mean as though
the model had produced prose. A hardware failure would be laundered into a quality
metric. So degenerate output is excluded from every mean and counted separately.
"""

from __future__ import annotations

import pytest

from ragmed.eval.generation_metrics import (
    ItemGenerationScores,
    aggregate_generation,
    score_generation,
)
from ragmed.types import Answer, is_degenerate


class ExplodingLLM:
    """Fails if consulted - a degenerate answer must never reach the judge."""

    name = "exploding"

    def available(self) -> bool:
        return True

    def complete(self, *a, **kw):
        raise AssertionError("the judge must not be called on degenerate output")

    def complete_json(self, *a, **kw):
        raise AssertionError("the judge must not be called on degenerate output")

    def stream(self, *a, **kw):
        raise AssertionError("the judge must not be called on degenerate output")


# --- detection ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "@" * 31,
        '{"question": "What is@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
        "143 RCTs@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
        "[@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
        "answer text ###################### more",
        "aaaaaaaaaaaaaaaaaaaaaaaa",
    ],
)
def test_collapsed_generations_are_detected(text):
    assert is_degenerate(text)


@pytest.mark.parametrize(
    "text",
    [
        "The dose is 500 mg daily [PMID:1 §Results].",
        "INSUFFICIENT CONTEXT. The sources do not cover this.",
        "",
        "Values were 0.62, 0.87 and 1.37 across the three trials.",
        "A short run @@@ is fine.",
        "---",  # markdown separators are short
        "Hazard ratio 0.61 (95% CI, 0.43-0.85) -- see table.",
    ],
)
def test_real_answers_are_not_flagged(text):
    assert not is_degenerate(text)


def test_none_is_handled():
    assert not is_degenerate(None)  # type: ignore[arg-type]


# --- scoring --------------------------------------------------------------------


def test_degenerate_output_is_never_sent_to_the_judge():
    s = score_generation(
        ExplodingLLM(), "q1", "factoid", "What dose?", "@" * 40,
        "some context", [], is_answerable=True,
    )
    assert s.degenerate is True
    assert s.errors and "degenerate" in s.errors[0]


def test_degenerate_output_produces_no_scores():
    """None, not zero - a non-answer has no quality, and zero would drag the mean."""
    s = score_generation(
        ExplodingLLM(), "q1", "factoid", "q?", "@" * 40, "ctx", [], is_answerable=True,
    )
    assert s.faithfulness is None
    assert s.answer_relevance is None
    assert s.context_precision is None
    assert s.abstention_correct is None


def test_degenerate_items_are_excluded_from_means():
    items = [
        ItemGenerationScores(qid="a", question_type="factoid", faithfulness=1.0, n_claims=2),
        ItemGenerationScores(qid="b", question_type="factoid", degenerate=True, errors=["degenerate"]),
    ]
    agg = aggregate_generation(items)
    assert agg.faithfulness == pytest.approx(1.0), "garbage must not enter the mean"
    assert agg.n_degenerate == 1
    assert agg.to_dict()["degenerate_rate"] == pytest.approx(0.5)


def test_degenerate_output_cannot_inflate_the_hallucination_rate():
    items = [
        ItemGenerationScores(qid="a", question_type="factoid", n_claims=4, n_unsupported=1),
        ItemGenerationScores(qid="b", question_type="factoid", degenerate=True, n_claims=9, n_unsupported=9),
    ]
    agg = aggregate_generation(items)
    assert agg.hallucination_rate == pytest.approx(0.25)


def test_degenerate_rate_is_reported_in_the_summary():
    agg = aggregate_generation(
        [ItemGenerationScores(qid=str(i), question_type="factoid", degenerate=(i < 16)) for i in range(57)]
    )
    d = agg.to_dict()
    assert d["n_degenerate"] == 16
    assert d["degenerate_rate"] == pytest.approx(16 / 57, abs=1e-3)


def test_answer_carries_the_flag():
    assert Answer(text="ok").degenerate is False
