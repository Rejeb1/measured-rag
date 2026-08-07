"""Abstention detection.

Abstention is the one generation metric measured without an LLM: the generator emits a
sentinel and the eval layer string-matches it. That makes it the most trustworthy number
in the generation table - and it makes the matcher a single point of failure.

It failed. The prompt asks for `INSUFFICIENT_CONTEXT`; the model wrote `INSUFFICIENT
CONTEXT`; an exact-substring check scored every correct refusal as a hallucination. The
reported abstention accuracy was wrong in the *flattering* direction for the metric and
the damning direction for the system - it claimed zero refusals when refusals were
happening.

A contract enforced by exact match against free-form model output is a contract a model
will break by doing something reasonable. These tests pin the tolerant matcher.
"""

from __future__ import annotations

import pytest

from ragmed.eval.generation_metrics import ABSTAIN_SENTINEL, detect_abstention
from ragmed.generate import answer_question
from ragmed.llm import NullLLM
from ragmed.types import is_abstention


@pytest.mark.parametrize(
    "text",
    [
        "INSUFFICIENT_CONTEXT",
        "INSUFFICIENT CONTEXT",  # the form the model actually produced
        "INSUFFICIENT CONTEXT. The sources do not cover insulin infusion rates.",
        "INSUFFICIENT_CONTEXT: nothing here about carotid endarterectomy.",
        "insufficient context",
        "Insufficient Context.",
        "INSUFFICIENT-CONTEXT",
        "I cannot answer: insufficient context in the provided sources.",
    ],
)
def test_refusal_variants_are_all_detected(text):
    assert is_abstention(text), f"missed a refusal phrased as {text!r}"
    assert detect_abstention(text)


@pytest.mark.parametrize(
    "text",
    [
        "The dose is 500 mg daily [PMID:1 §Results].",
        "Empagliflozin reduced HbA1c by 0.62 percentage points.",
        "",
        "   ",
        "The context was sufficient to answer this question.",
        "[PMID:39344785 §Introduction]",
    ],
)
def test_real_answers_are_not_mistaken_for_refusals(text):
    assert not is_abstention(text)
    assert not detect_abstention(text)


def test_none_is_handled():
    assert not is_abstention(None)  # type: ignore[arg-type]


def test_the_generator_and_the_eval_layer_agree():
    """The two must never disagree about what a refusal looks like."""
    refusal = "INSUFFICIENT CONTEXT. Nothing in the sources addresses this."
    answer = answer_question(NullLLM(), "q?", "", [])
    assert answer.abstained is True
    assert detect_abstention(answer.text) is True
    assert is_abstention(refusal) == detect_abstention(refusal)


def test_empty_context_abstains_without_calling_a_model():
    answer = answer_question(NullLLM(), "any question", "", [])
    assert answer.abstained
    assert ABSTAIN_SENTINEL in answer.text
    assert answer.citations == []


def test_the_sentinel_itself_matches_its_own_pattern():
    """Guards against someone editing the sentinel without editing the matcher."""
    assert is_abstention(ABSTAIN_SENTINEL)
