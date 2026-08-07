"""Generation metrics via LLM-as-judge, with explicit rubrics.

These answer the second question - "given the right context, did the generator use it
correctly" - and they are fundamentally weaker evidence than the retrieval metrics.
Three design decisions try to keep them honest:

1. **Faithfulness is decomposed, not holistic.** The judge extracts atomic claims from
   the answer and rules on each one separately against the context. Asking a model for
   "a faithfulness score from 0 to 1" produces a number correlated with fluency;
   asking "is this specific sentence supported by this specific text" is a question it
   can actually answer, and the resulting score is a countable ratio.

2. **Parse failures are errors, not zeros, and not defaults.** A judge that returns
   prose and gets coerced to 0.5 produces a number that looks like a measurement. Here
   the item is recorded as errored and excluded from the mean, and the error count is
   reported alongside every score.

3. **Abstention is not judged at all.** The generator emits a literal sentinel when it
   has insufficient context, so refusal is detected by string comparison. Asking a
   model to judge whether another model refused adds noise to the one metric that most
   needs to be exact.

Trust the numbers here only as far as `judge_validation.py` says you can.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ragmed.llm import LLM, LLMError
from ragmed.types import ABSTAIN_SENTINEL, Chunk, is_abstention, is_degenerate

log = logging.getLogger(__name__)

__all__ = [
    # Re-exported so callers can keep importing the sentinel from the module that
    # consumes it, even though it is defined in ragmed.types to break the import
    # cycle with ragmed.generate.
    "ABSTAIN_SENTINEL",
    "GenerationMetrics",
    "ItemGenerationScores",
    "aggregate_generation",
    "detect_abstention",
    "judge_context_precision",
    "judge_faithfulness",
    "judge_relevance",
    "score_generation",
]

JUDGE_SYSTEM = (
    "You are a strict evaluator of medical question-answering systems. You judge only "
    "what is written in front of you. You never use outside medical knowledge, and you "
    "never reward an answer for being plausible."
)

FAITHFULNESS_PROMPT = """Break the ANSWER into atomic factual claims, then rule on each \
one using only the CONTEXT.

Rules:
- A claim is "supported" only if the CONTEXT states it. Do not use outside knowledge. \
A claim that is true in general medicine but absent from the CONTEXT is "unsupported".
- A claim that contradicts the CONTEXT is "contradicted".
- Ignore hedging, citations, and restatements of the question; judge substantive claims only.
- If the answer contains no substantive claims, return an empty list.

Return JSON only:
{{"claims": [{{"claim": "...", "verdict": "supported|unsupported|contradicted", \
"evidence": "quote from CONTEXT, or empty"}}]}}

CONTEXT:
{context}

ANSWER:
{answer}"""

RELEVANCE_PROMPT = """Rate how well the ANSWER addresses the QUESTION.

Score with this rubric, and use the whole scale:
- 1.0: answers exactly what was asked, completely
- 0.75: answers the question but omits part of what was asked
- 0.5: partially addresses it, or answers a related but different question
- 0.25: mostly off-target, touches the topic only
- 0.0: does not address the question at all

Judge relevance to the QUESTION only. Do not reward or penalise factual accuracy here \
- that is measured separately.

Return JSON only: {{"score": 0.0, "reason": "one sentence"}}

QUESTION:
{question}

ANSWER:
{answer}"""

CONTEXT_PRECISION_PROMPT = """For each numbered CONTEXT PASSAGE, decide whether it \
contains information that helps answer the QUESTION.

"useful" means the passage contains at least one fact needed to answer the question. \
Being on the same general topic is not enough.

Return JSON only: {{"verdicts": [{{"index": 1, "useful": true}}, ...]}}
Return one verdict per passage, in order.

QUESTION:
{question}

CONTEXT PASSAGES:
{passages}"""


@dataclass(slots=True)
class ItemGenerationScores:
    qid: str
    question_type: str
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    abstained: bool = False
    abstention_correct: bool | None = None
    n_claims: int = 0
    n_unsupported: int = 0
    # Generation collapsed into repeated tokens. Excluded from every quality mean.
    degenerate: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GenerationMetrics:
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    abstention_accuracy: float | None = None
    hallucination_rate: float | None = None
    n_evaluated: int = 0
    n_errors: int = 0
    # Reported prominently: a high count invalidates every other number here, because
    # it means the model was not producing answers for that share of the set.
    n_degenerate: int = 0
    per_item: list[ItemGenerationScores] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        def r(v: float | None) -> float | None:
            return round(v, 4) if v is not None else None

        return {
            "faithfulness": r(self.faithfulness),
            "answer_relevance": r(self.answer_relevance),
            "context_precision": r(self.context_precision),
            "abstention_accuracy": r(self.abstention_accuracy),
            "hallucination_rate": r(self.hallucination_rate),
            "n_evaluated": self.n_evaluated,
            "n_judge_errors": self.n_errors,
            "n_degenerate": self.n_degenerate,
            "degenerate_rate": r(self.n_degenerate / self.n_evaluated) if self.n_evaluated else None,
        }


def detect_abstention(answer: str) -> bool:
    """Deterministic, by design - see the module docstring.

    Delegates to ragmed.types.is_abstention so the generator and the eval layer cannot
    drift apart on what counts as a refusal. It previously did an exact-substring check
    here, which missed every real refusal because the model wrote "INSUFFICIENT
    CONTEXT" rather than "INSUFFICIENT_CONTEXT".
    """
    return is_abstention(answer)


def judge_faithfulness(llm: LLM, context: str, answer: str) -> tuple[float, int, int]:
    """Returns (score, n_claims, n_unsupported). Raises LLMError on failure."""
    data = llm.complete_json(
        FAITHFULNESS_PROMPT.format(context=context, answer=answer), system=JUDGE_SYSTEM
    )
    claims = data.get("claims", []) if isinstance(data, dict) else []
    if not isinstance(claims, list):
        raise LLMError(f"faithfulness: expected a list of claims, got {type(claims).__name__}")
    if not claims:
        # No substantive claims means nothing was fabricated. Scoring this 0 would
        # punish a correct refusal harder than a confident hallucination.
        return 1.0, 0, 0

    supported = 0
    unsupported = 0
    for c in claims:
        verdict = str(c.get("verdict", "")).strip().lower() if isinstance(c, dict) else ""
        if verdict == "supported":
            supported += 1
        else:
            # "contradicted" and anything unrecognised both count against the score.
            unsupported += 1
    return supported / len(claims), len(claims), unsupported


def judge_relevance(llm: LLM, question: str, answer: str) -> float:
    data = llm.complete_json(
        RELEVANCE_PROMPT.format(question=question, answer=answer), system=JUDGE_SYSTEM
    )
    score = data.get("score") if isinstance(data, dict) else None
    if not isinstance(score, (int, float)):
        raise LLMError(f"relevance: expected a numeric score, got {score!r}")
    return max(0.0, min(1.0, float(score)))


def judge_context_precision(llm: LLM, question: str, chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    passages = "\n\n".join(
        f"[{i}] {c.text[:1200]}" for i, c in enumerate(chunks, start=1)
    )
    data = llm.complete_json(
        CONTEXT_PRECISION_PROMPT.format(question=question, passages=passages),
        system=JUDGE_SYSTEM,
    )
    verdicts = data.get("verdicts", []) if isinstance(data, dict) else []
    if not isinstance(verdicts, list) or not verdicts:
        raise LLMError("context precision: no verdicts returned")

    useful = 0
    seen: set[int] = set()
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        idx = v.get("index")
        if not isinstance(idx, int) or not 1 <= idx <= len(chunks) or idx in seen:
            continue
        seen.add(idx)
        if bool(v.get("useful")):
            useful += 1
    if not seen:
        raise LLMError("context precision: no usable verdicts")
    # Denominator is the passages actually ruled on, so a truncated judge response
    # does not silently look like a precision failure.
    return useful / len(seen)


def score_generation(
    llm: LLM,
    qid: str,
    question_type: str,
    question: str,
    answer: str,
    context: str,
    chunks: list[Chunk],
    is_answerable: bool,
) -> ItemGenerationScores:
    out = ItemGenerationScores(qid=qid, question_type=question_type)

    if is_degenerate(answer):
        # Collapsed generation is not a wrong answer, it is a non-answer. Scoring it
        # would let a hardware failure masquerade as a quality measurement: the judge
        # happily returns a faithfulness number for "@@@@@@@", and that number then
        # enters the mean. Record it as an error and leave every score None.
        out.degenerate = True
        out.errors.append("degenerate: generation collapsed into repeated tokens")
        return out

    out.abstained = detect_abstention(answer)
    # Refusing an unanswerable question is correct; refusing an answerable one is not.
    out.abstention_correct = out.abstained != is_answerable

    if out.abstained:
        # A refusal makes no claims and has no relevance to grade. Judging it anyway
        # would score every correct refusal as an irrelevant answer.
        out.faithfulness = 1.0
        return out

    try:
        out.faithfulness, out.n_claims, out.n_unsupported = judge_faithfulness(llm, context, answer)
    except LLMError as exc:
        out.errors.append(f"faithfulness: {exc}")

    try:
        out.answer_relevance = judge_relevance(llm, question, answer)
    except LLMError as exc:
        out.errors.append(f"relevance: {exc}")

    try:
        out.context_precision = judge_context_precision(llm, question, chunks)
    except LLMError as exc:
        out.errors.append(f"context_precision: {exc}")

    return out


def aggregate_generation(items: list[ItemGenerationScores]) -> GenerationMetrics:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    faith = [i.faithfulness for i in items if i.faithfulness is not None]
    rel = [i.answer_relevance for i in items if i.answer_relevance is not None]
    prec = [i.context_precision for i in items if i.context_precision is not None]
    abst = [i.abstention_correct for i in items if i.abstention_correct is not None]

    # Hallucination rate is over *answered* questions only: a refusal cannot
    # hallucinate, so including refusals would let a system that answers nothing
    # report a perfect score. Degenerate output is excluded for the same reason -
    # "@@@@@@@" makes no claims, true or false.
    answered = [i for i in items if not i.abstained and not i.degenerate and i.n_claims > 0]
    halluc = (
        sum(i.n_unsupported for i in answered) / sum(i.n_claims for i in answered)
        if answered
        else None
    )

    return GenerationMetrics(
        faithfulness=mean(faith),
        answer_relevance=mean(rel),
        context_precision=mean(prec),
        abstention_accuracy=(sum(abst) / len(abst) if abst else None),
        hallucination_rate=halluc,
        n_evaluated=len(items),
        n_errors=sum(1 for i in items if i.errors),
        n_degenerate=sum(1 for i in items if i.degenerate),
        per_item=items,
    )
