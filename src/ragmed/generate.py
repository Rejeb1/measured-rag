"""Answer generation.

The prompt does three things that the eval layer depends on, so they are contracts
rather than style choices:

1. It restricts the model to the supplied context. Without this the generator answers
   from parametric memory and faithfulness becomes unmeasurable - the model would be
   right for reasons the retriever cannot take credit for.
2. It requires inline ``[PMID:...]`` citations, which is what makes an answer
   checkable by a human in seconds.
3. It emits a literal ``INSUFFICIENT_CONTEXT`` sentinel when the context does not
   support an answer, which is what makes abstention measurable by string comparison
   instead of by a second model's opinion.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from ragmed.config import GenerationConfig
from ragmed.llm import LLM, LLMError

# From ragmed.types, not ragmed.eval: the eval package imports this module, so
# reaching into it here would close an import cycle.
from ragmed.types import ABSTAIN_SENTINEL, Answer, Scored, is_abstention, is_degenerate

log = logging.getLogger(__name__)

SYSTEM = (
    "You answer clinical questions using only the source passages provided to you. "
    "You are precise about numbers, doses, thresholds and drug names, and you never "
    "state anything the passages do not support."
)

PROMPT = """Answer the QUESTION using only the SOURCES below.

Rules:
- Use only the SOURCES. Do not add medical knowledge that is not written there, even \
if you are confident it is correct.
- Cite the source for every factual claim, inline, using its bracketed tag exactly as \
written, for example [PMID:31234567 §Results].
- Quote numbers, doses and thresholds exactly as the source states them. Do not round, \
convert units, or generalise.
- If the SOURCES do not contain enough information to answer, reply with exactly \
{sentinel} and one sentence saying what is missing. Do not guess. A refusal is a \
correct answer when the sources are inadequate.
- Be concise: answer the question asked, without preamble.

SOURCES:
{context}

QUESTION: {question}"""

# Matches the citation tags produced by Chunk.citation.
_CITATION_RE = re.compile(r"\[(PMID:[^\]\s]+(?:\s+§[^\]]+)?|[^\]]+?)\]")


def build_prompt(question: str, context: str) -> str:
    return PROMPT.format(question=question, context=context, sentinel=ABSTAIN_SENTINEL)


def extract_citations(text: str, contexts: list[Scored]) -> list[str]:
    """Return the citation tags used in the answer that correspond to real sources.

    Tags that do not match a supplied source are dropped rather than reported: a
    fabricated citation is a hallucination, and counting it as a citation would let a
    made-up reference improve the answer's apparent groundedness.
    """
    valid = {c.chunk.citation for c in contexts}
    found: list[str] = []
    for match in _CITATION_RE.findall(text):
        tag = match.strip()
        if tag in valid and tag not in found:
            found.append(tag)
    return found


def answer_question(
    llm: LLM,
    question: str,
    context: str,
    contexts: list[Scored],
    cfg: GenerationConfig | None = None,
) -> Answer:
    if not context.strip():
        # Nothing retrieved: refuse without spending a model call. The sentinel keeps
        # this indistinguishable from a model-issued refusal downstream, which is
        # correct - both are the system declining to answer.
        return Answer(
            text=f"{ABSTAIN_SENTINEL}: no source passages were retrieved for this question.",
            citations=[],
            abstained=True,
        )

    prompt = build_prompt(question, context)
    try:
        text = llm.complete(prompt, system=SYSTEM)
    except LLMError as exc:
        log.error("generation failed: %s", exc)
        raise

    text = text.strip()
    citations = extract_citations(text, contexts)
    # Same tolerant matcher the eval layer uses, so the flag stored on the Answer and
    # the flag computed during scoring can never disagree.
    abstained = is_abstention(text)

    degenerate = is_degenerate(text)
    if degenerate:
        # Loud, because the alternative is a judge scoring "@@@@@@@" as though it were
        # prose and folding that score into the reported mean.
        log.error(
            "generation collapsed into repeated tokens (%d chars) - this is a model or "
            "hardware failure, not an answer: %.60r",
            len(text), text,
        )

    if cfg is not None and cfg.require_citations and not abstained and not citations:
        # Not an error - the eval must see the answer as it was produced - but it is
        # a strong signal of ungrounded generation, so it is logged and flagged.
        log.warning("answer produced no valid citations; likely ungrounded")

    return Answer(text=text, citations=citations, abstained=abstained, degenerate=degenerate)


def stream_answer(
    llm: LLM,
    question: str,
    context: str,
) -> Iterator[str]:
    if not context.strip():
        yield f"{ABSTAIN_SENTINEL}: no source passages were retrieved for this question."
        return
    yield from llm.stream(build_prompt(question, context), system=SYSTEM)
