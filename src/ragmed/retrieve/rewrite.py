"""Query rewriting.

Real clinical questions arrive underspecified ("is metformin still first line?") while
the corpus is written in formal register ("metformin monotherapy as initial
pharmacologic treatment"). Rewriting generates a small set of alternative phrasings and
retrieves against all of them, then lets fusion sort it out.

It is off by default. It costs an LLM round trip on the critical path of every query,
and whether that buys enough recall to justify the latency is a question for the
ablation table, not an assumption.
"""

from __future__ import annotations

import logging

from ragmed.llm import LLM, LLMError

log = logging.getLogger(__name__)

SYSTEM = (
    "You rewrite clinical search queries to improve document retrieval over a corpus "
    "of PubMed abstracts and clinical guidelines. You do not answer questions."
)

PROMPT = """Rewrite the question below as {n} alternative search queries.

Make the variants genuinely different from each other and from the original:
- one using the formal clinical terminology a paper would use (generic drug names, \
full names of scales and classes, not abbreviations)
- one using the plain wording a clinician would say out loud
- one narrowed to the specific measurable outcome, threshold, or dose being asked about

Do not answer the question. Do not add facts that are not in it.

Return JSON only: {{"queries": ["...", "..."]}}

Question: {question}"""


def rewrite_query(llm: LLM, query: str, max_queries: int = 3) -> list[str]:
    """Return the original query plus any usable variants.

    The original is always first and is never dropped: a rewrite that drifts off-topic
    would otherwise be able to lose the answer entirely, turning an optional
    enhancement into a correctness risk.
    """
    if max_queries <= 1 or not llm.available():
        return [query]

    try:
        data = llm.complete_json(
            PROMPT.format(n=max_queries - 1, question=query),
            system=SYSTEM,
        )
    except LLMError as exc:
        # A rewrite failure must degrade to plain retrieval, never fail the request.
        log.warning("query rewrite failed, falling back to the original query: %s", exc)
        return [query]

    variants = data.get("queries", []) if isinstance(data, dict) else data
    if not isinstance(variants, list):
        log.warning("query rewrite returned %s, expected a list", type(variants).__name__)
        return [query]

    out = [query]
    seen = {query.strip().lower()}
    for v in variants:
        if not isinstance(v, str):
            continue
        v = v.strip()
        if v and v.lower() not in seen:
            out.append(v)
            seen.add(v.lower())
        if len(out) >= max_queries:
            break
    return out
