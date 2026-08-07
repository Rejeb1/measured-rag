"""Context assembly: deduplicate, order deliberately, enforce a budget.

Three things happen between "we have five good chunks" and "here is a prompt", and
each is a decision rather than plumbing:

*Deduplication* - overlapping chunks and near-identical abstracts (the same trial
reported twice, a guideline restating its own recommendation) waste budget and, worse,
make a claim look corroborated when it has a single source.

*Ordering* - models attend most strongly to the beginning and end of a context window
and sag in the middle, so the highest-scoring chunks go at the edges rather than in
descending order. This costs nothing and is measurable.

*Budget* - context is capped explicitly. Letting it grow with candidate count means
latency and cost scale with a number nobody chose deliberately.
"""

from __future__ import annotations

import logging

import numpy as np

from ragmed.config import AssemblyConfig
from ragmed.index.bm25 import tokenize
from ragmed.tokenization import Tokenizer
from ragmed.types import Scored

log = logging.getLogger(__name__)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(
    candidates: list[Scored],
    threshold: float,
    vectors: np.ndarray | None = None,
) -> tuple[list[Scored], int]:
    """Drop near-duplicates, keeping the higher-scoring copy.

    Uses embedding cosine when vectors are available and falls back to token-set
    Jaccard otherwise, so dedup still works in a BM25-only configuration - which
    matters, because that is one of the ablation rows.
    """
    kept: list[int] = []
    dropped = 0

    for i in range(len(candidates)):
        duplicate = False
        for j in kept:
            if vectors is not None:
                sim = float(np.dot(vectors[i], vectors[j]))
            else:
                sim = _jaccard(
                    set(tokenize(candidates[i].chunk.text)),
                    set(tokenize(candidates[j].chunk.text)),
                )
            if sim >= threshold:
                duplicate = True
                break
        if duplicate:
            dropped += 1
        else:
            kept.append(i)

    return [candidates[i] for i in kept], dropped


def order_for_attention(selected: list[Scored], ordering: str) -> list[Scored]:
    """Place the strongest chunks where the model attends most.

    ``edges``: best first, second-best last, third second, and so on - so the two
    highest-scoring chunks bracket the context. ``sequential``: plain descending.
    """
    if ordering == "sequential" or len(selected) <= 2:
        return list(selected)

    front: list[Scored] = []
    back: list[Scored] = []
    for i, item in enumerate(selected):
        (front if i % 2 == 0 else back).append(item)
    return front + list(reversed(back))


def assemble_context(
    candidates: list[Scored],
    cfg: AssemblyConfig,
    tok: Tokenizer,
    vectors: np.ndarray | None = None,
) -> tuple[list[Scored], str, dict[str, int]]:
    """Return (ordered chunks, rendered context, stats)."""
    if not candidates:
        return [], "", {"deduped": 0, "dropped_for_budget": 0, "context_tokens": 0}

    deduped, n_dropped = deduplicate(candidates, cfg.dedup_threshold, vectors)

    # Select in score order so the budget is spent on the best chunks, then reorder
    # for attention. Doing it the other way round would let a mid-ranked chunk
    # displace a better one purely because of its position.
    selected: list[Scored] = []
    used = 0
    budget_dropped = 0
    for cand in deduped:
        cost = cand.chunk.token_count
        if used + cost > cfg.max_context_tokens:
            budget_dropped += 1
            continue
        selected.append(cand)
        used += cost

    ordered = order_for_attention(selected, cfg.ordering)
    context = "\n\n---\n\n".join(c.chunk.render() for c in ordered)

    stats = {
        "deduped": n_dropped,
        "dropped_for_budget": budget_dropped,
        "context_tokens": used,
        "n_chunks": len(ordered),
    }
    return ordered, context, stats
