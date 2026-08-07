"""Golden set construction.

The recipe is: sample chunks, have a model write a question answerable from them, then
screen hard. Roughly a third of generated questions are unusable, and the screening
stage is what separates a golden set from a pile of LLM output.

Four question types, deliberately:

``factoid``       answerable from one chunk
``multi_hop``     needs two chunks from different documents
``aggregation``   "how many", "list all" - needs several chunks and a count
``unanswerable``  the answer is not in the corpus at all

The last one matters most and is the one almost nobody includes. Without it you cannot
tell a system that knows things from a system that will say anything, and a RAG system
that confidently answers questions its corpus cannot support is worse than useless in
a clinical setting. Everything else measures capability; this measures restraint.

The screening criteria below are not stylistic. LLM-generated questions default to
being *context-dependent* - "What was the primary endpoint of this trial?" - which is
unanswerable as a standalone query no matter how good the retriever is. Such a
question measures nothing except the question writer's carelessness, so self-
containment is a hard reject.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from ragmed.index.store import CorpusIndex
from ragmed.llm import LLM, LLMError
from ragmed.types import Chunk, GoldenItem

log = logging.getLogger(__name__)

WRITER_SYSTEM = (
    "You write evaluation questions for a clinical retrieval system. Your questions "
    "are answerable by someone who has never seen the source passage and is searching "
    "a large corpus. You never refer to 'this study', 'the passage', or 'the authors'."
)

SCREENER_SYSTEM = (
    "You are a strict reviewer of evaluation questions. You reject far more than you "
    "accept. A flawed question that reaches the benchmark corrupts every number "
    "measured with it."
)

FACTOID_PROMPT = """Write ONE question that is answered by the PASSAGE below.

Requirements:
- Self-contained: it must make sense to someone searching a large medical corpus who \
has never seen this passage. Never write "this study", "the trial", "the authors", \
"the passage".
- Name the specific drug, population, condition, or measurement the question is about.
- It must have a single, checkable answer that is stated in the passage.
- Do not ask about the passage's structure, its authors, or its publication.

Return JSON only: {{"question": "...", "answer": "..."}}

PASSAGE:
{passage}"""

MULTI_HOP_PROMPT = """Write ONE question that can only be answered by combining BOTH \
passages below.

Requirements:
- Answering it must genuinely require both passages. If either passage alone is \
enough, the question is wrong.
- Self-contained: never refer to "these studies", "the passages", or "both trials".
- Name the specific entities involved.

Return JSON only: {{"question": "...", "answer": "...", "why_both": "one sentence on \
what each passage contributes"}}

PASSAGE A:
{passage_a}

PASSAGE B:
{passage_b}"""

AGGREGATION_PROMPT = """Write ONE aggregation question answered by combining the \
PASSAGES below - a question starting with "How many", "Which", or "List all".

Requirements:
- The answer must require gathering information across several passages, not one.
- Self-contained and specific about what is being counted or listed.
- The answer must be fully determined by the passages shown.

Return JSON only: {{"question": "...", "answer": "..."}}

PASSAGES:
{passages}"""

UNANSWERABLE_PROMPT = """Read the PASSAGE below, then write ONE question that is \
closely related to its topic but that the passage CANNOT answer.

Requirements:
- It must be a realistic clinical question a doctor might genuinely ask.
- It must be adjacent to the passage - same disease area or drug class - so that a \
retrieval system will confidently retrieve this passage for it. Something obviously \
off-topic is useless here.
- The passage must contain nothing that answers it. Shift to a different outcome, a \
different population, a different comparison, or a specific number the passage omits.
- Self-contained and specific.

Return JSON only: {{"question": "...", "why_unanswerable": "one sentence"}}

PASSAGE:
{passage}"""

SCREEN_PROMPT = """Review this evaluation question against the SOURCE PASSAGES it was \
written from.

Reject the question if ANY of these is true:
1. It is not self-contained: it refers to "this study", "the passage", "the authors", \
"the trial", or otherwise assumes the reader can already see the source.
2. Its answer is not actually stated in the source passages.
3. It is vague, or several different answers would be equally correct.
4. It asks about document metadata (authors, journal, publication date, funding) \
rather than clinical content.
5. It is so generic that it could be asked of almost any medical paper.
{extra_criteria}
Return JSON only, using the RULE NUMBERS above:
{{"keep": true, "violated_rules": []}}
If any rule is violated, set "keep" to false and put every violated rule's number in \
"violated_rules".

QUESTION: {question}
PROPOSED ANSWER: {answer}

SOURCE PASSAGES:
{passages}"""

MULTI_HOP_EXTRA = (
    "6. Either passage alone is sufficient to answer it - it is not genuinely multi-hop.\n"
)
UNANSWERABLE_EXTRA = (
    "6. The source passages DO in fact answer it - it is not genuinely unanswerable.\n"
    "7. It is so unrelated to the passages that no retrieval system would surface them.\n"
)

# Small models reliably return rule *numbers* and unreliably return prose. Asking for
# numbers and mapping them here gives a rejection report that is actually readable -
# the first pilot reported its top reasons as "27  1" and "15  3", which is worthless
# for deciding whether the screener is behaving.
RULE_NAMES: dict[int, str] = {
    1: "not self-contained (refers to 'this study' / 'the passage')",
    2: "answer not stated in the source passages",
    3: "vague, or several answers equally correct",
    4: "asks about document metadata, not clinical content",
    5: "too generic to discriminate between papers",
}
MULTI_HOP_RULE_NAMES = {**RULE_NAMES, 6: "not genuinely multi-hop (one passage suffices)"}
UNANSWERABLE_RULE_NAMES = {
    **RULE_NAMES,
    6: "actually answerable from the passages",
    7: "so unrelated no retriever would surface these passages",
}


@dataclass
class BuildSpec:
    n_factoid: int = 80
    n_multi_hop: int = 40
    n_aggregation: int = 20
    n_unanswerable: int = 40
    seed: int = 20260806
    # Aggregation questions need several related chunks to count over.
    aggregation_group_size: int = 4
    max_attempts_multiplier: int = 3


@dataclass
class BuildReport:
    generated: int = 0
    kept: int = 0
    rejected: int = 0
    errors: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.generated if self.generated else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "kept": self.kept,
            "rejected": self.rejected,
            "errors": self.errors,
            "rejection_rate": round(self.rejection_rate, 4),
            "rejection_reasons": dict(
                sorted(self.rejection_reasons.items(), key=lambda kv: -kv[1])
            ),
        }


def _screen(
    llm: LLM,
    question: str,
    answer: str,
    passages: list[Chunk],
    extra: str = "",
    rule_names: dict[int, str] | None = None,
) -> tuple[bool, list[str]]:
    names = rule_names or RULE_NAMES
    body = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(passages, start=1))
    try:
        data = llm.complete_json(
            SCREEN_PROMPT.format(
                question=question, answer=answer, passages=body, extra_criteria=extra
            ),
            system=SCREENER_SYSTEM,
        )
    except LLMError as exc:
        # A screening failure must not admit an unscreened question: reject.
        log.warning("screening failed, rejecting the question: %s", exc)
        return False, ["screening_error"]

    if not isinstance(data, dict):
        return False, ["screening_error: response was not an object"]

    keep = bool(data.get("keep"))
    # Accept the documented key and the one models drift to, then resolve numbers
    # (and numeric strings) to rule names.
    raw = data.get("violated_rules", data.get("reasons", []))
    if not isinstance(raw, list):
        raw = []

    reasons: list[str] = []
    for entry in raw:
        try:
            num = int(str(entry).strip().rstrip("."))
        except (TypeError, ValueError):
            reasons.append(str(entry)[:80])
            continue
        reasons.append(names.get(num, f"rule {num}"))

    # A model that says keep=false without naming a rule has still rejected it.
    if not keep and not reasons:
        reasons = ["rejected without a stated rule"]
    return keep, reasons


def _related_chunk(index: CorpusIndex, seed: Chunk, rng: random.Random) -> Chunk | None:
    """Find a chunk from a *different* document that is topically close.

    Uses dense similarity when available, MeSH-term overlap otherwise. The
    different-document constraint is what makes a multi-hop question genuinely
    multi-hop rather than two halves of the same abstract.
    """
    if index.dense is not None:
        i = index.by_id.get(seed.chunk_id)
        if i is not None:
            sims = index.dense.matrix @ index.dense.matrix[i]
            order = sims.argsort()[::-1]
            for j in order[:60]:
                cand = index.chunks[int(j)]
                if cand.doc_id != seed.doc_id:
                    return cand
        return None

    mesh = set(seed.meta.get("mesh_terms", []))
    if mesh:
        pool = [
            c
            for c in index.chunks
            if c.doc_id != seed.doc_id and mesh & set(c.meta.get("mesh_terms", []))
        ]
        if pool:
            return rng.choice(pool)
    return None


def _aggregation_group(index: CorpusIndex, seed: Chunk, size: int, rng: random.Random) -> list[Chunk]:
    """A set of chunks sharing a MeSH term, so counting across them is meaningful."""
    mesh = set(seed.meta.get("mesh_terms", []))
    if not mesh:
        return []
    pool = [
        c
        for c in index.chunks
        if c.chunk_id != seed.chunk_id and mesh & set(c.meta.get("mesh_terms", []))
    ]
    if len(pool) < size - 1:
        return []
    return [seed, *rng.sample(pool, size - 1)]


def build_golden_set(
    llm: LLM,
    index: CorpusIndex,
    spec: BuildSpec | None = None,
) -> tuple[list[GoldenItem], BuildReport]:
    spec = spec or BuildSpec()
    if not llm.available():
        raise RuntimeError(
            "golden-set construction needs an LLM. Start Ollama and pull the model, "
            "or hand-write data/golden/golden_set.jsonl."
        )

    rng = random.Random(spec.seed)
    report = BuildReport()
    items: list[GoldenItem] = []

    # Sample without replacement so one verbose document cannot dominate the set.
    pool = list(index.chunks)
    rng.shuffle(pool)

    targets = [
        ("factoid", spec.n_factoid),
        ("multi_hop", spec.n_multi_hop),
        ("aggregation", spec.n_aggregation),
        ("unanswerable", spec.n_unanswerable),
    ]

    cursor = 0
    for qtype, target in targets:
        kept = 0
        attempts = 0
        max_attempts = target * spec.max_attempts_multiplier
        log.info("building %d %s questions", target, qtype)

        while kept < target and attempts < max_attempts and cursor < len(pool):
            attempts += 1
            seed = pool[cursor % len(pool)]
            cursor += 1

            try:
                built = _generate_one(llm, index, qtype, seed, spec, rng)
            except LLMError as exc:
                log.warning("%s generation failed: %s", qtype, exc)
                report.errors += 1
                continue
            if built is None:
                continue

            question, answer, sources, extra = built
            report.generated += 1

            rule_names = {
                "multi_hop": MULTI_HOP_RULE_NAMES,
                "unanswerable": UNANSWERABLE_RULE_NAMES,
            }.get(qtype, RULE_NAMES)
            keep, reasons = _screen(llm, question, answer, sources, extra, rule_names)
            if not keep:
                report.rejected += 1
                for r in reasons or ["unspecified"]:
                    key = r[:80]
                    report.rejection_reasons[key] = report.rejection_reasons.get(key, 0) + 1
                continue

            report.kept += 1
            kept += 1
            items.append(
                GoldenItem(
                    qid=f"{qtype}-{kept:04d}",
                    question=question,
                    question_type=qtype,  # type: ignore[arg-type]
                    # Unanswerable questions carry no gold chunks by definition.
                    gold_chunk_ids=[] if qtype == "unanswerable" else [c.chunk_id for c in sources],
                    answer=None if qtype == "unanswerable" else answer,
                    provenance={
                        "seed_chunk": seed.chunk_id,
                        "source_docs": sorted({c.doc_id for c in sources}),
                        "generator": llm.name,
                    },
                )
            )

        if kept < target:
            log.warning(
                "only produced %d/%d %s questions after %d attempts",
                kept, target, qtype, attempts,
            )

    log.info(
        "golden set: %d kept, %d rejected (%.0f%% rejection rate), %d errors",
        report.kept, report.rejected, 100 * report.rejection_rate, report.errors,
    )
    return items, report


def _generate_one(
    llm: LLM,
    index: CorpusIndex,
    qtype: str,
    seed: Chunk,
    spec: BuildSpec,
    rng: random.Random,
) -> tuple[str, str, list[Chunk], str] | None:
    """Returns (question, answer, source_chunks, extra_screening_criteria)."""
    if qtype == "factoid":
        data = llm.complete_json(FACTOID_PROMPT.format(passage=seed.text), system=WRITER_SYSTEM)
        q, a = data.get("question"), data.get("answer")
        return (q, a, [seed], "") if q and a else None

    if qtype == "multi_hop":
        partner = _related_chunk(index, seed, rng)
        if partner is None:
            return None
        data = llm.complete_json(
            MULTI_HOP_PROMPT.format(passage_a=seed.text, passage_b=partner.text),
            system=WRITER_SYSTEM,
        )
        q, a = data.get("question"), data.get("answer")
        return (q, a, [seed, partner], MULTI_HOP_EXTRA) if q and a else None

    if qtype == "aggregation":
        group = _aggregation_group(index, seed, spec.aggregation_group_size, rng)
        if not group:
            return None
        body = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(group, start=1))
        data = llm.complete_json(
            AGGREGATION_PROMPT.format(passages=body), system=WRITER_SYSTEM
        )
        q, a = data.get("question"), data.get("answer")
        return (q, a, group, "") if q and a else None

    if qtype == "unanswerable":
        data = llm.complete_json(
            UNANSWERABLE_PROMPT.format(passage=seed.text), system=WRITER_SYSTEM
        )
        q = data.get("question")
        why = data.get("why_unanswerable", "")
        # The screener is handed the seed passage and asked to confirm it does NOT
        # answer the question - so "unanswerable" is verified, not asserted.
        return (q, why, [seed], UNANSWERABLE_EXTRA) if q else None

    raise ValueError(f"unknown question type {qtype!r}")
