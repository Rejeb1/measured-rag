"""Core data model.

Everything downstream - indexes, fusion, reranking, evaluation, citations - operates
on ``Chunk``. Chunks are deliberately flat rather than nested inside documents: the
retrieval unit and the citation unit are the same object, so a chunk that surfaces in
a result already carries everything needed to point a reader at a real source.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# The literal string the generator is asked to emit when the retrieved context does not
# support an answer. It lives here, in the lowest-level module, because it is a contract
# *between* the generator and the eval layer: generation writes it, evaluation reads
# it. Defining it on either side makes the two import each other in a cycle, and
# defining it twice makes abstention silently unmeasurable the first time one copy is
# edited.
ABSTAIN_SENTINEL = "INSUFFICIENT_CONTEXT"

# Match it tolerantly. Instruction-following models routinely normalise an underscore
# into a space - the first generation run produced "INSUFFICIENT CONTEXT." on every
# refusal, and an exact-match check scored all 14 correct refusals as failures. The
# reported abstention accuracy was therefore *pessimistic* by exactly the number of
# questions the system got right.
#
# The lesson generalises past this one string: a contract enforced by exact match
# against free-form model output is a contract that will be broken by a model doing
# something reasonable. Match the intent, not the byte sequence.
import re as _re  # noqa: E402  (kept adjacent to the pattern it serves)

ABSTAIN_PATTERN = _re.compile(r"INSUFFICIENT[\s_-]*CONTEXT", _re.IGNORECASE)


def is_abstention(text: str) -> bool:
    """True when an answer declines for lack of supporting context.

    The single place this judgement is made. Both the generator (setting
    ``Answer.abstained``) and the eval layer (scoring abstention accuracy) call it, so
    they can never disagree about what a refusal looks like.
    """
    return bool(ABSTAIN_PATTERN.search(text or ""))


# A long run of one repeated character is the signature of a collapsed generation:
#     {"question": "What is@@@@@@@@@@@@@@@@@@@@@@@
# Small quantised models on a memory-constrained GPU do this intermittently, and it got
# worse over a four-hour eval run (28% of answers, clustering in the final third, with
# per-question latency drifting 270s -> 386s: thermal throttling, not prompt content).
#
# This MUST be detected rather than scored. A judge handed "@@@@@@@" will dutifully
# return a faithfulness number for it, and that number then enters the mean as though
# the model had produced content. Garbage has to be excluded loudly, not averaged in.
_DEGENERATE_RUN = _re.compile(r"(.)\1{11,}")


def is_degenerate(text: str) -> bool:
    """True when generation collapsed into a repeated token."""
    return bool(_DEGENERATE_RUN.search(text or ""))

QuestionType = Literal["factoid", "multi_hop", "aggregation", "unanswerable"]

QUESTION_TYPES: tuple[QuestionType, ...] = (
    "factoid",
    "multi_hop",
    "aggregation",
    "unanswerable",
)


@dataclass(slots=True)
class Section:
    """A titled span of a document. ``heading`` is None for preamble text."""

    heading: str | None
    text: str


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    source_type: str  # "pubmed" | "guideline"
    sections: list[Section] = field(default_factory=list)
    url: str | None = None
    date: str | None = None  # often just a year; kept as a string on purpose
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts = []
        for s in self.sections:
            if s.heading:
                parts.append(f"{s.heading}\n{s.text}")
            else:
                parts.append(s.text)
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        return cls(
            doc_id=d["doc_id"],
            title=d["title"],
            source_type=d["source_type"],
            sections=[Section(**s) for s in d.get("sections", [])],
            url=d.get("url"),
            date=d.get("date"),
            meta=d.get("meta", {}),
        )


@dataclass(slots=True)
class Chunk:
    """The retrieval and citation unit.

    ``chunk_id`` is content-addressed so that re-running ingestion over an unchanged
    corpus produces identical ids. That is what lets a golden set stay valid across
    re-indexing runs - without it, every re-ingest would silently invalidate every
    gold label and the eval suite would quietly start measuring nothing.
    """

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    token_count: int
    title: str
    source_type: str
    section: str | None = None
    url: str | None = None
    date: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, ordinal: int, text: str) -> str:
        h = hashlib.sha1(f"{doc_id}::{ordinal}::{text}".encode()).hexdigest()
        return h[:16]

    @property
    def citation(self) -> str:
        """A short human-checkable pointer, e.g. ``PMID:31234567 §Methods``."""
        ref = self.meta.get("pmid")
        head = f"PMID:{ref}" if ref else self.doc_id
        return f"{head} §{self.section}" if self.section else head

    def render(self) -> str:
        """The form a chunk takes inside an assembled context window."""
        header = f"[{self.citation}] {self.title}"
        if self.date:
            header += f" ({self.date})"
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


@dataclass(slots=True)
class Scored:
    """A chunk with the score that put it where it is, plus the sub-scores behind it.

    ``components`` is kept populated through every stage rather than overwritten, so a
    failure analysis can ask "was this ranked highly by BM25 and then demoted by the
    reranker?" - which is the question that actually explains a bad result.
    """

    chunk: Chunk
    score: float
    rank: int
    stage: str
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "score": self.score,
            "rank": self.rank,
            "stage": self.stage,
            "components": self.components,
            "citation": self.chunk.citation,
        }


@dataclass(slots=True)
class GoldenItem:
    """One question / answer / source-chunk triple.

    ``gold_chunk_ids`` is empty exactly when ``question_type == "unanswerable"``. Those
    items are excluded from retrieval metrics (there is nothing to retrieve) and drive
    the abstention rate instead.
    """

    qid: str
    question: str
    question_type: QuestionType
    gold_chunk_ids: list[str] = field(default_factory=list)
    answer: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_answerable(self) -> bool:
        return self.question_type != "unanswerable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldenItem:
        return cls(
            qid=d["qid"],
            question=d["question"],
            question_type=d["question_type"],
            gold_chunk_ids=list(d.get("gold_chunk_ids", [])),
            answer=d.get("answer"),
            provenance=d.get("provenance", {}),
        )


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    confidence: float | None = None
    # Set when generation collapsed into repeated tokens. Such an answer is not a
    # wrong answer - it is not an answer at all - and must be excluded from quality
    # means rather than scored as content.
    degenerate: bool = False
