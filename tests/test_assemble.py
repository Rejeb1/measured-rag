from __future__ import annotations

import numpy as np
import pytest

from ragmed.config import AssemblyConfig
from ragmed.retrieve.assemble import assemble_context, deduplicate, order_for_attention
from ragmed.types import Chunk, Scored


class WordTokenizer:
    is_exact = True

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def make(cid: str, text: str, score: float, tokens: int = 10) -> Scored:
    chunk = Chunk(
        chunk_id=cid,
        doc_id=f"pmid:{cid}",
        ordinal=0,
        text=text,
        token_count=tokens,
        title=f"Title {cid}",
        source_type="pubmed",
        section="Results",
        meta={"pmid": cid},
    )
    return Scored(chunk=chunk, score=score, rank=0, stage="test")


# --- deduplication -------------------------------------------------------------


def test_near_duplicates_are_dropped_keeping_the_first():
    a = make("1", "Metformin reduces HbA1c by about one percent in adults.", 0.9)
    b = make("2", "Metformin reduces HbA1c by about one percent in adults.", 0.5)
    c = make("3", "Amoxicillin treats community acquired pneumonia.", 0.4)
    kept, dropped = deduplicate([a, b, c], threshold=0.9)
    assert dropped == 1
    assert [s.chunk.chunk_id for s in kept] == ["1", "3"]


def test_dedup_uses_embeddings_when_available():
    a = make("1", "totally different words here", 0.9)
    b = make("2", "nothing lexically in common at all", 0.5)
    # Lexically disjoint but semantically identical vectors.
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    kept, dropped = deduplicate([a, b], threshold=0.92, vectors=vectors)
    assert dropped == 1
    assert [s.chunk.chunk_id for s in kept] == ["1"]


def test_dedup_falls_back_to_lexical_overlap_without_vectors():
    """BM25-only is an ablation row, so dedup must work with no embeddings."""
    a = make("1", "alpha beta gamma delta epsilon", 0.9)
    b = make("2", "alpha beta gamma delta epsilon", 0.5)
    kept, dropped = deduplicate([a, b], threshold=0.9, vectors=None)
    assert dropped == 1
    assert len(kept) == 1


def test_distinct_chunks_are_all_kept():
    items = [make(str(i), f"unique content number {i} here", 1.0 / (i + 1)) for i in range(5)]
    kept, dropped = deduplicate(items, threshold=0.92)
    assert dropped == 0 and len(kept) == 5


# --- ordering ------------------------------------------------------------------


def test_edge_ordering_brackets_the_context_with_the_best_chunks():
    items = [make(str(i), f"text {i}", 1.0 - i / 10) for i in range(5)]
    ordered = order_for_attention(items, "edges")
    ids = [s.chunk.chunk_id for s in ordered]
    assert ids[0] == "0", "best chunk goes first"
    assert ids[-1] == "1", "second-best goes last"
    assert ids == ["0", "2", "4", "3", "1"]


def test_sequential_ordering_is_plain_descending():
    items = [make(str(i), f"text {i}", 1.0 - i / 10) for i in range(5)]
    assert [s.chunk.chunk_id for s in order_for_attention(items, "sequential")] == [
        "0", "1", "2", "3", "4"
    ]


def test_ordering_is_a_permutation_not_a_filter():
    items = [make(str(i), f"text {i}", 1.0 - i / 10) for i in range(7)]
    ordered = order_for_attention(items, "edges")
    assert sorted(s.chunk.chunk_id for s in ordered) == sorted(s.chunk.chunk_id for s in items)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_ordering_handles_degenerate_sizes(n):
    items = [make(str(i), f"text {i}", 1.0) for i in range(n)]
    assert len(order_for_attention(items, "edges")) == n


# --- budget and assembly --------------------------------------------------------


def test_budget_is_enforced():
    tok = WordTokenizer()
    cfg = AssemblyConfig(max_context_tokens=25, dedup_threshold=0.92, ordering="sequential")
    items = [make(str(i), f"unique text number {i}", 1.0 - i / 10, tokens=10) for i in range(5)]
    contexts, _, stats = assemble_context(items, cfg, tok)
    assert stats["context_tokens"] <= 25
    assert len(contexts) == 2
    assert stats["dropped_for_budget"] == 3


def test_budget_is_spent_on_the_highest_scoring_chunks():
    tok = WordTokenizer()
    cfg = AssemblyConfig(max_context_tokens=20, dedup_threshold=0.92, ordering="edges")
    items = [make(str(i), f"unique text number {i}", 1.0 - i / 10, tokens=10) for i in range(5)]
    contexts, _, _ = assemble_context(items, cfg, tok)
    assert sorted(s.chunk.chunk_id for s in contexts) == ["0", "1"]


def test_rendered_context_carries_checkable_citations():
    tok = WordTokenizer()
    cfg = AssemblyConfig(max_context_tokens=1000, dedup_threshold=0.92, ordering="edges")
    items = [make(str(i), f"unique text number {i}", 1.0 - i / 10) for i in range(3)]
    _, text, _ = assemble_context(items, cfg, tok)
    assert "[PMID:0 §Results]" in text
    assert text.count("---") == 2


def test_empty_candidates_are_safe():
    tok = WordTokenizer()
    cfg = AssemblyConfig()
    contexts, text, stats = assemble_context([], cfg, tok)
    assert contexts == [] and text == "" and stats["context_tokens"] == 0
