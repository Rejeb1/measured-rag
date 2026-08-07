"""Chunking tests.

These use a word-count stub instead of a real tokenizer so the suite is hermetic and
fast - no model download, no network. The properties being asserted (windows respect
the budget, section labels survive, ids are stable) are tokenizer-independent.
"""

from __future__ import annotations

import pytest

from ragmed.config import ChunkingConfig
from ragmed.ingest.chunking import chunk_document, pack_sentences, split_sentences
from ragmed.types import Document, Section


class WordTokenizer:
    """One token per whitespace-separated word."""

    is_exact = True

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


@pytest.fixture
def tok() -> WordTokenizer:
    return WordTokenizer()


# --- sentence splitting ------------------------------------------------------


def test_decimals_and_units_are_not_split(tok):
    text = "The dose was 5.5 mg daily. HbA1c fell to 7.0% overall."
    assert split_sentences(text) == [
        "The dose was 5.5 mg daily.",
        "HbA1c fell to 7.0% overall.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Agents such as metformin, e.g. 500 mg, were used. A second sentence follows.",
        "Smith et al. reported the same effect. A second sentence follows.",
        "Compared with placebo vs. control arms. A second sentence follows.",
    ],
)
def test_abbreviations_do_not_end_a_sentence(text, tok):
    assert len(split_sentences(text)) == 2


def test_enumerators_merge_forward():
    # "1." is a list marker, not a sentence.
    assert split_sentences("1. Start therapy now.") == ["1. Start therapy now."]


# --- packing -----------------------------------------------------------------


def test_windows_respect_the_token_budget(tok):
    sentences = [f"word{i} " * 20 for i in range(20)]
    windows = pack_sentences(sentences, tok, target_tokens=100, overlap_tokens=20)
    assert windows
    assert all(tok.count(w) <= 100 for w in windows)


def test_overlap_carries_context_between_windows(tok):
    sentences = [f"sentence{i} alpha beta gamma delta" for i in range(20)]
    with_overlap = pack_sentences(sentences, tok, 20, 10)
    without = pack_sentences(sentences, tok, 20, 0)
    # Overlap trades index size for context continuity; it must produce more windows.
    assert len(with_overlap) > len(without)


def test_oversized_sentence_is_broken_up_rather_than_dropped(tok):
    monster = "token " * 500
    windows = pack_sentences([monster.strip()], tok, target_tokens=50, overlap_tokens=10)
    assert len(windows) >= 10
    assert all(tok.count(w) <= 50 for w in windows)
    # Nothing may be silently lost.
    assert sum(tok.count(w) for w in windows) == 500


def test_pathological_overlap_terminates(tok):
    """Overlap >= target must not loop forever or emit duplicate windows."""
    sentences = [f"s{i} a b c d e" for i in range(30)]
    windows = pack_sentences(sentences, tok, target_tokens=12, overlap_tokens=999)
    assert windows
    assert all(windows[i] != windows[i + 1] for i in range(len(windows) - 1))


# --- document chunking -------------------------------------------------------


def _doc() -> Document:
    return Document(
        doc_id="pmid:12345",
        title="A Trial of Something",
        source_type="pubmed",
        sections=[
            Section("Background", "Background sentence here. " * 10),
            Section("Methods", "Methods sentence here. " * 10),
            Section("Results", "Results sentence here. " * 60),
            Section("Conclusions", "It worked."),
        ],
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        date="2023",
        meta={"pmid": "12345"},
    )


def test_structure_strategy_preserves_section_labels(tok):
    cfg = ChunkingConfig(strategy="structure", target_tokens=60, overlap_tokens=10, min_tokens=5)
    chunks = chunk_document(_doc(), cfg, tok)
    labels = {c.section for c in chunks}
    assert any(label and "Results" in label for label in labels)
    assert all(c.section is not None for c in chunks)


def test_fixed_strategy_discards_structure(tok):
    cfg = ChunkingConfig(strategy="fixed", target_tokens=60, overlap_tokens=10, min_tokens=5)
    chunks = chunk_document(_doc(), cfg, tok)
    assert chunks
    assert all(c.section is None for c in chunks)


def test_small_sections_merge_instead_of_becoming_runts(tok):
    cfg = ChunkingConfig(strategy="structure", target_tokens=200, overlap_tokens=0, min_tokens=20)
    chunks = chunk_document(_doc(), cfg, tok)
    # "Conclusions: It worked." is 2 tokens; it must never be its own chunk.
    assert all(c.token_count >= 20 for c in chunks)


def test_chunk_ids_are_stable_across_runs(tok):
    cfg = ChunkingConfig(strategy="structure", target_tokens=60, overlap_tokens=10, min_tokens=5)
    a = chunk_document(_doc(), cfg, tok)
    b = chunk_document(_doc(), cfg, tok)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    # Stability is what keeps a committed golden set valid across re-indexing.
    assert len({c.chunk_id for c in a}) == len(a)


def test_chunk_ids_change_when_content_changes(tok):
    cfg = ChunkingConfig(strategy="structure", target_tokens=60, overlap_tokens=10, min_tokens=5)
    base = chunk_document(_doc(), cfg, tok)
    edited = _doc()
    edited.sections[0].text = "Completely different background text. " * 10
    changed = chunk_document(edited, cfg, tok)
    assert base[0].chunk_id != changed[0].chunk_id


def test_citation_points_at_a_real_source(tok):
    cfg = ChunkingConfig(strategy="structure", target_tokens=60, overlap_tokens=10, min_tokens=5)
    chunk = chunk_document(_doc(), cfg, tok)[0]
    assert chunk.citation.startswith("PMID:12345")
    assert "§" in chunk.citation
    assert chunk.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"
