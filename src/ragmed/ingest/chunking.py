"""Chunking.

Most systems pick 512 tokens on day one and never revisit it. Here chunking is a
first-class variable: strategy and size are config, so "structure-aware vs fixed" and
"256 vs 512 vs 1024" are rows in the ablation table rather than folklore.

Two strategies:

``structure``  Split on section boundaries first (BACKGROUND / METHODS / RESULTS for a
               structured abstract, headings for a guideline), then pack whole sections
               together up to the target size. A section is only broken apart if it
               exceeds the target on its own. Small sections merge rather than becoming
               3-token chunks that pollute the index.

``fixed``      Ignore structure entirely and slide a fixed window over the document.
               This is the baseline the structure-aware strategy has to beat.

Both pack *sentences*, never raw token slices. A chunk that begins mid-sentence is
unusable as a citation even when it is retrieved correctly, and it degrades the
cross-encoder's judgement because the reranker scores a fragment.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from ragmed.config import ChunkingConfig
from ragmed.tokenization import Tokenizer
from ragmed.types import Chunk, Document

log = logging.getLogger(__name__)

# Split on sentence punctuation followed by whitespace. Requiring the whitespace is
# what keeps "5.5 mg" and "HbA1c 7.0%" intact.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Tokens that end a fragment without ending a sentence. Biomedical prose is full of
# these; splitting on them produces one-word "sentences" that wreck the packer.
_ABBREVIATIONS = {
    "e.g.", "i.e.", "vs.", "cf.", "al.", "approx.", "ca.", "resp.", "etc.",
    "Fig.", "Figs.", "Tab.", "No.", "Nos.", "Dr.", "Prof.", "Inc.", "Ltd.",
    "St.", "Jr.", "Sr.", "p.", "pp.", "vol.", "ed.", "eds.", "min.", "max.",
}
_INITIAL = re.compile(r"^[A-Z]\.$")
_ENUMERATOR = re.compile(r"^\(?\d+[.)]$")


def split_sentences(text: str) -> list[str]:
    """Sentence-split with a merge pass for abbreviations and enumerators."""
    raw = _SENT_SPLIT.split(text.strip())
    out: list[str] = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if out and _is_continuation(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _is_continuation(prev: str) -> bool:
    words = prev.split()
    if not words:
        return True
    tail = words[-1]
    return tail in _ABBREVIATIONS or bool(_INITIAL.match(tail)) or bool(_ENUMERATOR.match(tail))


def _split_oversized(sentence: str, tok: Tokenizer, target: int) -> list[str]:
    """Break a single sentence that is longer than the whole target on its own.

    Rare, but real: some abstracts contain 600-token run-on results sentences packed
    with parenthetical statistics. Packing by word keeps the pieces readable.
    """
    words = sentence.split()
    windows: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for word in words:
        wt = tok.count(word)
        if cur and cur_tokens + wt > target:
            windows.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(word)
        cur_tokens += wt
    if cur:
        windows.append(" ".join(cur))
    return windows


def pack_sentences(
    sentences: Iterable[str],
    tok: Tokenizer,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Greedily pack sentences into windows of at most ``target_tokens``, with a
    trailing overlap carried into the next window."""
    # Overlap is capped at half the target: beyond that, consecutive windows are
    # more duplicate than new content and the index fills with near-copies.
    overlap = max(0, min(overlap_tokens, target_tokens // 2))

    windows: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    for sentence in sentences:
        st = tok.count(sentence)

        if st > target_tokens:
            if cur:
                windows.append(" ".join(cur))
                cur, cur_tokens = [], 0
            windows.extend(_split_oversized(sentence, tok, target_tokens))
            continue

        if cur and cur_tokens + st > target_tokens:
            windows.append(" ".join(cur))
            tail: list[str] = []
            tail_tokens = 0
            # Iterate cur[1:] so the first sentence is always left behind. Carrying
            # the *whole* window forward would re-emit it verbatim on the next flush,
            # which is how an aggressive overlap setting stalls progress. Dropping at
            # least one sentence guarantees every window differs from its predecessor
            # without having to compare content - which must not be done here, since
            # genuinely repeated text in a document is information, not noise.
            for prev in reversed(cur[1:]):
                pt = tok.count(prev)
                if tail_tokens + pt > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += pt
            cur, cur_tokens = tail, tail_tokens

        cur.append(sentence)
        cur_tokens += st

    if cur:
        windows.append(" ".join(cur))

    return windows


def _chunk_structure(doc: Document, cfg: ChunkingConfig, tok: Tokenizer) -> list[tuple[str, str | None]]:
    """Returns (text, section_label) pairs."""
    out: list[tuple[str, str | None]] = []

    # Buffer of consecutive small sections waiting to be merged into one chunk.
    buf_texts: list[str] = []
    buf_headings: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf_texts, buf_headings, buf_tokens
        if buf_texts:
            label = " / ".join(h for h in buf_headings if h) or None
            out.append(("\n\n".join(buf_texts), label))
            buf_texts, buf_headings, buf_tokens = [], [], 0

    for section in doc.sections:
        body = section.text.strip()
        if not body:
            continue
        st = tok.count(body)

        if st > cfg.target_tokens:
            # Oversized section: flush anything pending, then window this section
            # on its own so every piece keeps the correct heading.
            flush()
            for window in pack_sentences(
                split_sentences(body), tok, cfg.target_tokens, cfg.overlap_tokens
            ):
                out.append((window, section.heading))
            continue

        if buf_tokens + st > cfg.target_tokens:
            flush()
        buf_texts.append(body)
        buf_headings.append(section.heading or "")
        buf_tokens += st

    flush()
    return out


def _chunk_fixed(doc: Document, cfg: ChunkingConfig, tok: Tokenizer) -> list[tuple[str, str | None]]:
    # Headings are folded into the text so the fixed strategy is not handed a
    # structural advantage it is supposed to lack.
    body = doc.text
    windows = pack_sentences(split_sentences(body), tok, cfg.target_tokens, cfg.overlap_tokens)
    return [(w, None) for w in windows]


def chunk_document(doc: Document, cfg: ChunkingConfig, tok: Tokenizer) -> list[Chunk]:
    pairs = (
        _chunk_structure(doc, cfg, tok)
        if cfg.strategy == "structure"
        else _chunk_fixed(doc, cfg, tok)
    )

    # Fold a runt tail into its predecessor rather than indexing a fragment that can
    # never carry enough context to answer anything.
    cleaned: list[tuple[str, str | None]] = []
    for text, label in pairs:
        if cleaned and tok.count(text) < cfg.min_tokens:
            prev_text, prev_label = cleaned[-1]
            cleaned[-1] = (f"{prev_text}\n\n{text}", prev_label or label)
        else:
            cleaned.append((text, label))
    if len(cleaned) == 1 and tok.count(cleaned[0][0]) < cfg.min_tokens:
        return []

    chunks: list[Chunk] = []
    for ordinal, (text, label) in enumerate(cleaned):
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(doc.doc_id, ordinal, text),
                doc_id=doc.doc_id,
                ordinal=ordinal,
                text=text,
                token_count=tok.count(text),
                title=doc.title,
                source_type=doc.source_type,
                section=label,
                url=doc.url,
                date=doc.date,
                meta=dict(doc.meta),
            )
        )
    return chunks


def chunk_documents(docs: list[Document], cfg: ChunkingConfig, tok: Tokenizer) -> list[Chunk]:
    chunks: list[Chunk] = []
    dropped = 0
    for doc in docs:
        produced = chunk_document(doc, cfg, tok)
        if not produced:
            dropped += 1
        chunks.extend(produced)

    if chunks:
        sizes = [c.token_count for c in chunks]
        log.info(
            "chunked %d docs -> %d chunks (strategy=%s, target=%d): "
            "mean %.0f tok, min %d, max %d, %d docs dropped as too short",
            len(docs),
            len(chunks),
            cfg.strategy,
            cfg.target_tokens,
            sum(sizes) / len(sizes),
            min(sizes),
            max(sizes),
            dropped,
        )
    return chunks
