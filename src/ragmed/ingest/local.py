"""Ingest local guideline documents.

PubMed gives us abstracts; full clinical guidelines mostly live as PDFs and HTML on
society websites, behind layouts that change without warning. Rather than build a
scraper that silently rots, this loader takes whatever you drop into ``data/raw/`` -
markdown, text, or saved HTML - and normalises it into the same ``Document`` shape.

Markdown headings and HTML ``<h1>``-``<h4>`` become sections, so guideline documents
get the same structure-aware chunking that structured abstracts do.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from ragmed.types import Document, Section

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm"}

_MD_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
_HTML_HEADING = re.compile(r"<h[1-4][^>]*>(.*?)</h[1-4]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_DROP = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def _sections_from_markdown(text: str) -> list[Section]:
    matches = list(_MD_HEADING.finditer(text))
    if not matches:
        return [Section(heading=None, text=_clean(text))]

    sections: list[Section] = []
    preamble = _clean(text[: matches[0].start()])
    if preamble:
        sections.append(Section(heading=None, text=preamble))

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _clean(text[start:end])
        if body:
            sections.append(Section(heading=m.group(2).strip(), text=body))
    return sections


def _sections_from_html(raw: str) -> list[Section]:
    raw = _HTML_DROP.sub(" ", raw)
    # Convert headings to markdown so the two paths share one splitter.
    converted = _HTML_HEADING.sub(lambda m: f"\n\n## {_HTML_TAG.sub('', m.group(1))}\n\n", raw)
    stripped = _HTML_TAG.sub(" ", converted)
    return _sections_from_markdown(_clean(html.unescape(stripped)))


def load_local_documents(raw_dir: Path) -> list[Document]:
    if not raw_dir.exists():
        log.info("no local corpus at %s; skipping", raw_dir)
        return []

    docs: list[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("could not read %s: %s", path, exc)
            continue

        if path.suffix.lower() in {".html", ".htm"}:
            sections = _sections_from_html(raw)
        else:
            sections = _sections_from_markdown(raw)

        sections = [s for s in sections if s.text.strip()]
        if not sections:
            log.warning("no extractable text in %s; skipping", path)
            continue

        rel = path.relative_to(raw_dir).as_posix()
        # Prefer a leading markdown H1 as the title, else the filename.
        title = next((s.heading for s in sections if s.heading), path.stem.replace("_", " "))

        docs.append(
            Document(
                doc_id=f"local:{rel}",
                title=title,
                source_type="guideline",
                sections=sections,
                url=path.as_uri(),
                date=None,
                meta={"path": rel, "bytes": path.stat().st_size},
            )
        )

    if docs:
        log.info("loaded %d local documents from %s", len(docs), raw_dir)
    return docs
