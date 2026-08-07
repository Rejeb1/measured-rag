"""JSONL persistence for documents, chunks and golden sets.

JSONL rather than a database because every artifact here should be diffable in git
and greppable by hand. When an eval regresses, the first useful question is "what
changed in the corpus?", and `git diff` answers that for free.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ragmed.types import Chunk, Document, GoldenItem


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_documents(path: Path, docs: Iterable[Document]) -> int:
    return write_jsonl(path, (d.to_dict() for d in docs))


def load_documents(path: Path) -> list[Document]:
    return [Document.from_dict(r) for r in read_jsonl(path)]


def save_chunks(path: Path, chunks: Iterable[Chunk]) -> int:
    return write_jsonl(path, (c.to_dict() for c in chunks))


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.from_dict(r) for r in read_jsonl(path)]


def save_golden(path: Path, items: Iterable[GoldenItem]) -> int:
    return write_jsonl(path, (g.to_dict() for g in items))


def load_golden(path: Path) -> list[GoldenItem]:
    return [GoldenItem.from_dict(r) for r in read_jsonl(path)]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
