"""The index bundle: chunks plus both retrievers, saved and loaded as one unit.

Keeping the chunk list and the two indexes together is what guarantees they cannot
drift apart. A BM25 index built over one chunking and a dense index built over another
would still "work" - every query would return results, every number in the ablation
table would be wrong, and nothing would raise. The fingerprint check and the row-order
assertions exist to make that failure loud.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ragmed.config import Config
from ragmed.index.bm25 import BM25Index
from ragmed.index.dense import DenseIndex, Embedder
from ragmed.store import load_chunks, read_json, save_chunks, write_json
from ragmed.types import Chunk

log = logging.getLogger(__name__)


class IndexMismatchError(RuntimeError):
    """Raised when a persisted index does not match the current config."""


class CorpusIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Index | None,
        dense: DenseIndex | None,
        fingerprint: str,
    ):
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.fingerprint = fingerprint
        self.by_id = {c.chunk_id: i for i, c in enumerate(chunks)}
        self._validate()

    def _validate(self) -> None:
        if len(self.by_id) != len(self.chunks):
            raise IndexMismatchError(
                f"duplicate chunk_ids: {len(self.chunks)} chunks but "
                f"{len(self.by_id)} unique ids"
            )
        ids = [c.chunk_id for c in self.chunks]
        # Both indexes address chunks positionally, so their row order must match the
        # chunk list exactly - not merely contain the same ids.
        if self.bm25 is not None and self.bm25.doc_ids != ids:
            raise IndexMismatchError("BM25 index row order does not match the chunk list")
        if self.dense is not None and self.dense.doc_ids != ids:
            raise IndexMismatchError("dense index row order does not match the chunk list")

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        i = self.by_id.get(chunk_id)
        return self.chunks[i] if i is not None else None

    @classmethod
    def build(cls, chunks: list[Chunk], cfg: Config, embedder: Embedder | None = None) -> CorpusIndex:
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]

        bm25 = None
        if cfg.retrieval.bm25.enabled:
            bm25 = BM25Index.build(
                ids,
                texts,
                k1=cfg.retrieval.bm25.k1,
                b=cfg.retrieval.bm25.b,
                use_stopwords=cfg.retrieval.bm25.use_stopwords,
            )

        dense = None
        if cfg.retrieval.dense.enabled:
            embedder = embedder or Embedder(cfg.retrieval.dense, cfg.service.embedding_cache_size)
            dense = DenseIndex.build(ids, texts, embedder)

        return cls(chunks, bm25, dense, cfg.fingerprint())

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_chunks(path / "chunks.jsonl", self.chunks)
        if self.bm25 is not None:
            self.bm25.save(path)
        if self.dense is not None:
            self.dense.save(path)
        write_json(
            path / "index_meta.json",
            {
                "fingerprint": self.fingerprint,
                "n_chunks": len(self.chunks),
                "has_bm25": self.bm25 is not None,
                "has_dense": self.dense is not None,
            },
        )
        log.info("saved index (%d chunks, fingerprint %s) to %s", len(self.chunks), self.fingerprint, path)

    @classmethod
    def load(cls, path: Path, cfg: Config | None = None, strict: bool = True) -> CorpusIndex:
        meta_path = path / "index_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no index at {path}. Build one first: `ragmed index`"
            )
        meta = read_json(meta_path)
        chunks = load_chunks(path / "chunks.jsonl")

        bm25 = BM25Index.load(path) if meta.get("has_bm25") else None
        dense = DenseIndex.load(path) if meta.get("has_dense") else None

        if cfg is not None:
            current = cfg.fingerprint()
            if current != meta["fingerprint"]:
                msg = (
                    f"index fingerprint {meta['fingerprint']} does not match the current "
                    f"config ({current}). The chunking strategy or embedding model changed; "
                    f"re-run `ragmed index` or the evals will measure a stale corpus."
                )
                if strict:
                    raise IndexMismatchError(msg)
                log.warning(msg)

        return cls(chunks, bm25, dense, meta["fingerprint"])

    def stats(self) -> dict[str, object]:
        by_source: dict[str, int] = {}
        for c in self.chunks:
            by_source[c.source_type] = by_source.get(c.source_type, 0) + 1
        token_counts = [c.token_count for c in self.chunks]
        return {
            "n_chunks": len(self.chunks),
            "n_documents": len({c.doc_id for c in self.chunks}),
            "by_source": by_source,
            "mean_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "bm25_terms": len(self.bm25.terms) if self.bm25 else 0,
            "dense_dim": int(self.dense.matrix.shape[1]) if self.dense else 0,
            "fingerprint": self.fingerprint,
        }
