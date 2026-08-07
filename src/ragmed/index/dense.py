"""Dense retrieval: bi-encoder embeddings + exact cosine search.

Exact search, not ANN. At this corpus size (tens of thousands of chunks) a normalised
matrix product is sub-millisecond and exact, so adding FAISS or HNSW would trade real
recall for latency this system does not need to save. That is a deliberate call worth
stating: an ANN index is a *third* source of recall loss on top of chunking and the
embedding model, and it would contaminate the ablation - a drop attributable to the
approximate index would look like a drop attributable to the retriever.

The query embedding cache lives here rather than in the service layer because it is
keyed on the thing that actually determines the vector: model, prefix and text.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ragmed.config import DenseConfig

log = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class EmbeddingCache:
    """Thread-safe LRU over query vectors.

    Repeat queries are the norm in an eval loop (every ablation row re-runs the same
    golden set) and common in production. Caching here is what stops `embed_query`
    from dominating the latency table for reasons that have nothing to do with the
    retrieval design.
    """

    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            vec = self._data.get(key)
            if vec is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return vec

    def put(self, key: str, vec: np.ndarray) -> None:
        with self._lock:
            self._data[key] = vec
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    @property
    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._data),
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class Embedder:
    """Lazy-loading wrapper around a sentence-transformers bi-encoder."""

    def __init__(self, cfg: DenseConfig, cache_size: int = 4096):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._model = None
        self.cache = EmbeddingCache(cache_size)

    @property
    def model(self):
        # Loading is deferred so that config parsing, index loading and the offline
        # retrieval metrics never pay for a model they may not use.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s on %s", self.cfg.model, self.device)
            self._model = SentenceTransformer(self.cfg.model, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        prefixed = [self.cfg.passage_prefix + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        key = f"{self.cfg.model}\x00{self.cfg.query_prefix}\x00{query}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vec = self.model.encode(
            [self.cfg.query_prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
        )[0].astype(np.float32)
        self.cache.put(key, vec)
        return vec


class DenseIndex:
    def __init__(self, doc_ids: list[str], matrix: np.ndarray, model_name: str, normalized: bool):
        assert len(doc_ids) == matrix.shape[0], "doc_ids and embedding matrix disagree"
        self.doc_ids = doc_ids
        self.matrix = matrix
        self.model_name = model_name
        self.normalized = normalized

    @property
    def n_docs(self) -> int:
        return self.matrix.shape[0]

    @classmethod
    def build(cls, doc_ids: list[str], texts: list[str], embedder: Embedder) -> DenseIndex:
        matrix = embedder.encode_passages(texts)
        log.info("dense index: %d vectors, dim %d", matrix.shape[0], matrix.shape[1])
        return cls(doc_ids, matrix, embedder.cfg.model, embedder.cfg.normalize)

    def search(self, query_vec: np.ndarray, top_k: int = 50) -> list[tuple[int, float]]:
        if self.n_docs == 0:
            return []
        scores = self.matrix @ query_vec
        if not self.normalized:
            # Fall back to true cosine when vectors were not pre-normalised.
            denom = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(query_vec)
            scores = scores / np.maximum(denom, 1e-9)

        k = min(top_k, self.n_docs)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(int(i), float(scores[i])) for i in idx]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "dense.npy", self.matrix)
        (path / "dense_meta.json").write_text(
            json.dumps(
                {
                    "doc_ids": self.doc_ids,
                    "model_name": self.model_name,
                    "normalized": self.normalized,
                    "dim": int(self.matrix.shape[1]),
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        matrix = np.load(path / "dense.npy")
        meta = json.loads((path / "dense_meta.json").read_text(encoding="utf-8"))
        return cls(meta["doc_ids"], matrix, meta["model_name"], meta["normalized"])
