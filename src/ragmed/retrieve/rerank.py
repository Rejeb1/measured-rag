"""Cross-encoder reranking.

The bi-encoder that produced the index never saw a query and a passage together - it
encoded each independently, so the similarity it reports is between two summaries of
meaning rather than a judgement about whether this passage answers this question. A
cross-encoder reads the pair jointly and is far more accurate, but it is O(candidates)
model calls per query, so it cannot run over a corpus.

Hence retrieve-wide-then-rerank-narrow: 50 cheap candidates, 5 expensive survivors.
This stage is usually the single biggest quality jump in the ablation *and* the single
biggest latency cost, which is exactly the tradeoff the table exists to make visible.
"""

from __future__ import annotations

import logging

from ragmed.config import RerankConfig
from ragmed.index.dense import resolve_device
from ragmed.types import Chunk

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading cross-encoder %s on %s", self.cfg.model, self.device)
            self._model = CrossEncoder(self.cfg.model, device=self.device)
        return self._model

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        # The section heading is included because it carries real signal in this
        # corpus - "Results" vs "Background" changes whether a passage states a
        # finding or merely motivates one.
        pairs = [
            (query, f"{c.title}\n{c.section or ''}\n{c.text}".strip())
            for c in chunks
        ]
        scores = self.model.predict(
            pairs,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def rerank(self, query: str, chunks: list[Chunk], top_n: int | None = None) -> list[tuple[int, float]]:
        """Return (index_into_chunks, score) pairs, best first."""
        scores = self.score(query, chunks)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        limit = top_n if top_n is not None else self.cfg.top_n
        return [(i, scores[i]) for i in order[:limit]]
