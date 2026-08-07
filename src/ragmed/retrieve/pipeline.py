"""The retrieval pipeline.

Every stage is optional and driven by config, because every stage is a row in the
ablation table. Disabling the reranker must produce the dense+BM25 baseline exactly,
with no other behavioural difference - otherwise the table is comparing two systems
that differ in more ways than the label admits.

Stage order: rewrite -> (BM25 || dense) -> fuse -> rerank -> assemble.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ragmed.config import Config
from ragmed.index.dense import Embedder
from ragmed.index.store import CorpusIndex
from ragmed.llm import LLM, NullLLM
from ragmed.retrieve.assemble import assemble_context
from ragmed.retrieve.fusion import fuse
from ragmed.retrieve.rerank import CrossEncoderReranker
from ragmed.retrieve.rewrite import rewrite_query
from ragmed.telemetry import (
    STAGE_ASSEMBLE,
    STAGE_BM25,
    STAGE_DENSE,
    STAGE_EMBED_QUERY,
    STAGE_FUSION,
    STAGE_RERANK,
    STAGE_REWRITE,
    Trace,
)
from ragmed.tokenization import Tokenizer, get_tokenizer
from ragmed.types import Scored

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    query: str
    queries: list[str]
    # Post-fusion, pre-rerank. Kept so failure analysis can ask whether the right
    # chunk was ever a candidate at all - the difference between "retrieval missed
    # it" and "the reranker buried it".
    candidates: list[Scored]
    # The system's final ranking, full depth - post-rerank when reranking is on,
    # the fused list otherwise. Retrieval metrics are measured against this, not
    # against `contexts`: recall@20 on a config whose context holds 5 chunks would
    # otherwise be capped at 5/20 by construction and the ablation would compare
    # context size rather than ranking quality.
    ranked: list[Scored]
    # What actually reaches the prompt, after top-n selection, dedup and budgeting.
    contexts: list[Scored]
    context_text: str
    trace: Trace
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def context_ids(self) -> list[str]:
        return [s.chunk.chunk_id for s in self.contexts]

    @property
    def candidate_ids(self) -> list[str]:
        return [s.chunk.chunk_id for s in self.candidates]

    @property
    def ranked_ids(self) -> list[str]:
        return [s.chunk.chunk_id for s in self.ranked]


class RetrievalPipeline:
    def __init__(
        self,
        cfg: Config,
        index: CorpusIndex,
        embedder: Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: LLM | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        self.cfg = cfg
        self.index = index
        self.llm = llm or NullLLM()
        self.tok = tokenizer or get_tokenizer(cfg.retrieval.dense.model)

        self._embedder = embedder
        self._reranker = reranker

        # Rewriting without a usable LLM degrades to "return the original query",
        # which is correct behaviour at request time but catastrophic during an
        # ablation: the row runs, reports numbers identical to the baseline, and
        # reads as the clean finding "query rewriting does not help" when in fact
        # nothing was ever rewritten. It cost two invalid rows to notice, so the
        # no-op is now loud.
        if cfg.retrieval.rewrite.enabled and not self.llm.available():
            log.warning(
                "retrieval.rewrite.enabled=True but the LLM (%s) is unavailable - "
                "queries will NOT be rewritten. Any metrics from this run are the "
                "no-rewrite baseline and must not be reported as a rewrite result.",
                self.llm.name,
            )
            self.rewrite_active = False
        else:
            self.rewrite_active = cfg.retrieval.rewrite.enabled

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(
                self.cfg.retrieval.dense, self.cfg.service.embedding_cache_size
            )
        return self._embedder

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.cfg.retrieval.rerank)
        return self._reranker

    def retrieve(self, query: str, trace: Trace | None = None) -> RetrievalResult:
        rcfg = self.cfg.retrieval
        trace = trace or Trace()
        trace.meta.setdefault("query", query)

        # --- 1. rewrite -------------------------------------------------------
        queries = [query]
        if rcfg.rewrite.enabled:
            with trace.stage(STAGE_REWRITE) as span:
                queries = rewrite_query(self.llm, query, rcfg.rewrite.max_queries)
                span["n_queries"] = len(queries)
                # Recorded per request so a run's own trace shows whether rewriting
                # actually happened, rather than leaving it to be inferred from a
                # suspiciously round 0.0ms stage timing.
                span["rewrite_active"] = self.rewrite_active

        # --- 2. retrieve ------------------------------------------------------
        # Each (retriever, query) pair contributes its own ranked list. Fusion then
        # treats multi-query retrieval and multi-retriever retrieval uniformly,
        # rather than needing a separate merge step for rewrites.
        rankings: dict[str, list[tuple[int, float]]] = {}

        if rcfg.dense.enabled and self.index.dense is not None:
            with trace.stage(STAGE_EMBED_QUERY) as span:
                query_vectors = [self.embedder.encode_query(q) for q in queries]
                span["n_queries"] = len(queries)
                span["cache_hit_rate"] = self.embedder.cache.stats["hit_rate"]
            with trace.stage(STAGE_DENSE) as span:
                for qi, qvec in enumerate(query_vectors):
                    hits = self.index.dense.search(qvec, rcfg.dense.top_k)
                    rankings[self._source("dense", qi, len(queries))] = hits
                span["n_hits"] = sum(len(v) for k, v in rankings.items() if k.startswith("dense"))

        if rcfg.bm25.enabled and self.index.bm25 is not None:
            with trace.stage(STAGE_BM25) as span:
                for qi, q in enumerate(queries):
                    hits = self.index.bm25.search(q, rcfg.bm25.top_k)
                    rankings[self._source("bm25", qi, len(queries))] = hits
                span["n_hits"] = sum(len(v) for k, v in rankings.items() if k.startswith("bm25"))

        if not rankings:
            raise RuntimeError(
                "no retriever is enabled and available: check retrieval.bm25.enabled / "
                "retrieval.dense.enabled and that the index was built with them"
            )

        # --- 3. fuse ----------------------------------------------------------
        with trace.stage(STAGE_FUSION) as span:
            fused = fuse(rankings, method=rcfg.fusion.method, k=rcfg.fusion.k)
            span["n_candidates"] = len(fused)
            span["method"] = rcfg.fusion.method

        candidates = [
            Scored(
                chunk=self.index.chunks[h.doc_index],
                score=h.score,
                rank=rank,
                stage="fusion",
                components=h.components,
            )
            for rank, h in enumerate(fused, start=1)
        ]

        # --- 4. rerank --------------------------------------------------------
        if rcfg.rerank.enabled and candidates:
            with trace.stage(STAGE_RERANK) as span:
                pool = candidates[: rcfg.rerank.candidates]
                # Score the whole pool and keep it all in ranked order. top_n selects
                # what reaches the context; it must not truncate what the eval can
                # see, or recall@k above top_n becomes unmeasurable.
                scored_pool = self.reranker.rerank(query, [c.chunk for c in pool], top_n=len(pool))
                ranked = []
                for new_rank, (i, score) in enumerate(scored_pool, start=1):
                    src = pool[i]
                    # Preserve the pre-rerank position: a large gap between
                    # fused_rank and rerank_rank is the signal that the cross-encoder
                    # actually did something.
                    components = dict(src.components)
                    components["rerank_score"] = score
                    components["pre_rerank_rank"] = float(src.rank)
                    ranked.append(
                        Scored(
                            chunk=src.chunk,
                            score=score,
                            rank=new_rank,
                            stage="rerank",
                            components=components,
                        )
                    )
                # Anything beyond the rerank pool keeps its fused order underneath,
                # so deep-k metrics still see the tail the reranker never looked at.
                ranked.extend(candidates[rcfg.rerank.candidates :])
                span["n_scored"] = len(pool)
                span["n_kept"] = rcfg.rerank.top_n
        else:
            # Without a reranker the fused order *is* the final ranking, so the two
            # configurations differ in exactly one thing.
            ranked = candidates

        selected = ranked[: rcfg.rerank.top_n]

        # --- 5. assemble ------------------------------------------------------
        with trace.stage(STAGE_ASSEMBLE) as span:
            vectors = self._vectors_for(selected)
            contexts, context_text, stats = assemble_context(
                selected, rcfg.assembly, self.tok, vectors
            )
            span.update(stats)

        return RetrievalResult(
            query=query,
            queries=queries,
            candidates=candidates,
            ranked=ranked,
            contexts=contexts,
            context_text=context_text,
            trace=trace,
            stats=stats,
        )

    @staticmethod
    def _source(name: str, qi: int, n_queries: int) -> str:
        # Keep the plain name when there is one query, so component keys stay
        # readable ("bm25_rank") in the common case.
        return name if n_queries == 1 else f"{name}#{qi}"

    def _vectors_for(self, scored: list[Scored]) -> np.ndarray | None:
        """Embedding rows for a candidate set, used for dedup during assembly."""
        if self.index.dense is None or not scored:
            return None
        rows = []
        for s in scored:
            i = self.index.by_id.get(s.chunk.chunk_id)
            if i is None:
                return None
            rows.append(self.index.dense.matrix[i])
        return np.vstack(rows)
