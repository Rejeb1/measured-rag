"""Per-stage latency instrumentation and structured logging.

The latency budget is a deliverable, not a debugging aid, so timing is built into the
pipeline rather than bolted on. Every request carries a ``Trace``; every stage that
costs measurable time opens a span. The ablation runner aggregates those traces into
p50/p95 per stage, which is what turns "the reranker is slow" into "reranking costs
180ms p95 and buys 11 points of NDCG".
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Propagates the current trace id into log records without threading it through
# every function signature.
_current_trace_id: ContextVar[str | None] = ContextVar("ragmed_trace_id", default=None)

# Canonical stage names. Keeping them in one place stops the latency table from
# fragmenting into "rerank" / "reranking" / "cross_encoder" across call sites.
STAGE_REWRITE = "query_rewrite"
STAGE_EMBED_QUERY = "embed_query"
STAGE_BM25 = "bm25_search"
STAGE_DENSE = "dense_search"
STAGE_FUSION = "fusion"
STAGE_RERANK = "rerank"
STAGE_ASSEMBLE = "assemble_context"
STAGE_GENERATE = "generate"

RETRIEVAL_STAGES = (
    STAGE_REWRITE,
    STAGE_EMBED_QUERY,
    STAGE_BM25,
    STAGE_DENSE,
    STAGE_FUSION,
    STAGE_RERANK,
    STAGE_ASSEMBLE,
)


@dataclass(slots=True)
class Span:
    name: str
    ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans: list[Span] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[dict[str, Any]]:
        """Time a stage. The yielded dict can be mutated to attach late-known metadata
        (candidate counts, cache hits) that is only available once the stage has run."""
        extra: dict[str, Any] = dict(meta)
        t0 = time.perf_counter()
        try:
            yield extra
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.spans.append(Span(name=name, ms=elapsed, meta=extra))

    def ms(self, name: str) -> float:
        """Total milliseconds spent in a stage (summed if it ran more than once)."""
        return sum(s.ms for s in self.spans if s.name == name)

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.spans)

    @property
    def retrieval_ms(self) -> float:
        """Everything except generation - the part this project actually controls."""
        return sum(s.ms for s in self.spans if s.name in RETRIEVAL_STAGES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_ms": round(self.total_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "spans": [{"name": s.name, "ms": round(s.ms, 2), **s.meta} for s in self.spans],
            "meta": self.meta,
        }


@contextmanager
def trace_context(trace: Trace) -> Iterator[Trace]:
    token = _current_trace_id.set(trace.trace_id)
    try:
        yield trace
    finally:
        _current_trace_id.reset(token)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    # "lower" avoids inventing a latency value that was never observed, which matters
    # when reporting p95 off a small eval set.
    return float(np.percentile(np.asarray(values, dtype=float), q, method="lower"))


def aggregate_latency(traces: list[Trace]) -> dict[str, dict[str, float]]:
    """Roll a set of traces into per-stage p50/p95/mean.

    Stages that did not run in a given config are absent rather than zero, so a
    disabled reranker shows up as a missing row instead of a misleading 0ms.
    """
    by_stage: dict[str, list[float]] = {}
    for tr in traces:
        seen: dict[str, float] = {}
        for span in tr.spans:
            seen[span.name] = seen.get(span.name, 0.0) + span.ms
        for name, ms in seen.items():
            by_stage.setdefault(name, []).append(ms)

    out: dict[str, dict[str, float]] = {}
    for name, values in by_stage.items():
        out[name] = {
            "n": float(len(values)),
            "mean_ms": round(float(np.mean(values)), 2),
            "p50_ms": round(percentile(values, 50), 2),
            "p95_ms": round(percentile(values, 95), 2),
        }

    totals = [tr.total_ms for tr in traces]
    retrieval = [tr.retrieval_ms for tr in traces]
    if totals:
        out["TOTAL"] = {
            "n": float(len(totals)),
            "mean_ms": round(float(np.mean(totals)), 2),
            "p50_ms": round(percentile(totals, 50), 2),
            "p95_ms": round(percentile(totals, 95), 2),
        }
        out["TOTAL_RETRIEVAL"] = {
            "n": float(len(retrieval)),
            "mean_ms": round(float(np.mean(retrieval)), 2),
            "p50_ms": round(percentile(retrieval, 50), 2),
            "p95_ms": round(percentile(retrieval, 95), 2),
        }
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        tid = _current_trace_id.get()
        if tid:
            payload["trace_id"] = tid
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # These are noisy at INFO and drown out the pipeline's own logs.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})
