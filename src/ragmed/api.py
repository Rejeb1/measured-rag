"""FastAPI service.

Three production details that are easy to skip and expensive to retrofit:

*Models load once, at startup.* Loading the bi-encoder and cross-encoder per request
would dominate latency by two orders of magnitude and make the numbers in the README
unrelated to what the service actually does.

*Retrieval runs in a worker thread.* Embedding, BM25 scoring and cross-encoder
inference are synchronous CPU work. Running them directly in an async handler blocks
the event loop, so a second concurrent request waits for the first to finish - the
service would look fine under `curl` and collapse under two users.

*Every response carries its trace.* Per-stage timings are returned with the answer and
logged against a request-scoped trace id, so a slow request in production can be
attributed to a stage without reproducing it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ragmed.config import Config
from ragmed.generate import answer_question, stream_answer
from ragmed.index.dense import Embedder
from ragmed.index.store import CorpusIndex
from ragmed.llm import LLMError, generation_llm
from ragmed.retrieve.pipeline import RetrievalPipeline
from ragmed.retrieve.rerank import CrossEncoderReranker
from ragmed.telemetry import Trace, configure_logging, trace_context
from ragmed.telemetry import log as slog

log = logging.getLogger("ragmed.api")

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = Config.load()
    configure_logging(cfg.service.log_level)
    log.info("loading index from %s", cfg.paths.index_dir)

    index = CorpusIndex.load(cfg.paths.index_dir, cfg, strict=False)
    embedder = Embedder(cfg.retrieval.dense, cfg.service.embedding_cache_size)
    reranker = CrossEncoderReranker(cfg.retrieval.rerank)
    llm = generation_llm(cfg)

    # Force the weights in now rather than paying for them on the first user request.
    if cfg.retrieval.dense.enabled:
        embedder.encode_query("warmup")
    if cfg.retrieval.rerank.enabled:
        reranker.score("warmup", index.chunks[:1])

    STATE.update(
        cfg=cfg,
        index=index,
        pipeline=RetrievalPipeline(cfg, index, embedder=embedder, reranker=reranker, llm=llm),
        embedder=embedder,
        llm=llm,
        started_at=time.time(),
        requests=0,
    )
    log.info("ready: %d chunks, llm=%s", len(index), llm.available())
    yield
    STATE.clear()


app = FastAPI(
    title="ragmed",
    description="Hybrid-retrieval RAG over clinical literature, with a measured eval layer.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_n: int | None = Field(default=None, ge=1, le=20)
    include_context: bool = False


class SourceOut(BaseModel):
    rank: int
    score: float
    citation: str
    title: str
    section: str | None
    url: str | None
    date: str | None
    excerpt: str


class AskResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    abstained: bool
    citations: list[str]
    sources: list[SourceOut]
    timings_ms: dict[str, float]
    total_ms: float
    context: str | None = None


def _sources(result: Any, limit: int | None = None) -> list[SourceOut]:
    out = []
    for s in result.contexts[:limit] if limit else result.contexts:
        out.append(
            SourceOut(
                rank=s.rank,
                score=round(s.score, 5),
                citation=s.chunk.citation,
                title=s.chunk.title,
                section=s.chunk.section,
                url=s.chunk.url,
                date=s.chunk.date,
                excerpt=s.chunk.text[:400],
            )
        )
    return out


@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace = Trace()
    request.state.trace = trace
    with trace_context(trace):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["x-trace-id"] = trace.trace_id
        slog(
            log,
            logging.INFO,
            "request",
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed, 2),
        )
        return response


@app.get("/health")
async def health() -> dict[str, Any]:
    if not STATE:
        raise HTTPException(status_code=503, detail="service still starting")
    index: CorpusIndex = STATE["index"]
    llm = STATE["llm"]
    return {
        "status": "ok",
        "uptime_s": round(time.time() - STATE["started_at"], 1),
        "index": index.stats(),
        "llm": {"name": llm.name, "available": llm.available()},
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    if not STATE:
        raise HTTPException(status_code=503, detail="service still starting")
    embedder: Embedder = STATE["embedder"]
    return {
        "requests": STATE["requests"],
        "embedding_cache": embedder.cache.stats,
        "uptime_s": round(time.time() - STATE["started_at"], 1),
    }


@app.post("/retrieve")
async def retrieve(req: AskRequest, request: Request) -> dict[str, Any]:
    """Retrieval only - no generation. Useful for inspecting ranking behaviour."""
    if not STATE:
        raise HTTPException(status_code=503, detail="service still starting")
    STATE["requests"] += 1
    pipeline: RetrievalPipeline = STATE["pipeline"]
    trace: Trace = request.state.trace

    result = await run_in_threadpool(pipeline.retrieve, req.question, trace)
    return {
        "trace_id": trace.trace_id,
        "question": req.question,
        "queries": result.queries,
        "sources": [s.model_dump() for s in _sources(result, req.top_n)],
        "timings_ms": {s.name: round(s.ms, 2) for s in trace.spans},
        "total_ms": round(trace.total_ms, 2),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    if not STATE:
        raise HTTPException(status_code=503, detail="service still starting")
    STATE["requests"] += 1
    cfg: Config = STATE["cfg"]
    pipeline: RetrievalPipeline = STATE["pipeline"]
    llm = STATE["llm"]
    trace: Trace = request.state.trace

    result = await run_in_threadpool(pipeline.retrieve, req.question, trace)

    if not llm.available():
        raise HTTPException(
            status_code=503,
            detail=(
                f"no LLM backend available ({llm.name}). Retrieval works; start Ollama "
                f"and pull {cfg.generation.model} to enable answers."
            ),
        )

    try:
        with trace.stage("generate"):
            answer = await run_in_threadpool(
                answer_question, llm, req.question, result.context_text,
                result.contexts, cfg.generation,
            )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc

    return AskResponse(
        trace_id=trace.trace_id,
        question=req.question,
        answer=answer.text,
        abstained=answer.abstained,
        citations=answer.citations,
        sources=_sources(result, req.top_n),
        timings_ms={s.name: round(s.ms, 2) for s in trace.spans},
        total_ms=round(trace.total_ms, 2),
        context=result.context_text if req.include_context else None,
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
    """Server-sent events: sources first, then answer tokens.

    Sources are emitted before the first token so the UI can render citations while
    generation is still running - on CPU that gap is seconds, not milliseconds.
    """
    if not STATE:
        raise HTTPException(status_code=503, detail="service still starting")
    STATE["requests"] += 1
    pipeline: RetrievalPipeline = STATE["pipeline"]
    llm = STATE["llm"]
    trace: Trace = request.state.trace

    result = await run_in_threadpool(pipeline.retrieve, req.question, trace)
    if not llm.available():
        raise HTTPException(status_code=503, detail=f"no LLM backend available ({llm.name})")

    import json

    async def events() -> AsyncIterator[str]:
        payload = {
            "trace_id": trace.trace_id,
            "sources": [s.model_dump() for s in _sources(result, req.top_n)],
            "retrieval_ms": round(trace.retrieval_ms, 2),
        }
        yield f"event: sources\ndata: {json.dumps(payload)}\n\n"

        # The Ollama stream is a blocking generator; draining it in a thread keeps
        # the event loop free to serve other connections.
        def drain() -> list[str]:
            return list(stream_answer(llm, req.question, result.context_text))

        try:
            for piece in await run_in_threadpool(drain):
                yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"
        except LLMError as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return

        yield f"event: done\ndata: {json.dumps({'total_ms': round(trace.total_ms, 2)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    trace = getattr(request.state, "trace", None)
    tid = trace.trace_id if trace else "unknown"
    log.exception("unhandled error on %s (trace %s)", request.url.path, tid)
    # The trace id goes to the client so a user-reported failure can be found in logs.
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "trace_id": tid},
    )
