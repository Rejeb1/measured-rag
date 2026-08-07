"""The ablation matrix.

Each row is one config patch against a single base, so every row differs from the
baseline in exactly the thing its label claims. The output table is the artifact that
goes in the README: it is what turns "we use a hybrid retriever with a reranker" into
"the reranker buys 11 points of NDCG for 180ms p95, and query rewriting buys nothing".

Reporting a component that did *not* help is the point, not an embarrassment. A table
where every row improves is a table that was pruned.

One subtlety dominates this module: **chunking ablations invalidate the golden set.**
Gold labels are chunk ids, and chunk ids are content-addressed, so re-chunking at 256
tokens means not one gold id still exists. Scoring against them would report 0.0
recall for every chunking variant and look like a catastrophic result rather than a
broken measurement. `remap_golden` re-projects gold labels onto the new chunking by
text containment before those rows are scored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ragmed.config import Config
from ragmed.eval.runner import EvalRun, run_eval
from ragmed.index.bm25 import tokenize
from ragmed.index.dense import Embedder
from ragmed.index.store import CorpusIndex
from ragmed.ingest.chunking import chunk_documents
from ragmed.llm import LLM, NullLLM
from ragmed.retrieve.rerank import CrossEncoderReranker
from ragmed.tokenization import get_tokenizer
from ragmed.types import Chunk, Document, GoldenItem

log = logging.getLogger(__name__)


@dataclass
class AblationRow:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def config(self, base: Config) -> Config:
        return base.patch(self.overrides) if self.overrides else base


# The default matrix. Order matters only for readability of the emitted table.
DEFAULT_ROWS: list[AblationRow] = [
    AblationRow(
        "Dense only",
        {"retrieval.bm25.enabled": False, "retrieval.rerank.enabled": False},
        "bi-encoder alone; no lexical matching",
    ),
    AblationRow(
        "BM25 only",
        {"retrieval.dense.enabled": False, "retrieval.rerank.enabled": False},
        "lexical alone; no semantic matching",
    ),
    AblationRow(
        "Hybrid (RRF)",
        {"retrieval.rerank.enabled": False},
        "both retrievers, rank fusion",
    ),
    AblationRow(
        "Hybrid (normalized-sum)",
        {"retrieval.rerank.enabled": False, "retrieval.fusion.method": "normalized_sum"},
        "control: score fusion instead of rank fusion",
    ),
    AblationRow("Hybrid + reranker", {}, "the default configuration"),
    AblationRow(
        "Hybrid + reranker + query rewrite",
        {"retrieval.rewrite.enabled": True},
        "adds an LLM round trip to every query",
    ),
    AblationRow(
        "Hybrid + reranker, top-20 pool",
        {
            "retrieval.rerank.candidates": 20,
            "retrieval.bm25.top_k": 20,
            "retrieval.dense.top_k": 20,
        },
        "narrower candidate pool before reranking",
    ),
    AblationRow(
        "Hybrid + reranker, top-100 pool",
        {
            "retrieval.rerank.candidates": 100,
            "retrieval.bm25.top_k": 100,
            "retrieval.dense.top_k": 100,
        },
        "wider candidate pool before reranking",
    ),
    AblationRow(
        "Sequential context order",
        {"retrieval.assembly.ordering": "sequential"},
        "control: does edge-ordering matter?",
    ),
]

# Rows that change the chunking, and therefore require a rebuilt index and a remapped
# golden set. Kept separate because they are an order of magnitude more expensive.
CHUNKING_ROWS: list[AblationRow] = [
    AblationRow("Chunk 256 (structure)", {"chunking.target_tokens": 256}),
    AblationRow("Chunk 512 (structure)", {"chunking.target_tokens": 512}),
    AblationRow("Chunk 1024 (structure)", {"chunking.target_tokens": 1024}),
    AblationRow(
        "Chunk 512 (fixed-size)",
        {"chunking.strategy": "fixed", "chunking.target_tokens": 512},
        "ignores section boundaries",
    ),
]


def remap_golden(
    golden: list[GoldenItem],
    old_chunks: list[Chunk],
    new_chunks: list[Chunk],
    containment: float = 0.6,
) -> tuple[list[GoldenItem], dict[str, int]]:
    """Re-project gold chunk ids onto a different chunking.

    A new chunk inherits a gold label when it contains at least ``containment`` of the
    original gold chunk's tokens. Several new chunks can inherit one label - that is
    correct, since a 1024-token gold chunk legitimately becomes two 512-token ones and
    retrieving either should count.

    Items whose gold text cannot be located in the new chunking are dropped, and the
    count is reported. Silently keeping them would understate every chunking variant.
    """
    old_by_id = {c.chunk_id: c for c in old_chunks}
    by_doc: dict[str, list[Chunk]] = {}
    for c in new_chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    stats = {"remapped": 0, "dropped": 0, "unanswerable_passthrough": 0, "expanded": 0}
    out: list[GoldenItem] = []

    for item in golden:
        if not item.is_answerable:
            # No gold chunks to remap; these carry over untouched.
            stats["unanswerable_passthrough"] += 1
            out.append(item)
            continue

        new_ids: list[str] = []
        for gid in item.gold_chunk_ids:
            old = old_by_id.get(gid)
            if old is None:
                continue
            gold_tokens = set(tokenize(old.text))
            if not gold_tokens:
                continue
            for cand in by_doc.get(old.doc_id, []):
                overlap = len(gold_tokens & set(tokenize(cand.text))) / len(gold_tokens)
                if overlap >= containment:
                    new_ids.append(cand.chunk_id)

        new_ids = list(dict.fromkeys(new_ids))
        if not new_ids:
            stats["dropped"] += 1
            continue
        if len(new_ids) > len(item.gold_chunk_ids):
            stats["expanded"] += 1
        stats["remapped"] += 1
        out.append(
            GoldenItem(
                qid=item.qid,
                question=item.question,
                question_type=item.question_type,
                gold_chunk_ids=new_ids,
                answer=item.answer,
                provenance={**item.provenance, "remapped_from": item.gold_chunk_ids},
            )
        )

    if stats["dropped"]:
        log.warning(
            "remapping golden set: %d items dropped (gold text not locatable in the "
            "new chunking); metrics for this row cover %d items",
            stats["dropped"], len(out),
        )
    return out, stats


def run_ablation(
    base_cfg: Config,
    index: CorpusIndex,
    golden: list[GoldenItem],
    rows: list[AblationRow] | None = None,
    with_generation: bool = False,
    llm: LLM | None = None,
    judge: LLM | None = None,
) -> list[EvalRun]:
    """Run rows that share the base index (no re-chunking)."""
    rows = rows if rows is not None else DEFAULT_ROWS
    llm = llm or NullLLM()
    judge = judge or NullLLM()

    # Loading the encoder and cross-encoder dominates wall-clock on CPU, so they are
    # built once and shared across every row that can legitimately reuse them.
    embedder = Embedder(base_cfg.retrieval.dense, base_cfg.service.embedding_cache_size)
    reranker = CrossEncoderReranker(base_cfg.retrieval.rerank)

    runs: list[EvalRun] = []
    for row in rows:
        log.info("--- ablation row: %s ---", row.label)
        cfg = row.config(base_cfg)
        runs.append(
            run_eval(
                cfg,
                index,
                golden,
                label=row.label,
                overrides=row.overrides,
                with_generation=with_generation,
                llm=llm,
                judge=judge,
                embedder=embedder,
                reranker=reranker,
            )
        )
    return runs


def run_chunking_ablation(
    base_cfg: Config,
    documents: list[Document],
    base_chunks: list[Chunk],
    golden: list[GoldenItem],
    rows: list[AblationRow] | None = None,
    with_generation: bool = False,
    llm: LLM | None = None,
    judge: LLM | None = None,
) -> list[EvalRun]:
    """Run rows that change chunking: rebuild the index and remap the golden set."""
    rows = rows if rows is not None else CHUNKING_ROWS
    llm = llm or NullLLM()
    judge = judge or NullLLM()

    tok = get_tokenizer(base_cfg.retrieval.dense.model)
    embedder = Embedder(base_cfg.retrieval.dense, base_cfg.service.embedding_cache_size)
    reranker = CrossEncoderReranker(base_cfg.retrieval.rerank)

    runs: list[EvalRun] = []
    for row in rows:
        log.info("--- chunking ablation row: %s (re-indexing) ---", row.label)
        cfg = row.config(base_cfg)

        new_chunks = chunk_documents(documents, cfg.chunking, tok)
        new_index = CorpusIndex.build(new_chunks, cfg, embedder=embedder)
        remapped, stats = remap_golden(golden, base_chunks, new_chunks)
        log.info("golden set remapped: %s", stats)

        run = run_eval(
            cfg,
            new_index,
            remapped,
            label=row.label,
            overrides={**row.overrides, "_remap": stats},
            with_generation=with_generation,
            llm=llm,
            judge=judge,
            embedder=embedder,
            reranker=reranker,
        )
        runs.append(run)
    return runs


# --- table rendering -----------------------------------------------------------


def _fmt(value: float | None, places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def render_markdown_table(runs: list[EvalRun], k: int = 10) -> str:
    """The ablation table, ready to paste into the README."""
    has_gen = any(r.generation is not None for r in runs)

    header = ["Config", f"Recall@{k}", f"NDCG@{k}", "MRR"]
    if has_gen:
        header += ["Faithfulness", "Ctx precision"]
    header += ["p50 (ms)", "p95 (ms)"]

    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for run in runs:
        lat = run.latency.get("TOTAL_RETRIEVAL", {})
        cells = [
            run.label,
            _fmt(run.retrieval.recall_at_k.get(k)),
            _fmt(run.retrieval.ndcg),
            _fmt(run.retrieval.mrr),
        ]
        if has_gen:
            g = run.generation
            cells += [
                _fmt(g.faithfulness if g else None),
                _fmt(g.context_precision if g else None),
            ]
        cells += [_fmt(lat.get("p50_ms"), 0), _fmt(lat.get("p95_ms"), 0)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def render_latency_table(run: EvalRun) -> str:
    """Per-stage latency budget for one configuration."""
    lines = ["| Stage | p50 (ms) | p95 (ms) | mean (ms) | % of retrieval p50 |", "|---|---|---|---|---|"]
    total_p50 = run.latency.get("TOTAL_RETRIEVAL", {}).get("p50_ms", 0.0) or 0.0

    order = [
        "query_rewrite", "embed_query", "bm25_search", "dense_search",
        "fusion", "rerank", "assemble_context", "generate",
        "TOTAL_RETRIEVAL", "TOTAL",
    ]
    for stage in order:
        s = run.latency.get(stage)
        if not s:
            continue
        share = (
            f"{100 * s['p50_ms'] / total_p50:.1f}%"
            if total_p50 and stage not in ("TOTAL", "TOTAL_RETRIEVAL", "generate")
            else "—"
        )
        label = f"**{stage}**" if stage.startswith("TOTAL") else stage
        lines.append(
            f"| {label} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} | {s['mean_ms']:.1f} | {share} |"
        )
    return "\n".join(lines)


def render_question_type_table(run: EvalRun, k: int = 10) -> str:
    lines = [f"| Question type | n | Recall@{k} | Hit rate@{k} | MRR | NDCG |", "|---|---|---|---|---|---|"]
    for qtype, m in run.by_question_type.items():
        lines.append(
            f"| {qtype} | {int(m['n'])} | {m[f'recall@{k}']:.3f} | "
            f"{m[f'hit_rate@{k}']:.3f} | {m['mrr']:.3f} | {m['ndcg']:.3f} |"
        )
    return "\n".join(lines)
