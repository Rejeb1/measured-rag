"""Command line interface.

The pipeline is a sequence of durable artifacts, and each command produces one:

    ingest  -> data/corpus/documents.jsonl
    index   -> data/index/{chunks.jsonl, bm25.*, dense.*}
    golden  -> data/golden/golden_set.jsonl
    eval    -> runs/<label>/{metrics.json, records.jsonl, report.md}
    ablate  -> runs/ablation/{results.json, tables.md}

Every stage can be re-run independently, which matters because they differ in cost by
orders of magnitude: indexing is minutes, a generation eval on CPU is a coffee break.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from ragmed.config import Config
from ragmed.store import (
    load_chunks,
    load_documents,
    load_golden,
    save_documents,
    save_golden,
    write_json,
    write_jsonl,
)
from ragmed.telemetry import configure_logging

log = logging.getLogger("ragmed")


def _load_cfg(args: argparse.Namespace) -> Config:
    overrides = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        # Let YAML parse the scalar so --set retrieval.rerank.enabled=false
        # produces a bool rather than the string "false".
        import yaml

        overrides[key.strip()] = yaml.safe_load(raw)
    cfg = Config.load(args.config, overrides)
    cfg.paths.ensure()
    return cfg


# --- commands -------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    from ragmed.ingest.local import load_local_documents
    from ragmed.ingest.pubmed import fetch_corpus

    cfg = _load_cfg(args)
    docs = []
    if not args.local_only:
        docs.extend(fetch_corpus(cfg.corpus))
    docs.extend(load_local_documents(cfg.paths.data_dir / "raw"))

    if not docs:
        log.error("no documents ingested")
        return 1

    out = cfg.paths.corpus_dir / "documents.jsonl"
    n = save_documents(out, docs)
    by_source: dict[str, int] = {}
    for d in docs:
        by_source[d.source_type] = by_source.get(d.source_type, 0) + 1
    log.info("wrote %d documents to %s (%s)", n, out, by_source)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from ragmed.index.store import CorpusIndex
    from ragmed.ingest.chunking import chunk_documents
    from ragmed.tokenization import get_tokenizer

    cfg = _load_cfg(args)
    docs = load_documents(cfg.paths.corpus_dir / "documents.jsonl")
    if not docs:
        log.error("no corpus found. Run `ragmed ingest` first.")
        return 1

    tok = get_tokenizer(cfg.retrieval.dense.model)
    if not tok.is_exact:
        log.warning("token counts are approximate; chunk sizes will not be exact")

    chunks = chunk_documents(docs, cfg.chunking, tok)
    index = CorpusIndex.build(chunks, cfg)
    index.save(cfg.paths.index_dir)
    write_json(cfg.paths.index_dir / "stats.json", index.stats())
    print(f"indexed {len(chunks)} chunks from {len(docs)} documents")
    for k, v in index.stats().items():
        print(f"  {k}: {v}")
    return 0


def cmd_golden(args: argparse.Namespace) -> int:
    from ragmed.eval.build_golden import BuildSpec, build_golden_set
    from ragmed.index.store import CorpusIndex
    from ragmed.llm import generation_llm

    cfg = _load_cfg(args)
    index = CorpusIndex.load(cfg.paths.index_dir, cfg, strict=not args.allow_stale_index)
    llm = generation_llm(cfg)

    if not llm.available():
        log.error(
            "golden-set construction needs a working LLM. Install Ollama "
            "(https://ollama.com), run `ollama pull %s`, then retry.",
            cfg.generation.model,
        )
        return 1

    spec = BuildSpec(
        n_factoid=args.factoid,
        n_multi_hop=args.multi_hop,
        n_aggregation=args.aggregation,
        n_unanswerable=args.unanswerable,
        seed=args.seed,
    )
    items, report = build_golden_set(llm, index, spec)

    out = cfg.paths.golden_dir / "golden_set.jsonl"
    save_golden(out, items)
    write_json(cfg.paths.golden_dir / "build_report.json", report.to_dict())

    print(f"\nwrote {len(items)} questions to {out}")
    print(f"rejection rate: {report.rejection_rate:.0%} ({report.rejected}/{report.generated})")
    print("\nTop rejection reasons:")
    for reason, n in list(report.rejection_reasons.items())[:8]:
        print(f"  {n:3d}  {reason}")
    print(
        "\nThis set is machine-generated and machine-screened. Review it by hand "
        "before trusting any number measured against it."
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ragmed.eval.ablation import (
        render_latency_table,
        render_question_type_table,
    )
    from ragmed.eval.failure_analysis import analyse, render_failure_table
    from ragmed.eval.judge_validation import make_labeling_template
    from ragmed.eval.runner import check_gates, run_eval
    from ragmed.index.store import CorpusIndex
    from ragmed.llm import generation_llm, judge_llm

    cfg = _load_cfg(args)
    index = CorpusIndex.load(cfg.paths.index_dir, cfg, strict=not args.allow_stale_index)
    golden = load_golden(cfg.eval.golden_set)
    if not golden:
        log.error("no golden set at %s. Run `ragmed golden` first.", cfg.eval.golden_set)
        return 1
    if args.limit:
        golden = golden[: args.limit]

    # Also needed when rewriting is on, even for a retrieval-only run - otherwise the
    # rewrite stage silently no-ops and reports the baseline. Same trap as in `ablate`.
    needs_llm = args.generation or cfg.retrieval.rewrite.enabled
    llm = generation_llm(cfg) if needs_llm else None
    judge = judge_llm(cfg) if args.generation else None

    run = run_eval(
        cfg, index, golden, label=args.label, with_generation=args.generation,
        llm=llm, judge=judge,
    )

    out_dir = cfg.paths.runs_dir / args.label
    write_json(out_dir / "metrics.json", run.to_dict())
    # ItemRecord and ItemGenerationScores use slots=True, so asdict() rather than
    # __dict__, which does not exist on a slotted dataclass.
    write_jsonl(out_dir / "records.jsonl", [asdict(r) for r in run.records])

    report = analyse(run, top_n=args.worst)
    write_json(out_dir / "failures.json", report.to_dict())

    if run.generation:
        # validate-judge reads these back; without them the judge can never be
        # checked against human labels, which is the one thing that makes its
        # scores citable.
        write_jsonl(
            out_dir / "generation_items.jsonl", [asdict(s) for s in run.generation.per_item]
        )
        template = make_labeling_template(run.generation.per_item, run.records)
        write_jsonl(out_dir / "label_template.jsonl", template)

    md = [
        f"# Eval run: {args.label}\n",
        f"Golden set: {len(golden)} questions "
        f"({run.retrieval.n_evaluated} answerable, {run.retrieval.n_skipped} unanswerable)\n",
        "## Retrieval\n",
        "```json", str(run.retrieval.to_dict()), "```\n",
        "## By question type\n", render_question_type_table(run, cfg.eval.ndcg_k), "\n",
        "## Latency budget\n", render_latency_table(run), "\n",
        "## Failure analysis\n", render_failure_table(report), "\n",
    ]
    if run.generation:
        md += ["## Generation\n", "```json", str(run.generation.to_dict()), "```\n"]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n=== {args.label} ===")
    for k in sorted(run.retrieval.recall_at_k):
        print(f"  recall@{k:<3} {run.retrieval.recall_at_k[k]:.3f}   "
              f"hit_rate@{k:<3} {run.retrieval.hit_rate_at_k[k]:.3f}")
    print(f"  MRR        {run.retrieval.mrr:.3f}")
    print(f"  NDCG@{cfg.eval.ndcg_k}    {run.retrieval.ndcg:.3f}")
    if run.generation:
        g = run.generation
        print(f"  faithfulness      {g.faithfulness if g.faithfulness is None else round(g.faithfulness, 3)}")
        print(f"  answer relevance  {g.answer_relevance if g.answer_relevance is None else round(g.answer_relevance, 3)}")
        print(f"  abstention acc    {g.abstention_accuracy if g.abstention_accuracy is None else round(g.abstention_accuracy, 3)}")
    print("\n" + render_failure_table(report))
    print(f"\nwrote {out_dir}")

    if args.gate:
        ok, failures = check_gates(run, cfg)
        if not ok:
            print("\nBUILD GATE FAILED:")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("\nbuild gates passed")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from ragmed.eval.ablation import (
        CHUNKING_ROWS,
        DEFAULT_ROWS,
        render_latency_table,
        render_markdown_table,
        run_ablation,
        run_chunking_ablation,
    )
    from ragmed.index.store import CorpusIndex
    from ragmed.llm import generation_llm, judge_llm

    cfg = _load_cfg(args)
    index = CorpusIndex.load(cfg.paths.index_dir, cfg, strict=not args.allow_stale_index)
    golden = load_golden(cfg.eval.golden_set)
    if not golden:
        log.error("no golden set at %s. Run `ragmed golden` first.", cfg.eval.golden_set)
        return 1
    if args.limit:
        golden = golden[: args.limit]

    # The LLM is always constructed, even for a retrieval-only sweep: the query-rewrite
    # row needs one. Passing None there let rewriting silently fall back to the
    # original query, so the row reported numbers identical to the baseline to three
    # decimal places and looked like the clean finding "rewriting does not help" - when
    # in fact nothing had been rewritten at all. Construction is cheap; availability is
    # checked lazily.
    llm = generation_llm(cfg)
    judge = judge_llm(cfg) if args.generation else None

    if any(r.overrides.get("retrieval.rewrite.enabled") for r in DEFAULT_ROWS) and not llm.available():
        log.warning(
            "the query-rewrite ablation row needs a working LLM and none is available; "
            "that row will duplicate the baseline and must not be read as a result"
        )

    runs = run_ablation(
        cfg, index, golden, rows=DEFAULT_ROWS,
        with_generation=args.generation, llm=llm, judge=judge,
    )

    if args.chunking:
        docs = load_documents(cfg.paths.corpus_dir / "documents.jsonl")
        base_chunks = load_chunks(cfg.paths.index_dir / "chunks.jsonl")
        runs += run_chunking_ablation(
            cfg, docs, base_chunks, golden, rows=CHUNKING_ROWS,
            with_generation=args.generation, llm=llm, judge=judge,
        )

    out_dir = cfg.paths.runs_dir / "ablation"
    write_json(out_dir / "results.json", [r.to_dict() for r in runs])

    table = render_markdown_table(runs, k=cfg.eval.ndcg_k)
    baseline = next((r for r in runs if r.label == "Hybrid + reranker"), runs[0])
    tables = [
        "## Ablation\n", table, "\n",
        f"## Latency budget — {baseline.label}\n", render_latency_table(baseline), "\n",
    ]
    (out_dir / "tables.md").write_text("\n".join(tables), encoding="utf-8")

    print("\n" + table)
    print(f"\n## Latency budget — {baseline.label}\n")
    print(render_latency_table(baseline))
    print(f"\nwrote {out_dir}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from ragmed.generate import answer_question
    from ragmed.index.store import CorpusIndex
    from ragmed.llm import generation_llm
    from ragmed.retrieve.pipeline import RetrievalPipeline

    cfg = _load_cfg(args)
    index = CorpusIndex.load(cfg.paths.index_dir, cfg, strict=not args.allow_stale_index)
    llm = generation_llm(cfg)
    pipeline = RetrievalPipeline(cfg, index, llm=llm)

    result = pipeline.retrieve(args.question)

    print(f"\nRetrieved {len(result.contexts)} chunks "
          f"({result.stats.get('context_tokens', 0)} tokens):\n")
    for s in result.contexts:
        print(f"  [{s.rank}] {s.score:+.4f}  {s.chunk.citation}")
        print(f"       {s.chunk.text[:140]}...")

    if llm.available():
        answer = answer_question(llm, args.question, result.context_text, result.contexts, cfg.generation)
        print(f"\n--- Answer ---\n{answer.text}\n")
        print(f"Citations: {answer.citations or 'none'}")
    else:
        print("\n(no LLM available - retrieval only)")

    print("\n--- Latency ---")
    for span in result.trace.spans:
        print(f"  {span.name:<18} {span.ms:7.1f} ms")
    print(f"  {'TOTAL':<18} {result.trace.total_ms:7.1f} ms")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = _load_cfg(args)
    uvicorn.run(
        "ragmed.api:app",
        host=args.host or cfg.service.host,
        port=args.port or cfg.service.port,
        log_level=cfg.service.log_level.lower(),
        reload=args.reload,
    )
    return 0


def cmd_validate_judge(args: argparse.Namespace) -> int:
    from ragmed.eval.judge_validation import (
        load_human_labels,
        render_validation_table,
        validate_judge,
    )

    cfg = _load_cfg(args)
    labels_path = Path(args.labels) if args.labels else (cfg.judge.human_labels or None)
    if not labels_path or not Path(labels_path).exists():
        log.error(
            "no human labels found. Run `ragmed eval --generation --label X` first, then "
            "hand-label runs/X/label_template.jsonl and pass it with --labels."
        )
        return 1

    metrics = Path(args.run) / "metrics.json"
    if not metrics.exists():
        log.error("no run at %s", args.run)
        return 1

    # Judge scores live inside the run's generation per-item records.
    from ragmed.eval.generation_metrics import ItemGenerationScores
    from ragmed.store import read_json

    data = read_json(metrics)
    if not data.get("generation"):
        log.error("that run has no generation scores; re-run with --generation")
        return 1

    per_item_path = Path(args.run) / "generation_items.jsonl"
    if not per_item_path.exists():
        log.error("missing %s", per_item_path)
        return 1
    from ragmed.store import read_jsonl

    judged = [ItemGenerationScores(**row) for row in read_jsonl(per_item_path)]
    results = validate_judge(judged, load_human_labels(Path(labels_path)))

    print(render_validation_table(results))
    print()
    for r in results.values():
        print(f"  {r.verdict()}")
    write_json(Path(args.run) / "judge_validation.json", {k: v.to_dict() for k, v in results.items()})
    return 0


# --- parser ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragmed", description=__doc__ and __doc__.split("\n")[0])
    p.add_argument("--config", help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a config value, e.g. --set retrieval.rerank.enabled=false")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--plain-logs", action="store_true", help="human-readable logs instead of JSON")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("ingest", help="fetch the corpus from PubMed and data/raw")
    sp.add_argument("--local-only", action="store_true", help="skip PubMed, use data/raw only")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("index", help="chunk the corpus and build both indexes")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("golden", help="generate and screen a golden set")
    sp.add_argument("--factoid", type=int, default=80)
    sp.add_argument("--multi-hop", type=int, default=40)
    sp.add_argument("--aggregation", type=int, default=20)
    sp.add_argument("--unanswerable", type=int, default=40)
    sp.add_argument("--seed", type=int, default=20260806)
    sp.add_argument("--allow-stale-index", action="store_true")
    sp.set_defaults(func=cmd_golden)

    sp = sub.add_parser("eval", help="evaluate one configuration")
    sp.add_argument("--label", default="default")
    sp.add_argument("--generation", action="store_true", help="also run generation + LLM-judge")
    sp.add_argument("--limit", type=int, help="evaluate only the first N questions")
    sp.add_argument("--worst", type=int, default=20, help="how many failures to record")
    sp.add_argument("--gate", action="store_true", help="exit non-zero if build gates fail")
    sp.add_argument("--allow-stale-index", action="store_true")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("ablate", help="run the ablation matrix")
    sp.add_argument("--generation", action="store_true")
    sp.add_argument("--chunking", action="store_true", help="also run chunking rows (re-indexes)")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--allow-stale-index", action="store_true")
    sp.set_defaults(func=cmd_ablate)

    sp = sub.add_parser("ask", help="ask a single question")
    sp.add_argument("question")
    sp.add_argument("--allow-stale-index", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("serve", help="run the FastAPI service")
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("validate-judge", help="compare judge scores against human labels")
    sp.add_argument("--run", required=True, help="path to runs/<label>")
    sp.add_argument("--labels", help="path to human labels jsonl")
    sp.set_defaults(func=cmd_validate_judge)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_output=not args.plain_logs)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
