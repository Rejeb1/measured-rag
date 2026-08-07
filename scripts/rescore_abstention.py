"""Recompute abstention from stored answers, without re-running generation.

Abstention is a deterministic function of the answer text, and `records.jsonl` keeps
every answer verbatim. So a fix to the detector can be applied retroactively to a
completed run — which matters when that run cost four hours of GPU time.

This exists because the original exact-match detector missed every real refusal: the
model wrote "INSUFFICIENT CONTEXT" where the prompt asked for "INSUFFICIENT_CONTEXT".

Usage:  python scripts/rescore_abstention.py runs/baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragmed.types import is_abstention  # noqa: E402


def main() -> int:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/baseline")
    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        print(f"no records at {records_path}", file=sys.stderr)
        return 1

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]

    rows = []
    for r in records:
        answer = r.get("answer") or ""
        answerable = bool(r["gold_chunk_ids"])
        stored = bool(r.get("abstained"))
        now = is_abstention(answer)
        rows.append(
            {
                "qid": r["qid"],
                "answerable": answerable,
                "stored": stored,
                "recomputed": now,
                # Refusing is correct exactly when the question is unanswerable.
                "correct": now != answerable,
                "empty_answer": not answer.strip(),
            }
        )

    scored = [r for r in rows if r["qid"]]
    unans = [r for r in scored if not r["answerable"]]
    ans = [r for r in scored if r["answerable"]]

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"

    print(f"run: {run_dir}\n")
    print("                         stored detector    fixed detector")
    print(f"  refusals overall       {pct(sum(r['stored'] for r in scored), len(scored)):<18} "
          f"{pct(sum(r['recomputed'] for r in scored), len(scored))}")
    print(f"  correct refusals       {pct(sum(r['stored'] and not r['answerable'] for r in scored), len(unans)):<18} "
          f"{pct(sum(r['recomputed'] for r in unans), len(unans))}   (of {len(unans)} unanswerable)")
    print(f"  wrong refusals         {pct(sum(r['stored'] and r['answerable'] for r in scored), len(ans)):<18} "
          f"{pct(sum(r['recomputed'] for r in ans), len(ans))}   (of {len(ans)} answerable)")

    old_acc = sum((r["stored"] != r["answerable"]) for r in scored) / len(scored)
    new_acc = sum(r["correct"] for r in scored) / len(scored)
    print(f"\n  abstention accuracy    {old_acc:.3f}              {new_acc:.3f}")

    missed = [r for r in rows if r["recomputed"] and not r["stored"]]
    if missed:
        print(f"\n  {len(missed)} refusals the old detector missed: "
              f"{', '.join(r['qid'] for r in missed[:12])}"
              f"{' ...' if len(missed) > 12 else ''}")

    answered_unanswerable = [r for r in unans if not r["recomputed"]]
    if answered_unanswerable:
        print(f"\n  {len(answered_unanswerable)} unanswerable questions still answered "
              f"(genuine hallucination risk): {', '.join(r['qid'] for r in answered_unanswerable)}")

    out = run_dir / "abstention_rescored.json"
    out.write_text(json.dumps({"rows": rows, "accuracy": new_acc}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
