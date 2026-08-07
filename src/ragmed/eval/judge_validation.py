"""Validating the LLM-judge against human labels.

An unvalidated LLM-judge produces numbers, not measurements. This module answers the
only question that makes a judge score citable: *how often does it agree with a
human, and in which direction does it err?*

The protocol is deliberately small - label ~50 items by hand, run the judge on the
same items, and report agreement. Fifty is not many, and the confidence interval
reported here reflects that. The point is not to certify the judge; it is to know
whether faithfulness=0.68 means "roughly two thirds" or "a coin flip with extra steps".

Bias direction matters more than raw agreement. A judge that is systematically
generous inflates every faithfulness number in the README by a known amount, and a
known bias can be stated alongside the result. An unmeasured one cannot.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragmed.eval.generation_metrics import ItemGenerationScores
from ragmed.store import read_jsonl

log = logging.getLogger(__name__)

# Scores within this distance count as agreement on a 0-1 scale.
AGREEMENT_TOLERANCE = 0.25


@dataclass
class HumanLabel:
    qid: str
    metric: str  # "faithfulness" | "answer_relevance" | "context_precision"
    score: float
    note: str = ""


@dataclass
class ValidationResult:
    metric: str
    n: int = 0
    agreement: float = 0.0
    mean_judge: float = 0.0
    mean_human: float = 0.0
    bias: float = 0.0  # judge minus human; positive means the judge is generous
    mae: float = 0.0
    correlation: float | None = None
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agreement_ci95(self) -> tuple[float, float]:
        """Normal-approximation interval. Wide at n=50, and it should look wide."""
        if self.n == 0:
            return (0.0, 0.0)
        se = math.sqrt(max(self.agreement * (1 - self.agreement), 1e-9) / self.n)
        return (max(0.0, self.agreement - 1.96 * se), min(1.0, self.agreement + 1.96 * se))

    def verdict(self) -> str:
        """A plain-English statement of how far the judge can be trusted."""
        if self.n < 20:
            return (
                f"Only {self.n} human labels for {self.metric} - too few to say anything. "
                f"Treat judge scores as unvalidated."
            )
        lo, hi = self.agreement_ci95
        direction = "generous" if self.bias > 0.05 else "harsh" if self.bias < -0.05 else "unbiased"
        confidence = (
            "usable" if self.agreement >= 0.8 else "weak" if self.agreement >= 0.65 else "unreliable"
        )
        return (
            f"{self.metric}: {self.agreement:.0%} agreement with humans "
            f"(95% CI {lo:.0%}-{hi:.0%}, n={self.n}), MAE {self.mae:.2f}, "
            f"judge is {direction} by {self.bias:+.2f}. Verdict: {confidence}."
        )

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.agreement_ci95
        return {
            "metric": self.metric,
            "n": self.n,
            "agreement": round(self.agreement, 4),
            "agreement_ci95": [round(lo, 4), round(hi, 4)],
            "mean_judge": round(self.mean_judge, 4),
            "mean_human": round(self.mean_human, 4),
            "bias": round(self.bias, 4),
            "mae": round(self.mae, 4),
            "correlation": round(self.correlation, 4) if self.correlation is not None else None,
            "verdict": self.verdict(),
            "n_disagreements": len(self.disagreements),
        }


def load_human_labels(path: Path) -> list[HumanLabel]:
    labels: list[HumanLabel] = []
    for row in read_jsonl(path):
        labels.append(
            HumanLabel(
                qid=row["qid"],
                metric=row.get("metric", "faithfulness"),
                score=float(row["score"]),
                note=row.get("note", ""),
            )
        )
    return labels


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        # No variance in one series - correlation is undefined, not zero.
        return None
    return num / (dx * dy)


def validate_judge(
    judge_scores: list[ItemGenerationScores],
    human_labels: list[HumanLabel],
    tolerance: float = AGREEMENT_TOLERANCE,
) -> dict[str, ValidationResult]:
    by_qid = {s.qid: s for s in judge_scores}
    by_metric: dict[str, list[tuple[float, float, str]]] = {}

    for label in human_labels:
        scores = by_qid.get(label.qid)
        if scores is None:
            log.warning("human label for %s has no matching judge score", label.qid)
            continue
        judged = getattr(scores, label.metric, None)
        if judged is None:
            continue
        by_metric.setdefault(label.metric, []).append((float(judged), label.score, label.qid))

    results: dict[str, ValidationResult] = {}
    for metric, rows in by_metric.items():
        n = len(rows)
        judge_vals = [j for j, _, _ in rows]
        human_vals = [h for _, h, _ in rows]
        agree = sum(1 for j, h, _ in rows if abs(j - h) <= tolerance)
        disagreements = [
            {"qid": q, "judge": j, "human": h, "delta": round(j - h, 3)}
            for j, h, q in rows
            if abs(j - h) > tolerance
        ]
        disagreements.sort(key=lambda d: -abs(d["delta"]))

        results[metric] = ValidationResult(
            metric=metric,
            n=n,
            agreement=agree / n,
            mean_judge=sum(judge_vals) / n,
            mean_human=sum(human_vals) / n,
            bias=(sum(judge_vals) - sum(human_vals)) / n,
            mae=sum(abs(j - h) for j, h, _ in rows) / n,
            correlation=_pearson(judge_vals, human_vals),
            disagreements=disagreements[:10],
        )

    return results


def render_validation_table(results: dict[str, ValidationResult]) -> str:
    lines = [
        "| Metric | n | Agreement (±0.25) | 95% CI | MAE | Judge bias | Correlation |",
        "|---|---|---|---|---|---|---|",
    ]
    for metric, r in sorted(results.items()):
        lo, hi = r.agreement_ci95
        corr = f"{r.correlation:.2f}" if r.correlation is not None else "—"
        lines.append(
            f"| {metric} | {r.n} | {r.agreement:.0%} | {lo:.0%}–{hi:.0%} | "
            f"{r.mae:.2f} | {r.bias:+.2f} | {corr} |"
        )
    return "\n".join(lines)


def make_labeling_template(
    judge_scores: list[ItemGenerationScores],
    records: list[Any],
    metric: str = "faithfulness",
    n: int = 50,
) -> list[dict[str, Any]]:
    """Emit items for a human to label, with the judge's own score withheld.

    Showing the judge's verdict while labelling would anchor the human to it and the
    resulting "agreement" would measure suggestibility rather than accuracy.
    """
    by_qid = {r.qid: r for r in records}
    out: list[dict[str, Any]] = []
    for s in judge_scores[:n]:
        rec = by_qid.get(s.qid)
        if rec is None:
            continue
        out.append(
            {
                "qid": s.qid,
                "metric": metric,
                "question": rec.question,
                "answer": rec.answer,
                "score": None,  # <- fill this in by hand, 0.0 to 1.0
                "note": "",
            }
        )
    return out
