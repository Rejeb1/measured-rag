"""The evaluation layer.

Two metric families, measured and reported separately on purpose:

* **Retrieval** - recall@k, hit rate@k, MRR, NDCG@k. Deterministic, LLM-free, fast
  enough to gate CI. Answers "was the right chunk in the context?"
* **Generation** - faithfulness, answer relevance, context precision, abstention
  accuracy. LLM-judged, slower, noisier, and only as trustworthy as
  `judge_validation` says. Answers "given the context, did the model use it?"

Collapsing these into one score is the mistake this package exists to avoid.
"""

from ragmed.eval.ablation import (
    CHUNKING_ROWS,
    DEFAULT_ROWS,
    AblationRow,
    remap_golden,
    render_latency_table,
    render_markdown_table,
    render_question_type_table,
    run_ablation,
    run_chunking_ablation,
)
from ragmed.eval.build_golden import BuildSpec, build_golden_set
from ragmed.eval.failure_analysis import FailureReport, analyse, render_failure_table
from ragmed.eval.generation_metrics import (
    ABSTAIN_SENTINEL,
    GenerationMetrics,
    aggregate_generation,
    score_generation,
)
from ragmed.eval.judge_validation import (
    ValidationResult,
    load_human_labels,
    render_validation_table,
    validate_judge,
)
from ragmed.eval.retrieval_metrics import (
    RetrievalMetrics,
    breakdown_by_type,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
)
from ragmed.eval.runner import EvalRun, ItemRecord, check_gates, run_eval

__all__ = [
    "ABSTAIN_SENTINEL",
    "CHUNKING_ROWS",
    "DEFAULT_ROWS",
    "AblationRow",
    "BuildSpec",
    "EvalRun",
    "FailureReport",
    "GenerationMetrics",
    "ItemRecord",
    "RetrievalMetrics",
    "ValidationResult",
    "aggregate_generation",
    "analyse",
    "breakdown_by_type",
    "build_golden_set",
    "check_gates",
    "evaluate_retrieval",
    "load_human_labels",
    "ndcg_at_k",
    "recall_at_k",
    "remap_golden",
    "render_failure_table",
    "render_latency_table",
    "render_markdown_table",
    "render_question_type_table",
    "render_validation_table",
    "run_ablation",
    "run_chunking_ablation",
    "run_eval",
    "score_generation",
    "validate_judge",
]
