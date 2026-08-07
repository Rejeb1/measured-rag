"""Configuration.

Every knob the ablation table varies lives here, and every ablation row is expressed
as a set of dotted-path overrides against one base config. That is the whole reason
the config is this granular: "hybrid + reranker" and "dense only" are not two code
paths, they are two dicts. If a row of the table required a code change, the table
would stop being trustworthy.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class PathsConfig(BaseModel):
    data_dir: Path = REPO_ROOT / "data"
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    index_dir: Path = REPO_ROOT / "data" / "index"
    golden_dir: Path = REPO_ROOT / "data" / "golden"
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    runs_dir: Path = REPO_ROOT / "runs"

    def ensure(self) -> None:
        for p in (
            self.data_dir,
            self.corpus_dir,
            self.index_dir,
            self.golden_dir,
            self.cache_dir,
            self.runs_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


class CorpusConfig(BaseModel):
    # PubMed E-utilities search terms. Each becomes one esearch query.
    pubmed_queries: list[str] = Field(
        default_factory=lambda: [
            "type 2 diabetes management guideline[Title/Abstract]",
            "heart failure with preserved ejection fraction treatment",
            "community acquired pneumonia antibiotic therapy adult",
            "atrial fibrillation anticoagulation stroke prevention",
            "chronic kidney disease progression management",
        ]
    )
    max_per_query: int = 400
    # Only records with an abstract are useful; titles alone are not retrievable text.
    require_abstract: bool = True
    email: str | None = None
    api_key: str | None = None
    tool: str = "ragmed"


class ChunkingConfig(BaseModel):
    # "structure" splits on section boundaries first and only then packs to size;
    # "fixed" ignores structure entirely. This is an ablation row, not a constant.
    strategy: Literal["structure", "fixed"] = "structure"
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 32


class BM25Config(BaseModel):
    enabled: bool = True
    k1: float = 1.2
    b: float = 0.75
    top_k: int = 50
    use_stopwords: bool = True


class DenseConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-small-en-v1.5"
    # BGE models are trained with an asymmetric instruction prefix on the query side.
    # Dropping it costs real recall, so it is config rather than a magic string.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""
    top_k: int = 50
    batch_size: int = 32
    device: str = "auto"
    normalize: bool = True


class FusionConfig(BaseModel):
    # "rrf" fuses on rank; "normalized_sum" min-max normalises each retriever's raw
    # scores and adds them. The second exists purely so the ablation can show what
    # rank-based fusion buys - dense-only and BM25-only are expressed by disabling a
    # retriever, never here, so there is exactly one way to configure each row.
    method: Literal["rrf", "normalized_sum"] = "rrf"
    # 60 is the constant from the original RRF paper and what most production
    # systems use unchanged. Exposed so the ablation can show it barely matters.
    k: int = 60


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    batch_size: int = 16
    device: str = "auto"
    # How many fused candidates to feed the cross-encoder. The retrieve-wide,
    # rerank-narrow tradeoff is exactly what the ablation measures.
    candidates: int = 50


class RewriteConfig(BaseModel):
    enabled: bool = False
    max_queries: int = 3


class AssemblyConfig(BaseModel):
    max_context_tokens: int = 3000
    # Cosine similarity above which two chunks are treated as near-duplicates.
    dedup_threshold: float = 0.92
    # "edges" puts the highest-scoring chunks at the start and end of the context,
    # where attention is strongest; "sequential" is plain descending order.
    ordering: Literal["edges", "sequential"] = "edges"


class RetrievalConfig(BaseModel):
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
    assembly: AssemblyConfig = Field(default_factory=AssemblyConfig)


class GenerationConfig(BaseModel):
    provider: Literal["ollama", "null"] = "ollama"
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 800
    # MUST be set explicitly. Ollama defaults num_ctx to 4096 regardless of what the
    # model supports (llama3.2 advertises 131072), and long prompts do not degrade
    # gracefully at that budget - they fail as empty responses, HTTP 500s, or
    # degenerate repeated-token output that looks like a broken model.
    #
    # Measured over 5 runs per cell, varying only num_ctx:
    #     prompt              tokens   ok@4096   ok@8192
    #     factoid                595      5/5       5/5
    #     multi-hop              893      3/5       5/5
    #     aggregation           1264      0/5       5/5
    #
    # Failure rate tracks prompt length at 4096 and vanishes at 8192 on both GPU and
    # CPU. An assembled 5-chunk context is ~2500 tokens before instructions, and the
    # judge's context-precision prompt repeats every retrieved passage, so the real
    # workload sits squarely in the range that fails.
    num_ctx: int = 8192
    # Generous because the *first* call after a cold start pays for loading weights
    # into VRAM on top of inference. Observed: a 2GB model on a 4GB card exceeded
    # 120s on its first request while the disk was busy, and a timeout there aborts
    # a golden-set build that was otherwise working.
    timeout_s: float = 300.0
    # Refusing beats hallucinating; the prompt enforces it and the unanswerable
    # slice of the golden set measures whether it actually happens.
    require_citations: bool = True


class JudgeConfig(BaseModel):
    provider: Literal["ollama", "null"] = "ollama"
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    # See GenerationConfig.num_ctx. The judge's prompts are the longest in the system
    # - context precision passes every retrieved passage - so this matters most here.
    num_ctx: int = 8192
    timeout_s: float = 300.0
    # Path to human labels used to validate the judge. Without this the judge's
    # numbers are opinions, not measurements.
    human_labels: Path | None = None


class EvalConfig(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    ndcg_k: int = 10
    golden_set: Path = REPO_ROOT / "data" / "golden" / "golden_set.jsonl"
    # Build-gate thresholds. CI fails if a change drops below these.
    min_recall_at_10: float = 0.80
    min_ndcg_at_10: float = 0.60
    min_faithfulness: float = 0.70
    # Calibrated for CUDA torch (measured 758ms p95). Raise to ~12000 on CPU-only
    # torch - see configs/default.yaml for both measurements.
    max_p95_latency_ms: float = 1500.0


class ServiceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    embedding_cache_size: int = 4096


class Config(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        raw: dict[str, Any] = {}
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = cls.model_validate(raw)
        cfg = cfg.with_env()
        if overrides:
            cfg = cfg.patch(overrides)
        return cfg

    def with_env(self) -> Config:
        """Environment wins over the YAML file, for secrets and deploy-time settings."""
        env_map = {
            "RAGMED_OLLAMA_HOST": ("generation.host", "judge.host"),
            "RAGMED_GEN_MODEL": ("generation.model",),
            "RAGMED_JUDGE_MODEL": ("judge.model",),
            "RAGMED_NCBI_EMAIL": ("corpus.email",),
            "RAGMED_NCBI_API_KEY": ("corpus.api_key",),
            "RAGMED_NCBI_TOOL": ("corpus.tool",),
            "RAGMED_LOG_LEVEL": ("service.log_level",),
        }
        patch: dict[str, Any] = {}
        for env_key, targets in env_map.items():
            val = os.environ.get(env_key)
            if val:
                for t in targets:
                    patch[t] = val
        return self.patch(patch) if patch else self

    def patch(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with dotted-path overrides applied.

        ``{"retrieval.rerank.enabled": False}`` is how an ablation row is expressed.
        """
        data = copy.deepcopy(self.model_dump())
        for dotted, value in overrides.items():
            node: Any = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                if part not in node:
                    raise KeyError(f"unknown config path segment {part!r} in {dotted!r}")
                node = node[part]
            leaf = parts[-1]
            if leaf not in node:
                raise KeyError(f"unknown config key {dotted!r}")
            node[leaf] = value
        return Config.model_validate(data)

    def fingerprint(self) -> str:
        """Stable hash of the retrieval-relevant config.

        Index artifacts are keyed by this, so changing the embedding model or the
        chunking strategy can never silently reuse a stale index.
        """
        import hashlib
        import json

        relevant = {
            "chunking": self.chunking.model_dump(),
            "dense_model": self.retrieval.dense.model,
            "normalize": self.retrieval.dense.normalize,
            "passage_prefix": self.retrieval.dense.passage_prefix,
        }
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]
