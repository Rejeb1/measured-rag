"""Retrieval: fusion, reranking, context assembly, and the pipeline that joins them."""

from ragmed.retrieve.assemble import assemble_context, deduplicate, order_for_attention
from ragmed.retrieve.fusion import FusedHit, fuse, reciprocal_rank_fusion
from ragmed.retrieve.pipeline import RetrievalPipeline, RetrievalResult
from ragmed.retrieve.rerank import CrossEncoderReranker
from ragmed.retrieve.rewrite import rewrite_query

__all__ = [
    "CrossEncoderReranker",
    "FusedHit",
    "RetrievalPipeline",
    "RetrievalResult",
    "assemble_context",
    "deduplicate",
    "fuse",
    "order_for_attention",
    "reciprocal_rank_fusion",
    "rewrite_query",
]
