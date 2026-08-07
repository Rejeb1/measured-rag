"""ragmed - a hybrid-retrieval RAG system over clinical literature, with the
evaluation layer treated as the primary deliverable.
"""

from ragmed.config import Config
from ragmed.telemetry import Trace, configure_logging
from ragmed.types import Answer, Chunk, Document, GoldenItem, Scored, Section

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Config",
    "Document",
    "GoldenItem",
    "Scored",
    "Section",
    "Trace",
    "configure_logging",
]
