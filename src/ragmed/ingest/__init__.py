"""Corpus ingestion: fetch, normalise, chunk."""

from ragmed.ingest.chunking import chunk_document, chunk_documents
from ragmed.ingest.local import load_local_documents
from ragmed.ingest.pubmed import PubMedClient, fetch_corpus

__all__ = [
    "PubMedClient",
    "chunk_document",
    "chunk_documents",
    "fetch_corpus",
    "load_local_documents",
]
