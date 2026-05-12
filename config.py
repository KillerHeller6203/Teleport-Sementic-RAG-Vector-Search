"""
Application configuration module.

Centralizes all configuration settings including environment variable loading,
model parameters, FAISS index paths, retrieval hyperparameters, and logging
configuration. Uses python-dotenv for environment management.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = 384

TOP_K: int = 3
CHUNK_SIZE: int = 256
CHUNK_OVERLAP: int = 32
FAISS_INDEX_TYPE: str = "IndexFlatIP"
MMR_LAMBDA: float = 0.5

RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BENCHMARK_QUERIES: list[str] = [
    "How does the system handle peak load?",
    "What strategies are used for caching?",
    "How are failures detected and recovered?",
]

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")



@dataclass
class SearchResult:
    """A single retrieval hit returned by a search strategy."""

    chunk_id: str
    text: str
    score: float
    rank: int
    source_doc_id: str


@dataclass
class BenchmarkResult:
    """Aggregated output from running a single query through a strategy."""

    query: str
    strategy: str
    results: list[SearchResult] = field(default_factory=list)
    latency_ms: float = 0.0
    expanded_query: str | None = None
    stage_latencies: dict[str, float] = field(default_factory=dict)


@dataclass
class RAGASScores:
    """RAGAS evaluation scores for a retrieval run."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
