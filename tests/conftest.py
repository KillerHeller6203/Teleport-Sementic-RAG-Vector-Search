from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, ".")

from config import EMBEDDING_DIM, SearchResult
from src.embeddings.base import BaseEmbedder
from src.embeddings.vertex_mock import MockGenerativeModel
from src.retrieval.query_expander import QueryExpander
from src.storage.faiss_store import FAISSVectorStore

class _MockLocalEmbedder(BaseEmbedder):
    """Deterministic embedder for tests — no real model download."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        vecs = rng.randn(len(texts), self._dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


@pytest.fixture()
def sample_documents() -> list[dict]:
    """Return 3 documents from the corpus."""
    from data.corpus import DOCUMENTS
    return DOCUMENTS[:3]


@pytest.fixture()
def embedder() -> BaseEmbedder:
    """Return a lightweight mock embedder (no real model)."""
    return _MockLocalEmbedder()


@pytest.fixture()
def vector_store() -> FAISSVectorStore:
    """Return an empty FAISS vector store."""
    return FAISSVectorStore(dim=EMBEDDING_DIM)


@pytest.fixture()
def populated_store(
    embedder: BaseEmbedder,
    vector_store: FAISSVectorStore,
    sample_documents: list[dict],
) -> FAISSVectorStore:
    """Return a FAISS store populated with the sample documents."""
    ids = [doc["id"] for doc in sample_documents]
    texts = [doc["text"] for doc in sample_documents]
    embeddings = embedder.embed(texts)
    metadata = [
        {"text": doc["text"], "source_doc_id": doc["id"], "title": doc["title"]}
        for doc in sample_documents
    ]
    vector_store.add(ids, embeddings, metadata)
    return vector_store


@pytest.fixture()
def mock_generative_model() -> MockGenerativeModel:
    """Return a MockGenerativeModel instance."""
    return MockGenerativeModel()


@pytest.fixture()
def query_expander(mock_generative_model: MockGenerativeModel) -> QueryExpander:
    """Return a QueryExpander backed by the mock generative model."""
    return QueryExpander(mock_generative_model)


@pytest.fixture()
def strategy_a(embedder: BaseEmbedder, populated_store: FAISSVectorStore):
    """Return a StrategyA instance."""
    from src.retrieval.strategy_a import StrategyA
    return StrategyA(embedder=embedder, store=populated_store)


@pytest.fixture()
def strategy_b(
    embedder: BaseEmbedder,
    populated_store: FAISSVectorStore,
    query_expander: QueryExpander,
):
    """Return a StrategyB instance."""
    from src.retrieval.strategy_b import StrategyB
    return StrategyB(
        embedder=embedder, store=populated_store, expander=query_expander
    )


@pytest.fixture()
def rag_engine(
    embedder: BaseEmbedder,
    populated_store: FAISSVectorStore,
    strategy_a,
    strategy_b,
):
    """Return a fully assembled RAGEngine with mocked reranker/compressor."""
    from src.pipeline.rag_engine import RAGEngine
    from src.retrieval.compressor import ContextualCompressor

    mock_reranker = MagicMock()
    mock_reranker.rerank = MagicMock(
        side_effect=lambda query, results: results
    )

    compressor = ContextualCompressor(embedder=embedder)

    return RAGEngine(
        embedder=embedder,
        vector_store=populated_store,
        retriever_a=strategy_a,
        retriever_b=strategy_b,
        reranker=mock_reranker,
        compressor=compressor,
    )
