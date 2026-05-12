from __future__ import annotations

from config import BenchmarkResult, SearchResult
from src.pipeline.ingestion import DocumentChunker, IngestionPipeline


class TestIngestion:

    def test_ingestion_creates_chunks(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        doc = {
            "id": "d1",
            "title": "Test Doc",
            "text": (
                "First sentence about scaling systems. "
                "Second sentence about load balancing.\n\n"
                "Third sentence about caching data. "
                "Fourth sentence about Redis performance."
            ),
        }
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        assert all("chunk_id" in c for c in chunks)
        assert all("source_doc_id" in c for c in chunks)
        assert all(c["source_doc_id"] == "d1" for c in chunks)
        # chunk_id format
        assert chunks[0]["chunk_id"] == "d1_chunk_0"

    def test_ingestion_returns_stats(
        self, embedder, vector_store, sample_documents
    ):
        chunker = DocumentChunker()
        pipeline = IngestionPipeline(
            chunker=chunker, embedder=embedder, store=vector_store
        )
        stats = pipeline.ingest(sample_documents)

        assert "documents_ingested" in stats
        assert "chunks_created" in stats
        assert "embedding_time_ms" in stats
        assert stats["documents_ingested"] == len(sample_documents)
        assert stats["chunks_created"] > 0
        assert stats["embedding_time_ms"] >= 0


class TestRAGEngine:

    def test_rag_engine_ingest_then_retrieve_strategy_a(
        self, embedder, sample_documents
    ):
        from unittest.mock import MagicMock

        from src.pipeline.ingestion import DocumentChunker, IngestionPipeline
        from src.pipeline.rag_engine import RAGEngine
        from src.retrieval.compressor import ContextualCompressor
        from src.retrieval.query_expander import QueryExpander
        from src.retrieval.strategy_a import StrategyA
        from src.retrieval.strategy_b import StrategyB
        from src.storage.faiss_store import FAISSVectorStore
        from src.embeddings.vertex_mock import MockGenerativeModel

        store = FAISSVectorStore()
        chunker = DocumentChunker()
        pipeline = IngestionPipeline(chunker, embedder, store)
        pipeline.ingest(sample_documents)

        sa = StrategyA(embedder, store)
        sb = StrategyB(embedder, store, QueryExpander(MockGenerativeModel()))
        reranker = MagicMock()
        reranker.rerank = MagicMock(side_effect=lambda q, r: r)
        compressor = ContextualCompressor(embedder)

        engine = RAGEngine(embedder, store, sa, sb, reranker, compressor)
        result = engine.retrieve("test query", strategy="A", top_k=2)

        assert isinstance(result, BenchmarkResult)
        assert result.strategy == "A"
        assert len(result.results) == 2

    def test_rag_engine_ingest_then_retrieve_strategy_b(
        self, embedder, sample_documents
    ):
        from unittest.mock import MagicMock

        from src.pipeline.ingestion import DocumentChunker, IngestionPipeline
        from src.pipeline.rag_engine import RAGEngine
        from src.retrieval.compressor import ContextualCompressor
        from src.retrieval.query_expander import QueryExpander
        from src.retrieval.strategy_a import StrategyA
        from src.retrieval.strategy_b import StrategyB
        from src.storage.faiss_store import FAISSVectorStore
        from src.embeddings.vertex_mock import MockGenerativeModel

        store = FAISSVectorStore()
        chunker = DocumentChunker()
        pipeline = IngestionPipeline(chunker, embedder, store)
        pipeline.ingest(sample_documents)

        sa = StrategyA(embedder, store)
        sb = StrategyB(embedder, store, QueryExpander(MockGenerativeModel()))
        reranker = MagicMock()
        reranker.rerank = MagicMock(side_effect=lambda q, r: r)
        compressor = ContextualCompressor(embedder)

        engine = RAGEngine(embedder, store, sa, sb, reranker, compressor)
        result = engine.retrieve(
            "How does the system handle peak load?",
            strategy="B",
            top_k=2,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.strategy == "B"
        assert len(result.results) == 2

    def test_rag_engine_benchmark_returns_results_for_all_queries(
        self, embedder, sample_documents
    ):
        from unittest.mock import MagicMock

        from src.pipeline.ingestion import DocumentChunker, IngestionPipeline
        from src.pipeline.rag_engine import RAGEngine
        from src.retrieval.compressor import ContextualCompressor
        from src.retrieval.query_expander import QueryExpander
        from src.retrieval.strategy_a import StrategyA
        from src.retrieval.strategy_b import StrategyB
        from src.retrieval.hyde_retriever import HyDERetriever
        from src.storage.faiss_store import FAISSVectorStore
        from src.embeddings.vertex_mock import MockGenerativeModel

        store = FAISSVectorStore()
        chunker = DocumentChunker()
        pipeline = IngestionPipeline(chunker, embedder, store)
        pipeline.ingest(sample_documents)

        gen_model = MockGenerativeModel()
        sa = StrategyA(embedder, store)
        sb = StrategyB(embedder, store, QueryExpander(gen_model))
        reranker = MagicMock()
        reranker.rerank = MagicMock(side_effect=lambda q, r: r)
        compressor = ContextualCompressor(embedder)
        hyde = HyDERetriever(embedder, store, gen_model)

        engine = RAGEngine(
            embedder, store, sa, sb, reranker, compressor,
            hyde_retriever=hyde,
        )

        queries = ["query one", "query two"]
        results = engine.benchmark(queries)

        # 2 queries × 3 strategies = 6 results
        assert len(results) == 6
        assert all(isinstance(r, BenchmarkResult) for r in results)
        strategies = [r.strategy for r in results]
        assert strategies.count("A") == 2
        assert strategies.count("B") == 2
        assert strategies.count("HYDE") == 2
