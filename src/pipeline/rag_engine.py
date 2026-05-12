from __future__ import annotations

import logging
import time

from config import (
    TOP_K,
    MMR_LAMBDA,
    BenchmarkResult,
    SearchResult,
)
from src.embeddings.base import BaseEmbedder
from src.pipeline.ingestion import DocumentChunker, IngestionPipeline
from src.retrieval.compressor import ContextualCompressor, lost_in_middle_reorder
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.mmr import mmr_rerank
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.strategy_a import StrategyA
from src.retrieval.strategy_b import StrategyB
from src.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class StageTimer:

    def __init__(self, name, store):
        self.name = name
        self.store = store

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.store[self.name] = round(
            (time.perf_counter() - self._t) * 1000, 2
        )


class RAGEngine:

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        retriever_a: StrategyA,
        retriever_b: StrategyB,
        reranker: CrossEncoderReranker,
        compressor: ContextualCompressor,
        hyde_retriever: HyDERetriever | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._retriever_a = retriever_a
        self._retriever_b = retriever_b
        self._reranker = reranker
        self._compressor = compressor
        self._hyde_retriever = hyde_retriever

        self._ingestion = IngestionPipeline(
            chunker=DocumentChunker(),
            embedder=embedder,
            store=vector_store,
        )

    def ingest(self, documents: list[dict]) -> dict:
        return self._ingestion.ingest(documents)

    def retrieve(
        self,
        query: str,
        strategy: str = "A",
        top_k: int = TOP_K,
        use_mmr: bool = False,
        use_reranker: bool = False,
        use_compression: bool = False,
        use_lost_in_middle: bool = False,
    ) -> BenchmarkResult:
        logger.info("retrieve — strategy=%s, query=%.60s", strategy, query)

        t0 = time.perf_counter()
        stage_latencies = {}
        expanded_query = None
        strat_key = strategy.upper()

        fetch_k = top_k * 3 if (use_mmr or use_reranker) else top_k

        if strat_key == "HYDE":
            if self._hyde_retriever is None:
                logger.warning("HyDE not configured, falling back to A")
                strat_key = "A"
                with StageTimer("search_ms", stage_latencies):
                    results = self._retriever_a.retrieve(query, fetch_k)
            else:
                with StageTimer("expansion_ms", stage_latencies):
                    results = self._hyde_retriever.retrieve(query, fetch_k)
                expanded_query = self._hyde_retriever.last_hypothetical_doc

        elif strat_key == "B":
            with StageTimer("expansion_ms", stage_latencies):
                expanded_query = self._retriever_b._expander.expand(query)
            with StageTimer("search_ms", stage_latencies):
                results = self._retriever_b.retrieve(query, fetch_k)

        else:
            with StageTimer("search_ms", stage_latencies):
                results = self._retriever_a.retrieve(query, fetch_k)

        if not results:
            latency = (time.perf_counter() - t0) * 1000
            return BenchmarkResult(
                query=query,
                strategy=strat_key,
                results=[],
                latency_ms=round(latency, 2),
                expanded_query=expanded_query,
                stage_latencies=stage_latencies,
            )

        if use_mmr and results:
            q_emb = self._embedder.embed_query(query)
            results = mmr_rerank(
                results, q_emb, self._embedder,
                top_k=top_k, lambda_param=MMR_LAMBDA,
            )

        if use_reranker and results:
            with StageTimer("rerank_ms", stage_latencies):
                results = self._reranker.rerank(query, results)
                results = results[:top_k]

        if use_compression and results:
            with StageTimer("compress_ms", stage_latencies):
                results = self._compressor.compress(query, results)

        if use_lost_in_middle and results:
            with StageTimer("lim_ms", stage_latencies):
                results = lost_in_middle_reorder(results)

        results = results[:top_k]
        latency = (time.perf_counter() - t0) * 1000

        return BenchmarkResult(
            query=query,
            strategy=strat_key,
            results=results,
            latency_ms=round(latency, 2),
            expanded_query=expanded_query,
            stage_latencies=stage_latencies,
        )

    def benchmark(self, queries: list[str]) -> list[BenchmarkResult]:
        logger.info("Starting benchmark with %d queries...", len(queries))
        all_results = []

        for query in queries:
            result_a = self.retrieve(
                query,
                strategy="A",
                use_mmr=False,
                use_reranker=False,
                use_compression=False,
                use_lost_in_middle=False,
            )

            result_b = self.retrieve(
                query,
                strategy="B",
                use_mmr=True,
                use_reranker=True,
                use_compression=True,
                use_lost_in_middle=True,
            )

            result_hyde = self.retrieve(
                query,
                strategy="HYDE",
                use_reranker=True,
                use_compression=False,
                use_lost_in_middle=True,
            )

            all_results.extend([result_a, result_b, result_hyde])

        logger.info("Benchmark complete — %d total results", len(all_results))
        return all_results
