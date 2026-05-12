from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from config import EMBEDDING_DIM, SearchResult
from src.retrieval.compressor import ContextualCompressor
from src.retrieval.mmr import mmr_rerank


class TestStrategyA:

    def test_strategy_a_returns_results(self, strategy_a):
        results = strategy_a.retrieve("test query", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)


class TestStrategyB:

    def test_strategy_b_returns_results(self, strategy_b):
        results = strategy_b.retrieve(
            "How does the system handle peak load?", top_k=2
        )
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_strategy_b_expanded_query_differs_from_original(
        self, query_expander
    ):
        original = "peak load"
        expanded = query_expander.expand(original)
        assert expanded != original
        assert len(expanded) > len(original)


class TestMMR:

    def test_mmr_rerank_returns_diverse_results(self, embedder):
        results = [
            SearchResult(
                chunk_id=f"c{i}",
                text=f"Document text {i}",
                score=1.0 - i * 0.1,
                rank=i + 1,
                source_doc_id=f"d{i}",
            )
            for i in range(5)
        ]
        query_vec = embedder.embed_query("test")
        reranked = mmr_rerank(
            results,
            query_vec,
            embedder,
            top_k=3,
            lambda_param=0.5,
        )
        assert len(reranked) == 3
        assert [r.rank for r in reranked] == [1, 2, 3]
        original_ids = {r.chunk_id for r in results}
        assert all(r.chunk_id in original_ids for r in reranked)


class TestReranker:

    def test_reranker_changes_order(self):
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.1, 0.5, 0.9]

        with patch(
            "src.retrieval.reranker.CrossEncoder", return_value=mock_ce
        ):
            from src.retrieval.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker._model = mock_ce
            reranker._model_name = "mock"

            results = [
                SearchResult(chunk_id="c1", text="A", score=0.9, rank=1, source_doc_id="d1"),
                SearchResult(chunk_id="c2", text="B", score=0.8, rank=2, source_doc_id="d2"),
                SearchResult(chunk_id="c3", text="C", score=0.7, rank=3, source_doc_id="d3"),
            ]
            reranked = reranker.rerank("query", results)

            # c3 should now be rank 1 (highest cross-encoder score 0.9)
            assert reranked[0].chunk_id == "c3"
            assert reranked[0].rank == 1
            # c1 should now be rank 3 (lowest cross-encoder score 0.1)
            assert reranked[2].chunk_id == "c1"
            assert reranked[2].rank == 3


class TestCompressor:

    def test_compressor_reduces_text_length(self, embedder):
        compressor = ContextualCompressor(embedder=embedder)

        long_text = (
            "Horizontal scaling distributes traffic across servers. "
            "The weather is sunny today. "
            "Auto-scaling policies trigger when CPU exceeds threshold. "
            "Bananas are a yellow fruit."
        )
        results = [
            SearchResult(
                chunk_id="c1",
                text=long_text,
                score=0.9,
                rank=1,
                source_doc_id="d1",
            )
        ]
        compressed = compressor.compress("scaling and auto-scaling", results)
        assert len(compressed) == 1
        assert len(compressed[0].text) <= len(long_text)
