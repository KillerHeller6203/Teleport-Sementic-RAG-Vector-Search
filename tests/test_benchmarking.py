"""
Tests for the benchmarking sub-package.

Covers the evaluator (MRR, context precision) and the reporter
(JSON validity, markdown section presence).
"""

from __future__ import annotations

import json

from config import BenchmarkResult, SearchResult
from src.benchmarking.evaluator import RAGEvaluator
from src.benchmarking.reporter import BenchmarkReporter


def _make_benchmark_results() -> tuple[list[BenchmarkResult], list[dict]]:
    """Helper — build sample benchmark results and ground truth."""
    results_a = BenchmarkResult(
        query="How does the system handle peak load?",
        strategy="A",
        results=[
            SearchResult(chunk_id="c1", text="Scaling text", score=0.95, rank=1, source_doc_id="doc_01"),
            SearchResult(chunk_id="c2", text="Caching text", score=0.80, rank=2, source_doc_id="doc_04"),
            SearchResult(chunk_id="c3", text="LB text", score=0.75, rank=3, source_doc_id="doc_09"),
        ],
        latency_ms=12.5,
    )
    results_b = BenchmarkResult(
        query="How does the system handle peak load?",
        strategy="B",
        results=[
            SearchResult(chunk_id="c1", text="Scaling text", score=0.92, rank=1, source_doc_id="doc_01"),
            SearchResult(chunk_id="c4", text="Auto-scale", score=0.88, rank=2, source_doc_id="doc_02"),
            SearchResult(chunk_id="c3", text="LB text", score=0.70, rank=3, source_doc_id="doc_09"),
        ],
        latency_ms=18.3,
        expanded_query="traffic surge, horizontal scaling",
    )
    ground_truth = [
        {
            "query": "How does the system handle peak load?",
            "relevant_ids": ["doc_01", "doc_02", "doc_09"],
        },
    ]
    return [results_a, results_b], ground_truth


class TestEvaluator:
    """Tests for the RAGEvaluator."""

    def test_evaluator_computes_mrr(self):
        brs, gt = _make_benchmark_results()
        evaluator = RAGEvaluator()
        eval_results = evaluator.evaluate(brs, gt)

        query = "How does the system handle peak load?"
        # doc_01 is at rank 1 for both strategies → MRR = 1.0
        assert eval_results[query]["A"]["mrr"] == 1.0
        assert eval_results[query]["B"]["mrr"] == 1.0

    def test_evaluator_computes_context_precision(self):
        brs, gt = _make_benchmark_results()
        evaluator = RAGEvaluator()
        eval_results = evaluator.evaluate(brs, gt)

        query = "How does the system handle peak load?"
        # Strategy A: 2/3 relevant (doc_01, doc_09 yes; doc_04 no)
        assert abs(eval_results[query]["A"]["context_precision"] - 2 / 3) < 1e-3
        # Strategy B: 3/3 relevant (doc_01, doc_02, doc_09)
        assert eval_results[query]["B"]["context_precision"] == 1.0


class TestReporter:
    """Tests for the BenchmarkReporter."""

    def test_reporter_generates_valid_json(self):
        brs, gt = _make_benchmark_results()
        evaluator = RAGEvaluator()
        eval_results = evaluator.evaluate(brs, gt)

        reporter = BenchmarkReporter()
        json_str = reporter.generate_json(eval_results, brs)

        # Must be valid JSON
        parsed = json.loads(json_str)
        assert "evaluation_metrics" in parsed
        assert "benchmark_details" in parsed
        assert len(parsed["benchmark_details"]) == 2

    def test_reporter_generates_markdown_with_required_sections(self):
        brs, gt = _make_benchmark_results()
        evaluator = RAGEvaluator()
        eval_results = evaluator.evaluate(brs, gt)

        reporter = BenchmarkReporter()
        md = reporter.generate_markdown(eval_results, brs)

        assert "Strategy A" in md
        assert "Strategy B" in md
        assert "MRR" in md
        assert "Ctx Precision" in md
        assert "Executive Summary" in md
        assert "Analysis" in md
