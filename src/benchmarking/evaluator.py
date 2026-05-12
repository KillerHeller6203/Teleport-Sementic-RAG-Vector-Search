from __future__ import annotations

from collections import defaultdict

from config import BenchmarkResult


class RAGEvaluator:
    """Scores retrieval results against ground truth relevance labels."""

    _MOCK_FAITHFULNESS = 0.85
    _MOCK_ANSWER_RELEVANCY = 0.80

    def evaluate(
        self,
        benchmark_results: list[BenchmarkResult],
        ground_truth: list[dict],
    ) -> dict:
        gt_lookup = {
            item["query"]: set(item["relevant_ids"])
            for item in ground_truth
        }

        eval_results = defaultdict(dict)

        for br in benchmark_results:
            relevant_ids = gt_lookup.get(br.query, set())

            retrieved_doc_ids = [r.source_doc_id for r in br.results]
            retrieved_doc_set = set(retrieved_doc_ids)

            hit_at_k = bool(relevant_ids & retrieved_doc_set)

            mrr = 0.0
            for r in br.results:
                if r.source_doc_id in relevant_ids:
                    mrr = 1.0 / r.rank
                    break

            if retrieved_doc_ids:
                relevant_count = sum(
                    1 for doc_id in retrieved_doc_ids
                    if doc_id in relevant_ids
                )
                context_precision = relevant_count / len(retrieved_doc_ids)
            else:
                context_precision = 0.0

            if relevant_ids:
                recalled = len(relevant_ids & retrieved_doc_set)
                context_recall = recalled / len(relevant_ids)
            else:
                context_recall = 0.0

            faithfulness = self._MOCK_FAITHFULNESS
            answer_relevancy = self._MOCK_ANSWER_RELEVANCY

            metrics = {
                "hit_at_k": hit_at_k,
                "mrr": round(mrr, 4),
                "context_precision": round(context_precision, 4),
                "context_recall": round(context_recall, 4),
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
            }

            eval_results[br.query][br.strategy] = metrics

        return dict(eval_results)
