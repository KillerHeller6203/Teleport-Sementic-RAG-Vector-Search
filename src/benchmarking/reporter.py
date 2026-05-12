from __future__ import annotations

import json
import logging
from pathlib import Path

from tabulate import tabulate

from config import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkReporter:

    def generate_json(
        self,
        eval_results: dict,
        benchmark_results: list[BenchmarkResult],
    ) -> str:
        payload = {
            "evaluation_metrics": eval_results,
            "benchmark_details": [
                {
                    "query": br.query,
                    "strategy": br.strategy,
                    "latency_ms": br.latency_ms,
                    "expanded_query": br.expanded_query,
                    "results": [
                        {
                            "rank": r.rank,
                            "chunk_id": r.chunk_id,
                            "source_doc_id": r.source_doc_id,
                            "score": round(r.score, 4),
                            "text_preview": r.text[:120],
                        }
                        for r in br.results
                    ],
                }
                for br in benchmark_results
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def generate_markdown(
        self,
        eval_results: dict,
        benchmark_results: list[BenchmarkResult],
    ) -> str:
        lines = []

        lines.append("# Retrieval Benchmark Report\n")

        lines.append("## Executive Summary\n")
        lines.append(self._executive_summary(eval_results, benchmark_results))
        lines.append("")

        by_query = {}
        for br in benchmark_results:
            by_query.setdefault(br.query, []).append(br)

        for query, brs in by_query.items():
            lines.append(f"## Query: *{query}*\n")

            for br in brs:
                lines.append(f"### Strategy {br.strategy}\n")
                lines.append(f"- **Latency:** {br.latency_ms:.2f} ms")
                if br.expanded_query:
                    lines.append(
                        f"- **Expanded query:** {br.expanded_query}"
                    )
                lines.append("")

                table_data = [
                    [
                        r.rank,
                        r.chunk_id,
                        r.source_doc_id,
                        f"{r.score:.4f}",
                        r.text[:80] + ("…" if len(r.text) > 80 else ""),
                    ]
                    for r in br.results
                ]
                lines.append(
                    tabulate(
                        table_data,
                        headers=["Rank", "Chunk ID", "Source Doc", "Score", "Text Preview"],
                        tablefmt="github",
                    )
                )
                lines.append("")

            if query in eval_results:
                metrics_table = []
                for strategy, metrics in eval_results[query].items():
                    metrics_table.append(
                        [
                            strategy,
                            "✓" if metrics.get("hit_at_k") else "✗",
                            f"{metrics.get('mrr', 0):.4f}",
                            f"{metrics.get('context_precision', 0):.4f}",
                            f"{metrics.get('context_recall', 0):.4f}",
                            f"{metrics.get('faithfulness', 0):.2f}",
                            f"{metrics.get('answer_relevancy', 0):.2f}",
                        ]
                    )
                lines.append("#### Metrics Comparison\n")
                lines.append(
                    tabulate(
                        metrics_table,
                        headers=[
                            "Strategy", "Hit@K", "MRR",
                            "Ctx Precision", "Ctx Recall",
                            "Faithfulness*", "Answer Rel.*",
                        ],
                        tablefmt="github",
                    )
                )
                lines.append(
                    "\n*\\* Mocked values — in production, computed via "
                    "RAGAS with LLM-generated answers.*\n"
                )

            lines.append("#### Analysis\n")
            lines.append(self._qualitative_analysis(query, brs, eval_results))
            lines.append("")

        lines.append("## Aggregate Metrics Comparison\n")
        lines.append(self._aggregate_table(eval_results))
        lines.append("")

        return "\n".join(lines)

    def save_markdown(self, content: str, path: str) -> None:
        Path(path).write_text(content, encoding="utf-8")
        logger.info("Report saved to %s", path)

    @staticmethod
    def _executive_summary(eval_results, benchmark_results):
        n_queries = len(eval_results)

        latencies = {}
        for br in benchmark_results:
            latencies.setdefault(br.strategy, []).append(br.latency_ms)

        lat_strs = []
        for strat, lats in sorted(latencies.items()):
            avg = sum(lats) / len(lats) if lats else 0
            lat_strs.append(f"Strategy {strat}: {avg:.1f} ms avg")

        mrr_by_strat = {}
        for query_metrics in eval_results.values():
            for strat, metrics in query_metrics.items():
                mrr_by_strat.setdefault(strat, []).append(
                    metrics.get("mrr", 0)
                )

        mrr_strs = []
        for strat, mrrs in sorted(mrr_by_strat.items()):
            avg = sum(mrrs) / len(mrrs) if mrrs else 0
            mrr_strs.append("Strategy {}: {:.4f}".format(strat, avg))

        return (
            f"This benchmark evaluated **{n_queries} queries** across "
            f"two retrieval strategies.\n\n"
            f"**Average latency** — {'; '.join(lat_strs)}.\n\n"
            f"**Average MRR** — {'; '.join(mrr_strs)}."
        )

    @staticmethod
    def _qualitative_analysis(query, brs, eval_results):
        parts = []
        query_metrics = eval_results.get(query, {})

        for br in brs:
            m = query_metrics.get(br.strategy, {})
            hit = m.get("hit_at_k", False)
            mrr = m.get("mrr", 0)
            prec = m.get("context_precision", 0)
            recall = m.get("context_recall", 0)

            if hit and mrr == 1.0:
                quality = "returned the most relevant document at rank 1"
            elif hit:
                quality = (
                    "found a relevant document (MRR={:.2f}) but not at "
                    "the top position".format(mrr)
                )
            else:
                quality = "failed to retrieve any relevant document"

            parts.append(
                f"Strategy {br.strategy} {quality} with context "
                f"precision {prec:.2f} and recall {recall:.2f} "
                f"(latency {br.latency_ms:.1f} ms)."
            )
        return " ".join(parts)

    @staticmethod
    def _aggregate_table(eval_results):
        agg = {}

        for query_metrics in eval_results.values():
            for strat, metrics in query_metrics.items():
                if strat not in agg:
                    agg[strat] = {
                        "hit_rate": [],
                        "mrr": [],
                        "ctx_precision": [],
                        "ctx_recall": [],
                    }
                agg[strat]["hit_rate"].append(
                    1.0 if metrics.get("hit_at_k") else 0.0
                )
                agg[strat]["mrr"].append(metrics.get("mrr", 0))
                agg[strat]["ctx_precision"].append(
                    metrics.get("context_precision", 0)
                )
                agg[strat]["ctx_recall"].append(
                    metrics.get("context_recall", 0)
                )

        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        rows = []
        for strat in sorted(agg):
            d = agg[strat]
            rows.append(
                [
                    strat,
                    f"{_avg(d['hit_rate']):.2%}",
                    f"{_avg(d['mrr']):.4f}",
                    f"{_avg(d['ctx_precision']):.4f}",
                    f"{_avg(d['ctx_recall']):.4f}",
                ]
            )

        return tabulate(
            rows,
            headers=[
                "Strategy", "Hit Rate", "Avg MRR",
                "Avg Ctx Precision", "Avg Ctx Recall",
            ],
            tablefmt="github",
        )
