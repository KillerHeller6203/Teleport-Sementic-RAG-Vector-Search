"""
Main entry point for the RAG Vector Search Benchmark.

Runs the full pipeline end-to-end: corpus ingestion, retrieval benchmarking
across Strategy A, Strategy B, and HyDE, evaluation against ground truth,
and report generation.
"""

from colorama import Fore, Style, init as colorama_init
import logging
import json
from pathlib import Path

from config import BENCHMARK_QUERIES, LOG_LEVEL
from data.corpus import DOCUMENTS, GROUND_TRUTH
from src.benchmarking.evaluator import RAGEvaluator
from src.benchmarking.reporter import BenchmarkReporter
from src.embeddings.local_embedder import LocalEmbedder
from src.embeddings.vertex_mock import MockGenerativeModel
from src.pipeline.rag_engine import RAGEngine
from src.retrieval.compressor import ContextualCompressor
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.strategy_a import StrategyA
from src.retrieval.strategy_b import StrategyB
from src.storage.faiss_store import FAISSVectorStore
from tabulate import tabulate

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full benchmark pipeline end-to-end."""

    colorama_init(autoreset=True)

    print(Fore.CYAN + Style.BRIGHT + "=" * 60)
    print("  RAG Vector Search Benchmark")
    print("  Strategy A vs B vs HyDE")
    print("=" * 60 + Style.RESET_ALL)
    print()

    embedder = LocalEmbedder()
    mock_model = MockGenerativeModel()
    store = FAISSVectorStore()
    expander = QueryExpander(mock_model)
    strategy_a = StrategyA(embedder, store)
    strategy_b = StrategyB(embedder, store, expander)
    hyde = HyDERetriever(embedder, store, mock_model, n_hypothetical=3)
    reranker = CrossEncoderReranker()
    compressor = ContextualCompressor(embedder)
    engine = RAGEngine(
        embedder=embedder,
        vector_store=store,
        retriever_a=strategy_a,
        retriever_b=strategy_b,
        hyde_retriever=hyde,
        reranker=reranker,
        compressor=compressor,
    )

    stats = engine.ingest(DOCUMENTS)

    print(Fore.CYAN + f"  Documents ingested: {stats['documents_ingested']}")
    print(Fore.CYAN + f"  Chunks created:     {stats['chunks_created']}")
    print(
        Fore.CYAN
        + f"  Embedding time:     {stats['embedding_time_ms']:.2f}ms"
        + Style.RESET_ALL
    )
    print()

    results = engine.benchmark(BENCHMARK_QUERIES)

    print(
        Fore.GREEN
        + f"Benchmark complete — {len(results)} results collected"
        + Style.RESET_ALL
    )
    print()

    eval_scores = RAGEvaluator().evaluate(results, GROUND_TRUTH)

    result_rows: list[list] = []
    for br in results:
        query_short = (
            br.query[:30] + "..." if len(br.query) > 30 else br.query
        )
        metrics = eval_scores.get(br.query, {}).get(br.strategy, {})
        result_rows.append([
            query_short,
            br.strategy,
            f"{metrics.get('mrr', 0):.4f}",
            "Y" if metrics.get("hit_at_k") else "N",
            f"{metrics.get('context_precision', 0):.4f}",
            f"{metrics.get('context_recall', 0):.4f}",
            f"{br.latency_ms:.1f}",
        ])

    print(tabulate(
        result_rows,
        headers=[
            "Query (short)", "Strategy", "MRR",
            "Hit@3", "Ctx Precision", "Ctx Recall", "Latency ms",
        ],
        tablefmt="grid",
    ))
    print()

    stage_accum: dict[str, dict[str, list[float]]] = {}
    for br in results:
        if br.strategy not in stage_accum:
            stage_accum[br.strategy] = {
                "expansion_ms": [],
                "embedding_ms": [],
                "search_ms": [],
                "rerank_ms": [],
                "total_ms": [],
            }
        sl = br.stage_latencies
        stage_accum[br.strategy]["expansion_ms"].append(
            sl.get("expansion_ms", 0.0)
        )
        stage_accum[br.strategy]["embedding_ms"].append(
            sl.get("embedding_ms", 0.0)
        )
        stage_accum[br.strategy]["search_ms"].append(
            sl.get("search_ms", 0.0)
        )
        stage_accum[br.strategy]["rerank_ms"].append(
            sl.get("rerank_ms", 0.0)
        )
        stage_accum[br.strategy]["total_ms"].append(br.latency_ms)

    def _avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    stage_rows: list[list] = []
    for strat in sorted(stage_accum):
        d = stage_accum[strat]
        stage_rows.append([
            strat,
            f"{_avg(d['expansion_ms']):.1f}",
            f"{_avg(d['embedding_ms']):.1f}",
            f"{_avg(d['search_ms']):.1f}",
            f"{_avg(d['rerank_ms']):.1f}",
            f"{_avg(d['total_ms']):.1f}",
        ])

    print(tabulate(
        stage_rows,
        headers=[
            "Strategy", "Expand ms", "Embed ms",
            "Search ms", "Rerank ms", "Total ms",
        ],
        tablefmt="grid",
    ))
    print()

    agg: dict[str, dict[str, list[float]]] = {}
    for query_metrics in eval_scores.values():
        for strat, metrics in query_metrics.items():
            if strat not in agg:
                agg[strat] = {
                    "mrr": [],
                    "context_precision": [],
                    "context_recall": [],
                    "latency": [],
                }
            agg[strat]["mrr"].append(metrics.get("mrr", 0))
            agg[strat]["context_precision"].append(
                metrics.get("context_precision", 0)
            )
            agg[strat]["context_recall"].append(
                metrics.get("context_recall", 0)
            )
    for br in results:
        if br.strategy in agg:
            agg[br.strategy]["latency"].append(br.latency_ms)

    best_mrr = max(agg, key=lambda s: _avg(agg[s]["mrr"]))
    best_prec = max(agg, key=lambda s: _avg(agg[s]["context_precision"]))
    fastest = min(agg, key=lambda s: _avg(agg[s]["latency"]))
    best_recall = max(agg, key=lambda s: _avg(agg[s].get("context_recall", [0])))

    def _ratio(s: str) -> float:
        lat = _avg(agg[s]["latency"])
        return _avg(agg[s]["mrr"]) / lat if lat > 0 else 0.0

    best_ratio = max(agg, key=_ratio)

    print(f"  MRR winner:                    Strategy {best_mrr}")
    print(f"  Context Precision winner:       Strategy {best_prec}")
    print(f"  Fastest:                       Strategy {fastest}")
    print(f"  Best accuracy/latency ratio:    Strategy {best_ratio}")
    print(f"  Best recall winner:            Strategy {best_recall}")
    print()

    reporter = BenchmarkReporter()

    md = reporter.generate_markdown(eval_scores, results)
    reporter.save_markdown(md, "retrieval_benchmark.md")

    json_out = reporter.generate_json(eval_scores, results)
    Path("benchmark_results.json").write_text(json_out, encoding="utf-8")

    print(Fore.GREEN + "Saved retrieval_benchmark.md" + Style.RESET_ALL)
    print(Fore.GREEN + "Saved benchmark_results.json" + Style.RESET_ALL)



if __name__ == "__main__":
    main()
