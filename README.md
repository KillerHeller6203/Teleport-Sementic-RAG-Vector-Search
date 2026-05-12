![Python](https://img.shields.io/badge/python-3.10+-blue) ![Tests](https://img.shields.io/badge/tests-passing-green) ![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen) ![RAG](https://img.shields.io/badge/RAG-Advanced-purple)

# RAG Vector Search Benchmark

## Overview

Benchmarks three retrieval strategies against a 10-document distributed systems corpus: raw vector search (A), query expansion + reranking pipeline (B), and HyDE (hypothetical document embeddings). Measures MRR, Hit@3, context precision, and latency. Everything is built on abstract base classes so you can swap the local stack for Vertex AI without touching pipeline code.

## Pipeline Architecture

```
INGESTION
  Documents -> Chunker (256 chars, 32 overlap)
            -> LocalEmbedder (all-MiniLM-L6-v2, 384d)
            -> FAISSVectorStore (IndexFlatIP)

STRATEGY A (baseline)
  Query -> embed -> FAISS search -> results

STRATEGY B (enhanced)
  Query -> QueryExpander -> embed -> FAISS
        -> MMR rerank -> CrossEncoder rerank
        -> compress -> lost-in-middle reorder -> results

STRATEGY HyDE
  Query -> generate 3 hypothetical answer docs
        -> embed each -> average vectors
        -> FAISS -> CrossEncoder rerank
        -> lost-in-middle reorder -> results
```

## Project Structure

```
rag-vector-search/
├── main.py                     # runs the full benchmark
├── config.py                   # hyperparams, queries, dataclasses
├── pyproject.toml
├── requirements.txt
├── .env.example
│
├── data/
│   └── corpus.py               # 10 docs + ground truth labels
│
├── src/
│   ├── embeddings/
│   │   ├── base.py             # BaseEmbedder ABC
│   │   ├── local_embedder.py   # sentence-transformers wrapper
│   │   └── vertex_mock.py      # mock for Vertex AI models
│   │
│   ├── storage/
│   │   ├── base.py             # BaseVectorStore ABC
│   │   └── faiss_store.py      # FAISS IndexFlatIP store
│   │
│   ├── retrieval/
│   │   ├── base.py             # BaseRetriever ABC
│   │   ├── strategy_a.py       # raw vector search
│   │   ├── strategy_b.py       # query expansion pipeline
│   │   ├── hyde_retriever.py   # hypothetical doc embeddings
│   │   ├── query_expander.py   # LLM-based query expansion
│   │   ├── reranker.py         # cross-encoder reranker
│   │   ├── compressor.py       # contextual compression + lost-in-middle
│   │   └── mmr.py              # maximal marginal relevance
│   │
│   ├── pipeline/
│   │   ├── ingestion.py        # chunking + embedding pipeline
│   │   └── rag_engine.py       # main orchestrator
│   │
│   └── benchmarking/
│       ├── evaluator.py        # MRR, Hit@k, precision, recall
│       └── reporter.py         # markdown + JSON report gen
│
├── tests/                      # pytest suite (25 tests)
│
├── docs/
│   ├── similarity_metrics.md   # cosine vs euclidean analysis
│   └── vertex_ai_migration.md  # how to swap to Vertex AI
│
├── retrieval_benchmark.md      # benchmark report
└── benchmark_results.json      # generated results (after running main.py)
```

## Quick Start

```bash
git clone <repo-url>
cd rag-vector-search
pip install -e .
python main.py
```

This runs the full pipeline — ingests the corpus, benchmarks all 3 strategies, evaluates against ground truth, and saves the results. You should see a benchmark table printed to the console plus two files: `retrieval_benchmark.md` and `benchmark_results.json`.

## Running Tests

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

25 tests covering embeddings, storage, retrieval strategies, pipeline integration, and the evaluator/reporter.

## Retrieval Strategies

| Strategy | What it does | Latency | When to use |
|----------|-------------|---------|-------------|
| A | Direct embedding search | ~10ms | when latency matters most |
| B | Query expansion + MMR + cross-encoder rerank | ~50ms | balanced quality/speed |
| HyDE | Generate fake answers, embed those instead | ~100ms | when accuracy is everything |

## Evaluation Metrics

- **MRR** — reciprocal rank of the first relevant result. 1.0 means it was rank 1
- **Hit@3** — did any relevant chunk show up in the top 3?
- **Context Precision** — what fraction of retrieved chunks are actually relevant (less noise = better)
- **Context Recall** — what fraction of all relevant docs did we find (completeness)

## Key Design Decisions

- **Cosine similarity via IndexFlatIP** — normalize vectors to unit length, then inner product = cosine. Faster than L2. Details in [similarity_metrics.md](docs/similarity_metrics.md)
- **Abstract base classes everywhere** — `BaseEmbedder`, `BaseVectorStore`, `BaseRetriever`. Swap implementations without changing pipeline code
- **Mock uses real embeddings** — `MockGenerativeModel` does rule-based expansion, but `LocalEmbedder` produces real sentence-transformer vectors. So benchmark numbers are semantically meaningful, not random
- **HyDE bridges the query/doc gap** — the corpus has *answers*, queries are *questions*. HyDE embeds a fake answer so the search vector is in the right neighborhood. See [hyde_retriever.py](src/retrieval/hyde_retriever.py)
- **Lost-in-middle reordering** — LLMs pay more attention to the start and end of their context window, so we put the best chunks there
- **Dependency injection** — every component gets its deps through the constructor, so everything is independently testable

## Production Path

See [vertex_ai_migration.md](docs/vertex_ai_migration.md) for the full guide. Short version:

- `LocalEmbedder` → `textembedding-gecko@003` on Vertex AI
- `FAISSVectorStore` → Vertex AI Vector Search (Matching Engine)
- `MockGenerativeModel` → Gemini 1.5 Pro
- `CrossEncoderReranker` → Vertex AI Ranking API

## Assessment Notes

- **Similarity metric:** cosine, via IndexFlatIP + L2 normalization. Analysis in [similarity_metrics.md](docs/similarity_metrics.md)
- **GCP mocking:** `MockGenerativeModel` uses keyword expansion but the embedder underneath is a real sentence-transformer, so scores reflect actual semantic similarity
- **Three strategies:** A (baseline), B (query expansion + reranking), HyDE (hypothetical document embeddings)
- **Ground truth:** manually defined in `data/corpus.py` with relevant doc IDs per query
- **Swappable:** every component implements an abstract base class, maps 1:1 to a Vertex AI service
