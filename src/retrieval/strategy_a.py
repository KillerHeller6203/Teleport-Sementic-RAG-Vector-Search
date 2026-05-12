from __future__ import annotations

import logging

import numpy as np
from rank_bm25 import BM25Okapi

from config import SearchResult
from src.embeddings.base import BaseEmbedder
from src.retrieval.base import BaseRetriever
from src.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class StrategyA(BaseRetriever):
    """Baseline — direct embed and search, no preprocessing."""

    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        q_vec = self._embedder.embed_query(query)
        return self._store.search(q_vec, top_k)

    def hybrid_retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """BM25 + dense cosine fusion. Bit experimental but works ok."""
        all_ids = list(self._store._idx_to_id)
        all_meta = [self._store._metadata[doc_id] for doc_id in all_ids]

        tokenized_corpus = [
            chunk["text"].lower().split() for chunk in all_meta
        ]
        bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        bm25_max = float(np.max(bm25_scores)) + 1e-9
        bm25_norm = bm25_scores / bm25_max

        n_total = len(all_ids)
        dense_results = self.retrieve(query, top_k=n_total)
        dense_score_map = {r.chunk_id: r.score for r in dense_results}

        dense_scores = np.array(
            [dense_score_map.get(cid, 0.0) for cid in all_ids],
            dtype=np.float32,
        )
        dense_max = float(np.max(dense_scores)) + 1e-9
        dense_norm = dense_scores / dense_max

        fused = 0.5 * dense_norm + 0.5 * bm25_norm
        ranked_indices = np.argsort(fused)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc_id = all_ids[idx]
            meta = all_meta[idx]
            results.append(
                SearchResult(
                    chunk_id=doc_id,
                    text=meta.get("text", ""),
                    score=float(fused[idx]),
                    rank=rank,
                    source_doc_id=meta.get("source_doc_id", doc_id),
                )
            )
        return results
