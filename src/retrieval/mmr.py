from __future__ import annotations

import numpy as np

from config import SearchResult
from src.embeddings.base import BaseEmbedder


def mmr_rerank(
    results: list[SearchResult],
    query_embedding: np.ndarray,
    embedder: BaseEmbedder,
    top_k: int,
    lambda_param: float = 0.5,
) -> list[SearchResult]:
    if not results:
        return []

    n = min(top_k, len(results))

    texts = [r.text for r in results]
    doc_embeddings = embedder.embed(texts)

    q = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
    relevance = doc_embeddings @ q

    selected = []
    remaining = set(range(len(results)))

    for _ in range(n):
        best_idx = -1
        best_score = -np.inf

        for idx in remaining:
            rel = float(relevance[idx])

            if selected:
                sims = doc_embeddings[idx] @ doc_embeddings[selected].T
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0

            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx == -1:
            break

        selected.append(best_idx)
        remaining.discard(best_idx)

    reranked = []
    for rank, idx in enumerate(selected, start=1):
        r = results[idx]
        reranked.append(
            SearchResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                rank=rank,
                source_doc_id=r.source_doc_id,
            )
        )
    return reranked
