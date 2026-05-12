from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or RERANKER_MODEL
        logger.info('Loading cross-encoder: %s', self._model_name)
        self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []

        pairs = [[query, r.text] for r in results]
        scores = self._model.predict(pairs)

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        reranked = []
        for rank, (r, score) in enumerate(scored, start=1):
            reranked.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=float(score),
                    rank=rank,
                    source_doc_id=r.source_doc_id,
                )
            )
        return reranked
