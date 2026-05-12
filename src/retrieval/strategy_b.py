from __future__ import annotations

import logging
from collections import defaultdict

from config import SearchResult
from src.embeddings.base import BaseEmbedder
from src.embeddings.vertex_mock import MockGenerativeModel
from src.retrieval.base import BaseRetriever
from src.retrieval.query_expander import QueryExpander
from src.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class StrategyB(BaseRetriever):

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        expander: QueryExpander,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._expander = expander
        self._gen_model = MockGenerativeModel()

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        expanded = self._expander.expand(query)
        q_vec = self._embedder.embed_query(expanded)
        return self._store.search(q_vec, top_k)

    def multi_query_retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        sub_query_prompt = (
            f"Generate 3 different versions of this search query, "
            f"each on a new line. Original query: {query}"
        )
        response = self._gen_model.generate_content(sub_query_prompt)
        raw_parts = [p.strip() for p in response.text.split(",") if p.strip()]

        sub_queries = []
        chunk_size = max(1, len(raw_parts) // 3)
        for i in range(0, len(raw_parts), chunk_size):
            sub_q = ", ".join(raw_parts[i : i + chunk_size])
            sub_queries.append(sub_q)
            if len(sub_queries) == 3:
                break
        while len(sub_queries) < 3:
            sub_queries.append(query)

        score_accum = defaultdict(list)
        best_result = {}

        for sq in sub_queries:
            sq_vec = self._embedder.embed_query(sq)
            results = self._store.search(sq_vec, top_k)
            for r in results:
                score_accum[r.chunk_id].append(r.score)
                if (
                    r.chunk_id not in best_result
                    or r.score > best_result[r.chunk_id].score
                ):
                    best_result[r.chunk_id] = r

        merged = [
            (cid, sum(scores) / len(scores))
            for cid, scores in score_accum.items()
        ]
        merged.sort(key=lambda x: x[1], reverse=True)

        final = []
        for rank, (cid, avg_score) in enumerate(merged[:top_k], start=1):
            r = best_result[cid]
            final.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=avg_score,
                    rank=rank,
                    source_doc_id=r.source_doc_id,
                )
            )
        return final
