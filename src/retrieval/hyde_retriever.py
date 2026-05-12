from __future__ import annotations

import logging

import numpy as np

from config import SearchResult
from src.embeddings.base import BaseEmbedder
from src.embeddings.vertex_mock import MockGenerativeModel
from src.retrieval.base import BaseRetriever
from src.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class HyDERetriever(BaseRetriever):

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        generative_model: MockGenerativeModel,
        n_hypothetical: int = 3,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._gen_model = generative_model
        self._n_hypothetical = n_hypothetical

        self.last_hypothetical_doc = ""

    def _generate_hypothetical_doc(self, query):
        prompt = (
            "Write a short 3-sentence technical paragraph that directly "
            "answers this question: {}\n"
            "Use precise technical vocabulary about distributed systems."
        ).format(query)
        response = self._gen_model.generate_content(prompt)
        return response.text

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        hyp_docs = []
        for _ in range(self._n_hypothetical):
            hyp_docs.append(self._generate_hypothetical_doc(query))

        self.last_hypothetical_doc = hyp_docs[0]

        hyp_vecs = np.array(
            [self._embedder.embed_query(doc) for doc in hyp_docs],
            dtype=np.float32,
        )
        avg_vec = hyp_vecs.mean(axis=0)

        norm = np.linalg.norm(avg_vec)
        if norm > 0:
            avg_vec = avg_vec / norm

        logger.info("[HyDE] Averaged %d hypothetical embeddings", self._n_hypothetical)

        return self._store.search(avg_vec, top_k)
