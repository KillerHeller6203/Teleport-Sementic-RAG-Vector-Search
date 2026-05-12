from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from src.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


class LocalEmbedder(BaseEmbedder):

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or EMBEDDING_MODEL
        logger.info("Loading sentence-transformer model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._model.encode([query], show_progress_bar=False)
        vec = np.asarray(vec, dtype=np.float32)
        return _l2_normalize(vec)[0]
