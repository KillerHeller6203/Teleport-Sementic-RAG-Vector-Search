from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from config import SearchResult


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, ids: list[str], embeddings: np.ndarray, metadata: list[dict]) -> None:
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...
