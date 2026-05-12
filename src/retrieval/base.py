from __future__ import annotations

from abc import ABC, abstractmethod

from config import SearchResult


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        ...
