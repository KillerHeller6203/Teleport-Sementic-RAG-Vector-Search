from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """All embedders must return L2-normalized vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        ...
