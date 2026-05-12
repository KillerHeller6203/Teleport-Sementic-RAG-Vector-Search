from __future__ import annotations

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from config import EMBEDDING_DIM, SearchResult
from src.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS IndexFlatIP backed store. Expects L2-normalized vectors."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._idx_to_id: list[str] = []
        self._metadata: dict[str, dict] = {}

    def add(self, ids: list[str], embeddings: np.ndarray, metadata: list[dict]) -> None:
        if len(ids) != embeddings.shape[0] or len(ids) != len(metadata):
            raise ValueError(
                "Length mismatch: ids={}, embeddings={}, metadata={}".format(
                    len(ids), embeddings.shape[0], len(metadata)
                )
            )
        if embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Dimension mismatch: expected {self._dim}, got {embeddings.shape[1]}"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)

        for doc_id, meta in zip(ids, metadata):
            self._idx_to_id.append(doc_id)
            self._metadata[doc_id] = meta

        logger.info(
            "Added %d vectors — total in store: %d", len(ids), self._index.ntotal
        )

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._index.ntotal == 0:
            raise ValueError("Cannot search an empty FAISS index.")

        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )
        effective_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, effective_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            doc_id = self._idx_to_id[idx]
            meta = self._metadata[doc_id]
            results.append(
                SearchResult(
                    chunk_id=doc_id,
                    text=meta.get("text", ""),
                    score=float(score),
                    rank=rank,
                    source_doc_id=meta.get("source_doc_id", doc_id),
                )
            )
        return results

    def delete(self, ids: list[str]) -> None:
        ids_to_remove = set(ids)
        keep_indices = [
            i for i, doc_id in enumerate(self._idx_to_id)
            if doc_id not in ids_to_remove
        ]

        if len(keep_indices) == len(self._idx_to_id):
            return

        kept_vectors = np.vstack(
            [self._index.reconstruct(i) for i in keep_indices]
        ).astype(np.float32)

        kept_ids = [self._idx_to_id[i] for i in keep_indices]

        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(np.ascontiguousarray(kept_vectors))

        self._idx_to_id = kept_ids
        for doc_id in ids_to_remove:
            self._metadata.pop(doc_id, None)

    def count(self) -> int:
        return self._index.ntotal

    def save_local(self, path: str) -> None:
        index_path = f"{path}.faiss"
        meta_path = f"{path}.meta"

        faiss.write_index(self._index, index_path)
        with open(meta_path, "wb") as fh:
            pickle.dump(
                {"idx_to_id": self._idx_to_id, "metadata": self._metadata}, fh
            )
        logger.info("Saved FAISS index (%d vectors) to %s", self._index.ntotal, index_path)

    def load_local(self, path: str) -> None:
        index_path = Path("{}.faiss".format(path))
        meta_path = Path("{}.meta".format(path))

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self._index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as fh:
            data = pickle.load(fh)
            self._idx_to_id = data["idx_to_id"]
            self._metadata = data["metadata"]
