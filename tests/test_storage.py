from __future__ import annotations

import numpy as np
import pytest

from config import EMBEDDING_DIM
from src.storage.faiss_store import FAISSVectorStore


class TestFAISSVectorStore:

    def test_add_and_count(self, embedder, vector_store):
        texts = ["document one", "document two"]
        vecs = embedder.embed(texts)
        ids = ["d1", "d2"]
        meta = [{"text": t, "source_doc_id": i} for t, i in zip(texts, ids)]
        vector_store.add(ids, vecs, meta)
        assert vector_store.count() == 2

    def test_search_returns_top_k_results(self, embedder, populated_store):
        query_vec = embedder.embed_query("test query")
        results = populated_store.search(query_vec, top_k=2)
        assert len(results) == 2

    def test_search_returns_sorted_by_score(self, embedder, populated_store):
        query_vec = embedder.embed_query("test query")
        results = populated_store.search(query_vec, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store_raises_valueerror(self, vector_store):
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        with pytest.raises(ValueError, match="empty"):
            vector_store.search(query, top_k=3)

    def test_save_and_load_local(self, embedder, populated_store, tmp_path):
        base = str(tmp_path / "test_index")
        populated_store.save_local(base)

        new_store = FAISSVectorStore(dim=EMBEDDING_DIM)
        new_store.load_local(base)

        assert new_store.count() == populated_store.count()

        query_vec = embedder.embed_query("test query")
        orig_results = populated_store.search(query_vec, top_k=2)
        loaded_results = new_store.search(query_vec, top_k=2)
        assert [r.chunk_id for r in orig_results] == [
            r.chunk_id for r in loaded_results
        ]
