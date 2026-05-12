"""
Tests for the embeddings sub-package.

Covers the mock local embedder, the Vertex AI mock provider,
dimensionality consistency, batch encoding correctness, and adherence to the
base embedding interface.
"""

from __future__ import annotations

import numpy as np

from config import EMBEDDING_DIM
from src.embeddings.vertex_mock import MockGenerativeModel, MockVertexEmbeddingModel


class TestEmbedder:
    """Tests for BaseEmbedder implementations."""

    def test_embed_returns_correct_shape(self, embedder):
        texts = ["hello world", "foo bar", "test document"]
        result = embedder.embed(texts)
        assert result.shape == (3, EMBEDDING_DIM)

    def test_embed_returns_normalized_vectors(self, embedder):
        texts = ["normalize me", "check my norm"]
        result = embedder.embed(texts)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_query_returns_1d_array(self, embedder):
        result = embedder.embed_query("single query")
        assert result.ndim == 1
        assert result.shape == (EMBEDDING_DIM,)


class TestVertexMock:
    """Tests for the Vertex AI mock classes."""

    def test_vertex_mock_returns_values_attribute(self, embedder):
        mock_model = MockVertexEmbeddingModel(underlying_embedder=embedder)
        results = mock_model.get_embeddings(["test text"])
        assert len(results) == 1
        assert hasattr(results[0], "values")
        assert isinstance(results[0].values, list)
        assert len(results[0].values) == EMBEDDING_DIM

    def test_mock_generative_model_expands_query(self):
        model = MockGenerativeModel()
        query = "peak load"
        response = model.generate_content(query)
        # Expanded output should be longer than the original query
        assert len(response.text) > len(query)
        assert "," in response.text  # comma-separated terms
