"""Mock Vertex AI providers for local dev. Swap for real ones in production."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from src.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class _EmbeddingValue:
    values: list[float]


class MockVertexEmbeddingModel:

    def __init__(self, underlying_embedder: BaseEmbedder) -> None:
        self._embedder = underlying_embedder

    def get_embeddings(self, texts: list[str]) -> list[_EmbeddingValue]:
        matrix = self._embedder.embed(texts)
        return [_EmbeddingValue(values=row.tolist()) for row in matrix]


_EXPANSION_DICT = {
    'peak load': [
        'traffic surge', 'demand spike', 'high throughput',
        'capacity planning', 'horizontal scaling',
    ],
    'scaling': [
        'auto-scaling', 'scale-out', 'elasticity',
        'replica provisioning', 'horizontal expansion',
    ],
    'caching': [
        'distributed cache', 'Redis', 'Memcached',
        'cache-aside', 'write-through', 'TTL expiration',
    ],
    'failure': [
        'fault tolerance', 'circuit breaker', 'failover',
        'degradation', 'resilience', 'recovery',
    ],
    'recovery': [
        'fault recovery', 'self-healing', 'retry',
        'fallback', 'circuit breaker reset',
    ],
    'latency': [
        'response time', 'p99 latency', 'tail latency',
        'inference time', 'round-trip delay',
    ],
    'monitoring': [
        'observability', 'alerting', 'golden signals',
        'distributed tracing', 'log aggregation', 'anomaly detection',
    ],
}


@dataclass
class _GenerationResponse:
    text: str


class MockGenerativeModel:

    def __init__(self, model_name=None) -> None:
        logger.info("MockGenerativeModel initialised (rule-based expansion)")

    def generate_content(self, prompt: str) -> _GenerationResponse:
        prompt_lower = prompt.lower()

        expanded_terms = []
        for keyword, synonyms in _EXPANSION_DICT.items():
            if keyword in prompt_lower:
                expanded_terms.extend(synonyms)

        if not expanded_terms:
            match = re.search(r'["\'](.+?)["\']', prompt)
            fallback = match.group(1) if match else prompt.strip()
            return _GenerationResponse(text=fallback)

        seen = set()
        unique = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique.append(term)

        result = ", ".join(unique)
        logger.info("Expanded query to %d terms: %s", len(unique), result)
        return _GenerationResponse(text=result)
