from __future__ import annotations

import re

import numpy as np

from config import SearchResult
from src.embeddings.base import BaseEmbedder

_SIMILARITY_THRESHOLD = 0.4


def _split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


class ContextualCompressor:

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def compress(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []

        q_vec = self._embedder.embed_query(query)

        compressed = []
        for r in results:
            sentences = _split_sentences(r.text)
            if not sentences:
                compressed.append(r)
                continue

            sent_vecs = self._embedder.embed(sentences)
            sims = sent_vecs @ q_vec

            kept = [
                s for s, sim in zip(sentences, sims)
                if float(sim) >= _SIMILARITY_THRESHOLD
            ]

            new_text = " ".join(kept) if kept else r.text

            compressed.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    text=new_text,
                    score=r.score,
                    rank=r.rank,
                    source_doc_id=r.source_doc_id,
                )
            )
        return compressed


def lost_in_middle_reorder(results: list[SearchResult]) -> list[SearchResult]:
    if len(results) <= 2:
        return results

    sorted_r = sorted(results, key=lambda x: x.score, reverse=True)
    reordered = []
    left, right = 0, len(sorted_r) - 1
    turn = "left"
    while left <= right:
        if turn == "left":
            reordered.append(sorted_r[left])
            left += 1
            turn = "right"
        else:
            reordered.append(sorted_r[right])
            right -= 1
            turn = "left"

    for i, r in enumerate(reordered):
        r.rank = i + 1

    return reordered
