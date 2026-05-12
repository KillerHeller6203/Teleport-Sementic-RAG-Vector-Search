from __future__ import annotations

import logging
import time

import numpy as np

from config import CHUNK_OVERLAP, CHUNK_SIZE
from src.embeddings.base import BaseEmbedder
from src.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Recursive character splitter with overlap."""

    _SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, doc: dict) -> list[dict]:
        doc_id = doc["id"]
        title = doc["title"]
        text = doc["text"]

        raw_chunks = self._recursive_split(text, self._SEPARATORS)
        merged = self._merge_with_overlap(raw_chunks)

        chunks = []
        for i, chunk_text in enumerate(merged):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "source_doc_id": doc_id,
                    "title": title,
                    "text": chunk_text,
                    "chunk_index": i,
                }
            )

        logger.info(
            "Chunked '%s' -> %d chunk(s)", doc_id, len(chunks)
        )
        return chunks

    def _recursive_split(self, text, separators):
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        if not separators:
            pieces = []
            for start in range(0, len(text), self._chunk_size):
                piece = text[start : start + self._chunk_size].strip()
                if piece:
                    pieces.append(piece)
            return pieces

        sep = separators[0]
        parts = text.split(sep)
        result = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self._chunk_size:
                result.append(part)
            else:
                result.extend(
                    self._recursive_split(part, separators[1:])
                )
        return result

    def _merge_with_overlap(self, pieces):
        if not pieces:
            return []

        merged = []
        current = pieces[0]

        for piece in pieces[1:]:
            candidate = current + " " + piece
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                merged.append(current)
                if self._chunk_overlap > 0 and len(current) > self._chunk_overlap:
                    overlap_text = current[-self._chunk_overlap :]
                    current = overlap_text.strip() + " " + piece
                else:
                    current = piece

        if current.strip():
            merged.append(current)
        return merged


class IngestionPipeline:
    """Chunk -> embed -> store."""

    def __init__(
        self,
        chunker: DocumentChunker,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def ingest(self, documents: list[dict]) -> dict:
        logger.info("Starting ingestion of %d document(s)...", len(documents))

        all_chunks = []
        for doc in documents:
            all_chunks.extend(self._chunker.chunk(doc))

        logger.info("Total chunks: %d", len(all_chunks))

        if not all_chunks:
            logger.warning("No chunks produced — nothing to ingest.")
            return {
                "documents_ingested": len(documents),
                "chunks_created": 0,
                "embedding_time_ms": 0.0,
            }

        texts = [c["text"] for c in all_chunks]
        t0 = time.perf_counter()
        embeddings = self._embedder.embed(texts)
        embed_ms = (time.perf_counter() - t0) * 1000

        logger.info("Embedding done in %.1f ms", embed_ms)

        ids = [c["chunk_id"] for c in all_chunks]
        metadata = [
            {
                "text": c["text"],
                "source_doc_id": c["source_doc_id"],
                "title": c["title"],
                "chunk_index": c["chunk_index"],
            }
            for c in all_chunks
        ]
        self._store.add(ids, embeddings, metadata)

        logger.info(
            "Ingestion complete — %d docs, %d chunks, %.1f ms",
            len(documents), len(all_chunks), embed_ms,
        )

        return {
            "documents_ingested": len(documents),
            "chunks_created": len(all_chunks),
            "embedding_time_ms": round(embed_ms, 2),
        }
