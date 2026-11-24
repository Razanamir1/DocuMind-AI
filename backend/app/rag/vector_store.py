"""Persistent ChromaDB vector database wrapper for DocuMind AI."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.errors import LLMProviderError
from app.rag.chunker import TextChunk


@dataclass(frozen=True)
class VectorMatch:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
    vector_score: float


class ChromaVectorStore:
    """Manages a persistent local ChromaDB collection in cosine similarity space."""

    def __init__(self, persist_dir: Path, collection_name: str = "documind_chunks") -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._cached_dimension: int | None = None

    def count(self) -> int:
        with self._lock:
            return int(self._collection.count())

    def get_embedding_dimension(self) -> int | None:
        with self._lock:
            if self._cached_dimension is not None and self.count() > 0:
                return self._cached_dimension
            if self.count() == 0:
                self._cached_dimension = None
                return None
            sample = self._collection.peek(limit=1)
            embs = sample.get("embeddings")
            if embs is not None and len(embs) > 0:
                first = embs[0]
                if first is not None and len(first) > 0:
                    self._cached_dimension = int(len(first))
                    return self._cached_dimension
            return None

    def _validate_dimension(self, vectors: list[list[float]]) -> None:
        if not vectors:
            return
        dims = {len(v) for v in vectors}
        if len(dims) != 1 or 0 in dims:
            raise LLMProviderError(
                code="EMBEDDING_DIMENSION_INVALID",
                message="The embedding provider returned inconsistent or empty vector dimensions.",
                status_code=502,
            )
        incoming_dim = next(iter(dims))
        existing_dim = self.get_embedding_dimension()
        if existing_dim is not None and existing_dim != incoming_dim:
            raise LLMProviderError(
                code="EMBEDDING_DIMENSION_MISMATCH",
                message=(
                    f"Embedding dimensionality mismatch: collection expects {existing_dim}-dim vectors, "
                    f"but the configured embedding model returned {incoming_dim}-dim vectors. "
                    "Re-index documents or align GEMINI_EMBEDDING_MODEL."
                ),
                status_code=409,
            )

    def has_document_vectors(self, document_id: str) -> bool:
        with self._lock:
            result = self._collection.get(where={"document_id": document_id}, limit=1)
            return len(result.get("ids") or []) > 0

    def upsert_document_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match.")

        with self._lock:
            self._validate_dimension(embeddings)
            # Replace any prior chunks for this document atomically under lock so retries
            # (including after chunk-size changes) never leave orphaned or duplicate vectors.
            self._collection.delete(where={"document_id": document_id})
            ids = [f"{document_id}-chunk-{c.chunk_index}" for c in chunks]
            documents = [c.content for c in chunks]
            metadatas: list[dict[str, Any]] = [
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": int(c.page_number),
                    "chunk_index": int(c.chunk_index),
                    "char_start": int(c.char_start),
                    "char_end": int(c.char_end),
                }
                for c in chunks
            ]

            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            self._cached_dimension = len(embeddings[0])

    def query(
        self,
        *,
        query_embedding: list[float],
        n_results: int = 8,
        document_ids: list[str] | None = None,
    ) -> list[VectorMatch]:
        with self._lock:
            total = self.count()
            if total == 0:
                return []
            self._validate_dimension([query_embedding])

            where_filter: dict[str, Any] | None = None
            if document_ids:
                if len(document_ids) == 1:
                    where_filter = {"document_id": document_ids[0]}
                else:
                    where_filter = {"document_id": {"$in": document_ids}}

            k = min(max(1, n_results), total)
            raw = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        matches: list[VectorMatch] = []
        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
            if not meta or doc_text is None:
                continue
            # ChromaDB cosine distance = 1 - cosine_similarity; convert to bounded similarity [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            matches.append(
                VectorMatch(
                    chunk_id=str(chunk_id),
                    document_id=str(meta["document_id"]),
                    filename=str(meta["filename"]),
                    page_number=int(meta["page_number"]),
                    chunk_index=int(meta["chunk_index"]),
                    content=str(doc_text),
                    vector_score=round(similarity, 4),
                )
            )
        return matches

    def get_representative_chunks(
        self,
        *,
        document_ids: list[str] | None = None,
        limit: int = 5,
    ) -> list[VectorMatch]:
        """Samples representative chunks evenly across active documents and pages for summary/overview queries."""
        with self._lock:
            if self.count() == 0:
                return []

            where_filter: dict[str, Any] | None = None
            if document_ids:
                if len(document_ids) == 1:
                    where_filter = {"document_id": document_ids[0]}
                else:
                    where_filter = {"document_id": {"$in": document_ids}}

            raw = self._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )

        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []

        by_doc: dict[str, list[VectorMatch]] = {}
        for chunk_id, doc_text, meta in zip(ids, docs, metas):
            if not meta or doc_text is None:
                continue
            doc_id = str(meta["document_id"])
            by_doc.setdefault(doc_id, []).append(
                VectorMatch(
                    chunk_id=str(chunk_id),
                    document_id=doc_id,
                    filename=str(meta["filename"]),
                    page_number=int(meta["page_number"]),
                    chunk_index=int(meta["chunk_index"]),
                    content=str(doc_text),
                    vector_score=0.85,
                )
            )

        if not by_doc:
            return []

        max_items = max(1, min(12, limit))
        ordered_doc_ids = (
            [d for d in document_ids if d in by_doc]
            if document_ids
            else list(by_doc.keys())
        )
        for doc_id in ordered_doc_ids:
            by_doc[doc_id].sort(key=lambda m: (m.page_number, m.chunk_index))

        # Build evenly spaced per-document sample queues (always starting with page 1 / chunk 0)
        queues: list[list[VectorMatch]] = []
        per_doc_budget = max(1, (max_items + len(ordered_doc_ids) - 1) // len(ordered_doc_ids))
        for doc_id in ordered_doc_ids:
            chunks = by_doc[doc_id]
            if len(chunks) <= per_doc_budget:
                queues.append(list(chunks))
            elif per_doc_budget == 1:
                queues.append([chunks[0]])
            else:
                picked_indices: list[int] = []
                step = (len(chunks) - 1) / (per_doc_budget - 1)
                for idx in range(per_doc_budget):
                    pos = round(idx * step)
                    if pos not in picked_indices:
                        picked_indices.append(pos)
                queues.append([chunks[p] for p in picked_indices])

        # Round-robin across documents so multi-document summaries cover every scoped file
        selected: list[VectorMatch] = []
        round_idx = 0
        while len(selected) < max_items:
            added_any = False
            for q in queues:
                if round_idx < len(q):
                    selected.append(q[round_idx])
                    added_any = True
                    if len(selected) >= max_items:
                        break
            if not added_any:
                break
            round_idx += 1

        return selected

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._collection.delete(where={"document_id": document_id})
            if self.count() == 0:
                self._cached_dimension = None


_vector_store_singleton: ChromaVectorStore | None = None
_vector_store_lock = threading.Lock()


def get_vector_store() -> ChromaVectorStore:
    global _vector_store_singleton
    if _vector_store_singleton is None:
        with _vector_store_lock:
            if _vector_store_singleton is None:
                _vector_store_singleton = ChromaVectorStore(get_settings().chroma_path)
    return _vector_store_singleton
