"""Document ingestion, ChromaDB indexing, retrieval backfill, and deletion services."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import DocumentError, LLMProviderError
from app.models.entities import DocumentORM, DocumentPageORM, WorkspaceSettingsORM, utcnow
from app.providers.llm import LLMProvider
from app.rag.chunker import chunk_pages
from app.rag.pdf_processor import (
    PageText,
    extract_pdf_pages,
    resolve_safe_upload_path,
    sanitize_filename,
)
from app.rag.vector_store import ChromaVectorStore
from app.schemas.api_schemas import DocumentPageSchema, DocumentSchema


def document_to_schema(doc: DocumentORM) -> DocumentSchema:
    return DocumentSchema(
        id=doc.id,
        filename=doc.filename,
        storedFilename=doc.stored_filename,
        sha256=doc.sha256,
        fileSizeBytes=doc.file_size_bytes,
        pageCount=doc.page_count,
        chunkCount=doc.chunk_count,
        status=doc.status,  # type: ignore[arg-type]
        embeddingStatus=doc.embedding_status,  # type: ignore[arg-type]
        errorMessage=doc.error_message,
        createdAt=doc.created_at.isoformat(),
        updatedAt=doc.updated_at.isoformat(),
    )


def get_workspace_settings_row(db: Session) -> WorkspaceSettingsORM:
    row = db.get(WorkspaceSettingsORM, 1)
    if row is None:
        row = WorkspaceSettingsORM(id=1)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def list_documents(
    db: Session, *, query: str | None = None, status: str | None = None
) -> list[DocumentSchema]:
    q = db.query(DocumentORM)
    if query and query.strip():
        q = q.filter(DocumentORM.filename.ilike(f"%{query.strip()}%"))
    if status and status in {"uploading", "processing", "ready", "failed"}:
        q = q.filter(DocumentORM.status == status)
    rows = q.order_by(DocumentORM.created_at.desc()).all()
    return [document_to_schema(r) for r in rows]


def get_document_orm(db: Session, document_id: str) -> DocumentORM | None:
    return db.get(DocumentORM, document_id)


def get_document_page(
    db: Session, document_id: str, page_number: int
) -> tuple[DocumentSchema, DocumentPageSchema]:
    doc = db.get(DocumentORM, document_id)
    if not doc:
        raise DocumentError("DOCUMENT_NOT_FOUND", "Document not found.", 404)
    if page_number < 1 or page_number > doc.page_count:
        raise DocumentError("INVALID_PAGE", "Requested page number is out of range.", 400)

    page = (
        db.query(DocumentPageORM)
        .filter(
            DocumentPageORM.document_id == document_id,
            DocumentPageORM.page_number == page_number,
        )
        .first()
    )
    if not page:
        raise DocumentError("PAGE_NOT_FOUND", "Extracted text for this page was not found.", 404)

    return (
        document_to_schema(doc),
        DocumentPageSchema(
            documentId=page.document_id,
            pageNumber=page.page_number,
            content=page.content,
        ),
    )


def _embed_chunks_in_batches(
    provider: LLMProvider, texts: list[str], batch_size: int = 20
) -> list[list[float]]:
    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_vectors.extend(provider.embed_texts(batch))
    return all_vectors


def ingest_pdf_document(
    db: Session,
    *,
    provider: LLMProvider,
    vector_store: ChromaVectorStore,
    upload_dir: Path,
    original_filename: str,
    content_type: str | None,
    pdf_bytes: bytes,
) -> DocumentSchema:
    """
    Full real PDF RAG ingestion pipeline:
    1. Validate file extension, MIME type, size limit, and SHA-256 uniqueness
    2. Extract page-by-page text via PyMuPDF (reject corrupted, encrypted, empty, or scanned PDFs)
    3. Split into overlapping chunks with page metadata
    4. Generate real vector embeddings via configured LLMProvider (Gemini)
    5. Store vectors and metadata in persistent ChromaDB
    6. Persist binary safely on disk and document/page metadata in SQLite
    """
    settings = get_settings()
    if not original_filename or not original_filename.lower().endswith(".pdf"):
        raise DocumentError(
            "UNSUPPORTED_FILE_TYPE",
            "Only PDF (.pdf) documents are supported for Knowledge Base indexing.",
            415,
        )

    allowed_mimes = {
        None,
        "",
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    }
    if content_type not in allowed_mimes:
        raise DocumentError(
            "UNSUPPORTED_FILE_TYPE",
            "Only PDF (.pdf) documents are supported for Knowledge Base indexing.",
            415,
        )

    if not pdf_bytes:
        raise DocumentError("EMPTY_FILE", "The uploaded PDF file is empty.", 400)

    if len(pdf_bytes) > settings.max_file_size_bytes:
        raise DocumentError(
            "FILE_TOO_LARGE",
            f"PDF exceeds the {settings.max_file_size_mb} MB maximum file size limit.",
            413,
        )

    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    duplicate = db.query(DocumentORM).filter(DocumentORM.sha256 == sha256).first()
    if duplicate:
        raise DocumentError(
            "DUPLICATE_DOCUMENT",
            f'This document has already been uploaded as "{duplicate.filename}".',
            409,
        )

    # Extract text with PyMuPDF before storing so invalid/scanned files fail cleanly
    extracted = extract_pdf_pages(pdf_bytes)
    ws_settings = get_workspace_settings_row(db)
    chunks = chunk_pages(
        extracted.pages,
        chunk_size=ws_settings.chunk_size,
        chunk_overlap=ws_settings.chunk_overlap,
    )
    if not chunks:
        raise DocumentError(
            "EMPTY_PDF",
            "No indexable text chunks could be extracted from this PDF.",
            422,
        )

    # Generate real embeddings (fail honestly if provider is missing key or unavailable)
    embeddings = _embed_chunks_in_batches(provider, [c.content for c in chunks])

    doc_id = f"doc-{uuid.uuid4()}"
    safe_name = sanitize_filename(original_filename)
    stored_filename = f"{doc_id}.pdf"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = resolve_safe_upload_path(upload_dir, stored_filename)
    target_path.write_bytes(pdf_bytes)

    try:
        vector_store.upsert_document_chunks(
            document_id=doc_id,
            filename=safe_name,
            chunks=chunks,
            embeddings=embeddings,
        )

        now = utcnow()
        doc = DocumentORM(
            id=doc_id,
            filename=safe_name,
            stored_filename=stored_filename,
            sha256=sha256,
            file_size_bytes=len(pdf_bytes),
            page_count=extracted.page_count,
            chunk_count=len(chunks),
            status="ready",
            embedding_status="vector",
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        db.add(doc)
        for page in extracted.pages:
            db.add(
                DocumentPageORM(
                    document_id=doc_id,
                    page_number=page.page_number,
                    content=page.text,
                )
            )
        db.commit()
        db.refresh(doc)
        return document_to_schema(doc)
    except Exception:
        db.rollback()
        try:
            vector_store.delete_document(doc_id)
        except Exception:
            pass
        if target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise


@dataclass
class _IndexingCoordinatorState:
    in_progress: bool = False
    last_error_code: str | None = None
    attempts: int = 0


_backfill_lock = threading.Lock()
_state_lock = threading.Lock()
_indexing_state = _IndexingCoordinatorState()


def reset_indexing_coordinator_state() -> None:
    """Resets in-memory coordinator diagnostics (useful in isolated tests)."""
    with _state_lock:
        _indexing_state.in_progress = False
        _indexing_state.last_error_code = None
        _indexing_state.attempts = 0


def get_indexing_readiness(
    db: Session,
    *,
    vector_store: ChromaVectorStore,
) -> dict:
    """Computes accurate vector indexing readiness from SQLite + ChromaDB."""
    ready_docs = db.query(DocumentORM).filter(DocumentORM.status == "ready").all()
    indexed_count = 0
    pending_count = 0
    for doc in ready_docs:
        if doc.embedding_status == "vector" and vector_store.has_document_vectors(doc.id):
            indexed_count += 1
        else:
            pending_count += 1

    with _state_lock:
        in_progress = _indexing_state.in_progress
        last_error = _indexing_state.last_error_code

    if in_progress:
        state = "indexing"
    elif pending_count == 0:
        state = "ready"
    elif last_error:
        state = "degraded"
    else:
        state = "pending"

    return {
        "documentsIndexed": indexed_count,
        "documentsPending": pending_count,
        "indexingState": state,
        "lastIndexingError": last_error if pending_count > 0 else None,
    }


def ensure_documents_indexed_in_chroma(
    db: Session,
    *,
    provider: LLMProvider,
    vector_store: ChromaVectorStore,
    document_ids: list[str] | None = None,
) -> list[str]:
    """
    Idempotently ensures requested ready documents have live vector embeddings in ChromaDB.
    Serialized via _backfill_lock so concurrent requests or retries never race or create
    duplicate vectors. Preserves 'pending' status on provider failure so work is never lost.
    """
    with _backfill_lock:
        q = db.query(DocumentORM).filter(DocumentORM.status == "ready")
        if document_ids:
            q = q.filter(DocumentORM.id.in_(document_ids))
        docs = q.all()
        if not docs:
            return []

        ws_settings = get_workspace_settings_row(db)
        ready_ids: list[str] = [doc.id for doc in docs]
        pending_docs = [
            doc
            for doc in docs
            if not (doc.embedding_status == "vector" and vector_store.has_document_vectors(doc.id))
        ]
        if not pending_docs:
            with _state_lock:
                _indexing_state.last_error_code = None
            return ready_ids

        with _state_lock:
            _indexing_state.in_progress = True
            _indexing_state.attempts += 1

        try:
            for doc in pending_docs:
                pages = [
                    PageText(page_number=p.page_number, text=p.content)
                    for p in sorted(doc.pages, key=lambda item: item.page_number)
                ]
                chunks = chunk_pages(
                    pages,
                    chunk_size=ws_settings.chunk_size,
                    chunk_overlap=ws_settings.chunk_overlap,
                )
                if not chunks:
                    continue

                embeddings = _embed_chunks_in_batches(provider, [c.content for c in chunks])
                vector_store.upsert_document_chunks(
                    document_id=doc.id,
                    filename=doc.filename,
                    chunks=chunks,
                    embeddings=embeddings,
                )
                doc.chunk_count = len(chunks)
                doc.embedding_status = "vector"
                doc.error_message = None
                doc.updated_at = utcnow()
                db.commit()

            with _state_lock:
                _indexing_state.last_error_code = None
        except LLMProviderError as exc:
            db.rollback()
            with _state_lock:
                _indexing_state.last_error_code = exc.code
            raise
        except Exception as exc:
            db.rollback()
            with _state_lock:
                _indexing_state.last_error_code = type(exc).__name__
            raise
        finally:
            with _state_lock:
                _indexing_state.in_progress = False

        return ready_ids


def delete_document(
    db: Session,
    *,
    vector_store: ChromaVectorStore,
    upload_dir: Path,
    document_id: str,
) -> bool:
    """Deletes a document from SQLite, ChromaDB vector index, and disk storage."""
    doc = db.get(DocumentORM, document_id)
    if not doc:
        return False

    vector_store.delete_document(document_id)

    try:
        file_path = resolve_safe_upload_path(upload_dir, doc.stored_filename)
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

    db.delete(doc)
    db.commit()
    return True
