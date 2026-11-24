"""REST API v1 routes for DocuMind AI (Chat SSE/JSON, Conversations, Documents, Settings, Health)."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import ChatError, DocumentError
from app.db.session import get_db
from app.models.entities import ConversationORM, DocumentORM
from app.providers.llm import LLMProvider, get_llm_provider
from app.rag.pdf_processor import resolve_safe_upload_path
from app.rag.vector_store import ChromaVectorStore, get_vector_store
from app.schemas.api_schemas import (
    ChatRequest,
    CreateConversationRequest,
    HealthResponseSchema,
    HealthStatsSchema,
    ProviderInfoSchema,
    UpdateConversationRequest,
    WorkspaceSettingsPatchSchema,
)
from app.services import chat_service, document_service

router = APIRouter(prefix="/api/v1", tags=["DocuMind AI v1"])


def get_provider_dep() -> LLMProvider:
    return get_llm_provider()


def get_vector_store_dep() -> ChromaVectorStore:
    return get_vector_store()


# ==================== System & Settings ====================


@router.get("/health", response_model=HealthResponseSchema)
def health(
    db: Session = Depends(get_db),
    provider: LLMProvider = Depends(get_provider_dep),
    vector_store: ChromaVectorStore = Depends(get_vector_store_dep),
):
    docs = db.query(DocumentORM).all()
    convos_count = db.query(ConversationORM).count()
    readiness = document_service.get_indexing_readiness(db, vector_store=vector_store)
    return HealthResponseSchema(
        status="ok",
        service="DocuMind AI FastAPI Backend",
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=ProviderInfoSchema(
            name=provider.provider_name,
            chatModel=provider.chat_model,
            embeddingModel=provider.embedding_model,
            configured=provider.is_configured(),
            backendEngine="Python FastAPI + ChromaDB",
        ),
        stats=HealthStatsSchema(
            documentsTotal=len(docs),
            documentsReady=sum(1 for d in docs if d.status == "ready"),
            documentsIndexed=readiness["documentsIndexed"],
            documentsPending=readiness["documentsPending"],
            indexingState=readiness["indexingState"],
            storageMode="ephemeral-local",
            lastIndexingError=readiness["lastIndexingError"],
            totalChunks=sum(int(d.chunk_count or 0) for d in docs),
            chromaVectorCount=vector_store.count(),
            embeddingDimension=vector_store.get_embedding_dimension(),
            conversationsTotal=convos_count,
        ),
    )


@router.get("/settings")
def get_workspace_settings(
    db: Session = Depends(get_db),
    provider: LLMProvider = Depends(get_provider_dep),
):
    return {
        "settings": chat_service.get_settings_schema(db).model_dump(),
        "provider": ProviderInfoSchema(
            name=provider.provider_name,
            chatModel=provider.chat_model,
            embeddingModel=provider.embedding_model,
            configured=provider.is_configured(),
            backendEngine="Python FastAPI + ChromaDB",
        ).model_dump(),
    }


@router.patch("/settings")
def patch_workspace_settings(
    payload: WorkspaceSettingsPatchSchema,
    db: Session = Depends(get_db),
):
    updated = chat_service.update_settings(db, payload)
    return {"settings": updated.model_dump()}


# ==================== Conversations ====================


@router.get("/conversations")
def list_conversations(db: Session = Depends(get_db)):
    conversations = chat_service.list_conversations(db)
    return {"conversations": [c.model_dump() for c in conversations]}


@router.post("/conversations", status_code=201)
def create_conversation(
    payload: CreateConversationRequest,
    db: Session = Depends(get_db),
):
    convo = chat_service.create_conversation(
        db, title=payload.title, mode=payload.mode
    )
    return {"conversation": convo.model_dump()}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    found = chat_service.get_conversation_with_messages(db, conversation_id)
    if not found:
        raise ChatError("CONVERSATION_NOT_FOUND", "Conversation not found.", 404)
    convo, messages = found
    return {
        "conversation": convo.model_dump(),
        "messages": [m.model_dump() for m in messages],
    }


@router.patch("/conversations/{conversation_id}")
def patch_conversation(
    conversation_id: str,
    payload: UpdateConversationRequest,
    db: Session = Depends(get_db),
):
    updated = chat_service.update_conversation(
        db, conversation_id, title=payload.title, mode=payload.mode
    )
    if not updated:
        raise ChatError("CONVERSATION_NOT_FOUND", "Conversation not found.", 404)
    return {"conversation": updated.model_dump()}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    deleted = chat_service.delete_conversation(db, conversation_id)
    if not deleted:
        raise ChatError("CONVERSATION_NOT_FOUND", "Conversation not found.", 404)
    return {"deleted": True, "id": conversation_id}


@router.post("/conversations/{conversation_id}/clear")
def clear_conversation(conversation_id: str, db: Session = Depends(get_db)):
    cleared = chat_service.clear_conversation(db, conversation_id)
    if not cleared:
        raise ChatError("CONVERSATION_NOT_FOUND", "Conversation not found.", 404)
    return {"cleared": True, "id": conversation_id}


# ==================== Chat (JSON & Streaming SSE) ====================


@router.post("/chat")
def chat_endpoint(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    provider: LLMProvider = Depends(get_provider_dep),
    vector_store: ChromaVectorStore = Depends(get_vector_store_dep),
):
    if not payload.stream:
        return chat_service.execute_chat_turn(
            db, provider=provider, vector_store=vector_store, req=payload
        )

    prepared = chat_service.prepare_chat_turn(
        db, provider=provider, vector_store=vector_store, req=payload
    )

    return StreamingResponse(
        chat_service.stream_chat_turn_sse(
            db, provider=provider, vector_store=vector_store, prepared=prepared
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== Documents & PDF Streaming ====================


@router.get("/documents")
def list_documents(
    q: str | None = Query(default=None),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    docs = document_service.list_documents(db, query=q, status=status)
    return {"documents": [d.model_dump() for d in docs]}


@router.get("/documents/{document_id}")
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = document_service.get_document_orm(db, document_id)
    if not doc:
        raise DocumentError("DOCUMENT_NOT_FOUND", "Document not found.", 404)
    return {"document": document_service.document_to_schema(doc).model_dump()}


@router.post("/documents/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    provider: LLMProvider = Depends(get_provider_dep),
    vector_store: ChromaVectorStore = Depends(get_vector_store_dep),
):
    settings = get_settings()
    # Bound in-memory read to MAX_FILE_SIZE_MB + 1 byte to protect container RAM
    raw_bytes = await file.read(settings.max_file_size_bytes + 1)
    doc = document_service.ingest_pdf_document(
        db,
        provider=provider,
        vector_store=vector_store,
        upload_dir=settings.upload_path,
        original_filename=file.filename or "",
        content_type=file.content_type,
        pdf_bytes=raw_bytes,
    )
    return {"document": doc.model_dump()}


@router.post("/documents/reindex")
def reindex_pending_documents(
    db: Session = Depends(get_db),
    provider: LLMProvider = Depends(get_provider_dep),
    vector_store: ChromaVectorStore = Depends(get_vector_store_dep),
):
    """Idempotently backfills any pending ready documents into ChromaDB on demand."""
    document_service.ensure_documents_indexed_in_chroma(
        db,
        provider=provider,
        vector_store=vector_store,
    )
    readiness = document_service.get_indexing_readiness(db, vector_store=vector_store)
    return {
        "documentsIndexed": readiness["documentsIndexed"],
        "documentsPending": readiness["documentsPending"],
        "indexingState": readiness["indexingState"],
        "chromaVectorCount": vector_store.count(),
    }


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    vector_store: ChromaVectorStore = Depends(get_vector_store_dep),
):
    settings = get_settings()
    deleted = document_service.delete_document(
        db,
        vector_store=vector_store,
        upload_dir=settings.upload_path,
        document_id=document_id,
    )
    if not deleted:
        raise DocumentError("DOCUMENT_NOT_FOUND", "Document not found.", 404)
    return {"deleted": True, "id": document_id}


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str, db: Session = Depends(get_db)):
    doc = document_service.get_document_orm(db, document_id)
    if not doc:
        raise DocumentError("DOCUMENT_NOT_FOUND", "Document not found.", 404)
    settings = get_settings()
    file_path = resolve_safe_upload_path(settings.upload_path, doc.stored_filename)
    if not file_path.exists():
        raise DocumentError(
            "FILE_UNAVAILABLE",
            "The source PDF binary is no longer available on disk.",
            404,
        )
    safe_header_name = doc.filename.replace('"', "")
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{safe_header_name}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/documents/{document_id}/pages/{page_number}")
def get_document_page(
    document_id: str, page_number: int, db: Session = Depends(get_db)
):
    doc, page = document_service.get_document_page(db, document_id, page_number)
    return {"document": doc.model_dump(), "page": page.model_dump()}
