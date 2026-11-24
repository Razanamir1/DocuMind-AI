"""Pydantic v2 schemas matching the DocuMind AI REST API v1 contract."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator

ChatMode = Literal["general", "documents"]
DocumentStatus = Literal["uploading", "processing", "ready", "failed"]
EmbeddingStatus = Literal["vector", "lexical_fallback", "pending"]


class CitationSchema(BaseModel):
    id: str
    documentId: str
    filename: str
    pageNumber: int = Field(ge=1)
    excerpt: str
    score: float
    chunkIndex: int = Field(ge=0)


class ConversationSchema(BaseModel):
    id: str
    title: str
    mode: ChatMode
    createdAt: str
    updatedAt: str
    messageCount: int | None = None
    lastMessagePreview: str | None = None


class MessageSchema(BaseModel):
    id: str
    conversationId: str
    role: Literal["user", "assistant"]
    content: str
    mode: ChatMode
    citations: list[CitationSchema] = Field(default_factory=list)
    insufficientEvidence: bool = False
    createdAt: str


class DocumentSchema(BaseModel):
    id: str
    filename: str
    storedFilename: str
    sha256: str
    fileSizeBytes: int
    pageCount: int
    chunkCount: int
    status: DocumentStatus
    embeddingStatus: EmbeddingStatus
    errorMessage: str | None = None
    createdAt: str
    updatedAt: str


class DocumentPageSchema(BaseModel):
    documentId: str
    pageNumber: int
    content: str


class WorkspaceSettingsSchema(BaseModel):
    topK: int = Field(default=5, ge=1, le=12)
    minRelevanceScore: float = Field(default=0.34, ge=0.15, le=0.85)
    chunkSize: int = Field(default=850, ge=300, le=2000)
    chunkOverlap: int = Field(default=160, ge=0, le=400)
    streamResponses: bool = True


class WorkspaceSettingsPatchSchema(BaseModel):
    topK: int | None = Field(default=None, ge=1, le=12)
    minRelevanceScore: float | None = Field(default=None, ge=0.15, le=0.85)
    chunkSize: int | None = Field(default=None, ge=300, le=2000)
    chunkOverlap: int | None = Field(default=None, ge=0, le=400)
    streamResponses: bool | None = None


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    mode: ChatMode = "general"


class UpdateConversationRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    mode: ChatMode | None = None


class ChatRequest(BaseModel):
    conversationId: str | None = Field(default=None, min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=12000)
    mode: ChatMode = "general"
    documentIds: list[str] | None = Field(default=None, max_length=50)
    stream: bool = True
    regenerate: bool = False

    @field_validator("message")
    @classmethod
    def validate_non_blank_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Please enter a message before sending.")
        return cleaned


class ProviderInfoSchema(BaseModel):
    name: str
    chatModel: str
    embeddingModel: str
    configured: bool
    backendEngine: str = "Python FastAPI + ChromaDB"


IndexingState = Literal["ready", "indexing", "pending", "degraded"]


class HealthStatsSchema(BaseModel):
    documentsTotal: int
    documentsReady: int
    documentsIndexed: int = 0
    documentsPending: int = 0
    indexingState: IndexingState = "ready"
    storageMode: Literal["ephemeral-local"] = "ephemeral-local"
    lastIndexingError: str | None = None
    totalChunks: int
    chromaVectorCount: int
    embeddingDimension: int | None = None
    conversationsTotal: int


class HealthResponseSchema(BaseModel):
    status: str
    service: str
    timestamp: str
    provider: ProviderInfoSchema
    stats: HealthStatsSchema
