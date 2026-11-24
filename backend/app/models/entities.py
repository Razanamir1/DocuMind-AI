"""SQLAlchemy ORM entities for conversations, messages, documents, pages, and settings."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationORM(Base):
    __tablename__ = "conversations"

    id = Column(String(120), primary_key=True, default=lambda: f"conv-{uuid.uuid4()}")
    title = Column(String(160), nullable=False, default="New Conversation")
    mode = Column(String(20), nullable=False, default="general")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    messages = relationship(
        "MessageORM",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageORM.created_at.asc()",
    )


class MessageORM(Base):
    __tablename__ = "messages"

    id = Column(String(120), primary_key=True, default=lambda: f"msg-{uuid.uuid4()}")
    conversation_id = Column(
        String(120),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    mode = Column(String(20), nullable=False, default="general")
    citations_json = Column(Text, nullable=False, default="[]")
    insufficient_evidence = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    conversation = relationship("ConversationORM", back_populates="messages")


class DocumentORM(Base):
    __tablename__ = "documents"

    id = Column(String(120), primary_key=True, default=lambda: f"doc-{uuid.uuid4()}")
    filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True, index=True)
    file_size_bytes = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)
    status = Column(String(20), nullable=False, default="processing")
    embedding_status = Column(String(30), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    pages = relationship(
        "DocumentPageORM",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentPageORM.page_number.asc()",
    )


class DocumentPageORM(Base):
    __tablename__ = "document_pages"

    document_id = Column(
        String(120),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    page_number = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)

    document = relationship("DocumentORM", back_populates="pages")


class WorkspaceSettingsORM(Base):
    __tablename__ = "workspace_settings"

    id = Column(Integer, primary_key=True, default=1)
    top_k = Column(Integer, nullable=False, default=5)
    min_relevance_score = Column(Float, nullable=False, default=0.34)
    chunk_size = Column(Integer, nullable=False, default=850)
    chunk_overlap = Column(Integer, nullable=False, default=160)
    stream_responses = Column(Boolean, nullable=False, default=True)
