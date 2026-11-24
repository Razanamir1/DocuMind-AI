"""Chat orchestration, conversation persistence, RAG retrieval, and SSE streaming service."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.errors import ChatError, LLMProviderError
from app.models.entities import ConversationORM, MessageORM, utcnow
from app.providers.llm import HistoryTurn, LLMProvider
from app.rag.memory import (
    build_contextual_retrieval_query,
    generate_conversation_title,
    select_conversation_window,
)
from app.rag.retriever import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    INSUFFICIENT_MARKER,
    NO_DOCUMENTS_INDEXED_RESPONSE,
    RankedPassage,
    build_grounded_prompt,
    build_structured_citations,
    is_overview_query,
    rank_and_filter_matches,
)
from app.rag.vector_store import ChromaVectorStore
from app.schemas.api_schemas import (
    ChatMode,
    ChatRequest,
    CitationSchema,
    ConversationSchema,
    MessageSchema,
    WorkspaceSettingsPatchSchema,
    WorkspaceSettingsSchema,
)
from app.services.document_service import (
    ensure_documents_indexed_in_chroma,
    get_workspace_settings_row,
)

GENERAL_SYSTEM_INSTRUCTION = "\n".join(
    [
        "You are DocuMind AI, an intelligent, concise, and helpful enterprise AI workspace assistant.",
        "Provide accurate, well-structured answers using clean Markdown formatting (headings, bullet points, tables, and fenced code blocks with language identifiers when sharing code).",
        "Be transparent about uncertainty and avoid fabricating statistics, citations, or URLs.",
    ]
)

INSUFFICIENT_REGEX = re.compile(
    r"do not contain enough information|not contain sufficient information|insufficient evidence|"
    r"لا تحتوي(?: المستندات| الوثائق)? على معلومات كافية|لا توجد معلومات كافية",
    re.IGNORECASE,
)


def _strip_insufficient_marker(text: str) -> tuple[str, bool]:
    stripped = text.lstrip()
    if stripped.startswith(INSUFFICIENT_MARKER):
        return stripped[len(INSUFFICIENT_MARKER) :].lstrip(" \n:-"), True
    return text, False


def conversation_to_schema(
    convo: ConversationORM,
    *,
    message_count: int | None = None,
    last_preview: str | None = None,
) -> ConversationSchema:
    return ConversationSchema(
        id=convo.id,
        title=convo.title,
        mode=convo.mode,  # type: ignore[arg-type]
        createdAt=convo.created_at.isoformat(),
        updatedAt=convo.updated_at.isoformat(),
        messageCount=message_count,
        lastMessagePreview=last_preview,
    )


def message_to_schema(msg: MessageORM) -> MessageSchema:
    try:
        raw_cites = json.loads(msg.citations_json or "[]")
        citations = [CitationSchema(**item) for item in raw_cites]
    except Exception:
        citations = []

    return MessageSchema(
        id=msg.id,
        conversationId=msg.conversation_id,
        role=msg.role,  # type: ignore[arg-type]
        content=msg.content,
        mode=msg.mode,  # type: ignore[arg-type]
        citations=citations,
        insufficientEvidence=bool(msg.insufficient_evidence),
        createdAt=msg.created_at.isoformat(),
    )


def get_settings_schema(db: Session) -> WorkspaceSettingsSchema:
    row = get_workspace_settings_row(db)
    return WorkspaceSettingsSchema(
        topK=row.top_k,
        minRelevanceScore=row.min_relevance_score,
        chunkSize=row.chunk_size,
        chunkOverlap=row.chunk_overlap,
        streamResponses=bool(row.stream_responses),
    )


def update_settings(
    db: Session, patch: WorkspaceSettingsPatchSchema
) -> WorkspaceSettingsSchema:
    row = get_workspace_settings_row(db)
    if patch.topK is not None:
        row.top_k = patch.topK
    if patch.minRelevanceScore is not None:
        row.min_relevance_score = patch.minRelevanceScore
    if patch.chunkSize is not None:
        row.chunk_size = patch.chunkSize
    if patch.chunkOverlap is not None:
        row.chunk_overlap = patch.chunkOverlap
    if patch.streamResponses is not None:
        row.stream_responses = patch.streamResponses
    db.commit()
    db.refresh(row)
    return get_settings_schema(db)


def list_conversations(db: Session) -> list[ConversationSchema]:
    convos = db.query(ConversationORM).order_by(ConversationORM.updated_at.desc()).all()
    counts = dict(
        db.query(MessageORM.conversation_id, func.count(MessageORM.id))
        .group_by(MessageORM.conversation_id)
        .all()
    )
    results: list[ConversationSchema] = []
    for convo in convos:
        last_msg = (
            db.query(MessageORM.content)
            .filter(MessageORM.conversation_id == convo.id)
            .order_by(MessageORM.created_at.desc())
            .first()
        )
        results.append(
            conversation_to_schema(
                convo,
                message_count=int(counts.get(convo.id, 0)),
                last_preview=last_msg[0] if last_msg else None,
            )
        )
    return results


def create_conversation(
    db: Session, *, title: str | None = None, mode: ChatMode = "general"
) -> ConversationSchema:
    now = utcnow()
    convo = ConversationORM(
        title=(title or "").strip() or "New Conversation",
        mode=mode,
        created_at=now,
        updated_at=now,
    )
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return conversation_to_schema(convo, message_count=0)


def get_conversation_with_messages(
    db: Session, conversation_id: str
) -> tuple[ConversationSchema, list[MessageSchema]] | None:
    convo = db.get(ConversationORM, conversation_id)
    if not convo:
        return None
    msgs = (
        db.query(MessageORM)
        .filter(MessageORM.conversation_id == conversation_id)
        .order_by(MessageORM.created_at.asc())
        .all()
    )
    return (
        conversation_to_schema(convo, message_count=len(msgs)),
        [message_to_schema(m) for m in msgs],
    )


def update_conversation(
    db: Session,
    conversation_id: str,
    *,
    title: str | None = None,
    mode: ChatMode | None = None,
) -> ConversationSchema | None:
    convo = db.get(ConversationORM, conversation_id)
    if not convo:
        return None
    if title is not None and title.strip():
        convo.title = title.strip()
    if mode is not None:
        convo.mode = mode
    convo.updated_at = utcnow()
    db.commit()
    db.refresh(convo)
    return conversation_to_schema(convo)


def delete_conversation(db: Session, conversation_id: str) -> bool:
    convo = db.get(ConversationORM, conversation_id)
    if not convo:
        return False
    db.delete(convo)
    db.commit()
    return True


def clear_conversation(db: Session, conversation_id: str) -> bool:
    convo = db.get(ConversationORM, conversation_id)
    if not convo:
        return False
    db.query(MessageORM).filter(MessageORM.conversation_id == conversation_id).delete()
    convo.updated_at = utcnow()
    db.commit()
    return True


def persist_message(
    db: Session,
    *,
    conversation_id: str,
    role: str,
    content: str,
    mode: ChatMode,
    citations: list[CitationSchema],
    insufficient_evidence: bool,
) -> MessageSchema:
    now = utcnow()
    msg = MessageORM(
        conversation_id=conversation_id,
        role=role,
        content=content,
        mode=mode,
        citations_json=json.dumps([c.model_dump() for c in citations]),
        insufficient_evidence=insufficient_evidence,
        created_at=now,
    )
    db.add(msg)
    convo = db.get(ConversationORM, conversation_id)
    if convo:
        convo.updated_at = now
    db.commit()
    db.refresh(msg)
    return message_to_schema(msg)


@dataclass
class PreparedChatTurn:
    conversation: ConversationSchema
    user_message: MessageSchema
    mode: ChatMode
    citations: list[CitationSchema]
    insufficient_evidence: bool
    direct_answer: str | None
    system_instruction: str
    user_prompt: str
    history: list[HistoryTurn]


def prepare_chat_turn(
    db: Session,
    *,
    provider: LLMProvider,
    vector_store: ChromaVectorStore,
    req: ChatRequest,
) -> PreparedChatTurn:
    prompt = req.message.strip()
    if not prompt:
        raise ChatError("EMPTY_PROMPT", "Message cannot be empty.", 400)

    if req.conversationId:
        convo = db.get(ConversationORM, req.conversationId)
        if not convo:
            raise ChatError(
                "CONVERSATION_NOT_FOUND",
                "The requested conversation does not exist.",
                404,
            )
    else:
        now = utcnow()
        convo = ConversationORM(
            title=generate_conversation_title(prompt),
            mode=req.mode,
            created_at=now,
            updated_at=now,
        )
        db.add(convo)
        db.commit()
        db.refresh(convo)

    existing_msgs = (
        db.query(MessageORM)
        .filter(MessageORM.conversation_id == convo.id)
        .order_by(MessageORM.created_at.asc())
        .all()
    )

    if req.regenerate and existing_msgs:
        if existing_msgs[-1].role == "assistant":
            db.delete(existing_msgs[-1])
            db.commit()
            existing_msgs = existing_msgs[:-1]
        if existing_msgs and existing_msgs[-1].role == "user":
            user_message = message_to_schema(existing_msgs[-1])
            existing_msgs = existing_msgs[:-1]
        else:
            user_message = persist_message(
                db,
                conversation_id=convo.id,
                role="user",
                content=prompt,
                mode=req.mode,
                citations=[],
                insufficient_evidence=False,
            )
    else:
        if not existing_msgs and (not convo.title.strip() or convo.title == "New Conversation"):
            convo.title = generate_conversation_title(prompt)
        convo.mode = req.mode
        convo.updated_at = utcnow()
        db.commit()
        db.refresh(convo)

        user_message = persist_message(
            db,
            conversation_id=convo.id,
            role="user",
            content=prompt,
            mode=req.mode,
            citations=[],
            insufficient_evidence=False,
        )

    convo_schema = conversation_to_schema(convo)
    history = select_conversation_window(
        [(m.role, m.content) for m in existing_msgs],
        max_tokens=3200,
        max_messages=10,
    )

    if req.mode == "general":
        return PreparedChatTurn(
            conversation=convo_schema,
            user_message=user_message,
            mode="general",
            citations=[],
            insufficient_evidence=False,
            direct_answer=None,
            system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            user_prompt=prompt,
            history=history,
        )

    # ==================== Document-Grounded RAG via ChromaDB ====================
    ws_settings = get_workspace_settings_row(db)
    ready_doc_ids = ensure_documents_indexed_in_chroma(
        db,
        provider=provider,
        vector_store=vector_store,
        document_ids=req.documentIds,
    )

    if not ready_doc_ids:
        return PreparedChatTurn(
            conversation=convo_schema,
            user_message=user_message,
            mode="documents",
            citations=[],
            insufficient_evidence=True,
            direct_answer=NO_DOCUMENTS_INDEXED_RESPONSE,
            system_instruction="",
            user_prompt=prompt,
            history=history,
        )

    overview_intent = is_overview_query(prompt)
    prior_user_prompts = [m.content for m in existing_msgs if m.role == "user"]
    retrieval_query = (
        prompt if overview_intent else build_contextual_retrieval_query(prompt, prior_user_prompts)
    )

    if overview_intent:
        rep_matches = vector_store.get_representative_chunks(
            document_ids=ready_doc_ids,
            limit=ws_settings.top_k,
        )
        ranked = [
            RankedPassage(
                match=m,
                lexical_score=0.0,
                combined_score=m.vector_score,
            )
            for m in rep_matches
        ]
    else:
        # Real query embedding + higher-recall ChromaDB vector search
        query_embedding = provider.embed_texts([retrieval_query])[0]
        raw_matches = vector_store.query(
            query_embedding=query_embedding,
            n_results=max(ws_settings.top_k * 4, 20),
            document_ids=ready_doc_ids,
        )

        ranked = rank_and_filter_matches(
            query=retrieval_query,
            matches=raw_matches,
            top_k=ws_settings.top_k,
            min_relevance_score=ws_settings.min_relevance_score,
            single_document_scope=len(ready_doc_ids) == 1,
        )

    if not ranked:
        return PreparedChatTurn(
            conversation=convo_schema,
            user_message=user_message,
            mode="documents",
            citations=[],
            insufficient_evidence=True,
            direct_answer=INSUFFICIENT_EVIDENCE_RESPONSE,
            system_instruction="",
            user_prompt=prompt,
            history=history,
        )

    citations = build_structured_citations(ranked, retrieval_query)
    system_instruction, grounded_prompt = build_grounded_prompt(
        prompt, ranked, is_overview=overview_intent
    )

    return PreparedChatTurn(
        conversation=convo_schema,
        user_message=user_message,
        mode="documents",
        citations=citations,
        insufficient_evidence=False,
        direct_answer=None,
        system_instruction=system_instruction,
        user_prompt=grounded_prompt,
        history=history,
    )


def execute_chat_turn(
    db: Session,
    *,
    provider: LLMProvider,
    vector_store: ChromaVectorStore,
    req: ChatRequest,
) -> dict:
    prepared = prepare_chat_turn(
        db, provider=provider, vector_store=vector_store, req=req
    )

    if prepared.direct_answer is not None:
        answer_text = prepared.direct_answer
    else:
        answer_text = provider.generate_text(
            user_prompt=prepared.user_prompt,
            system_instruction=prepared.system_instruction,
            history=prepared.history,
        )

    answer_text, had_marker = _strip_insufficient_marker(answer_text)
    has_inline_citation = bool(
        re.search(r"\[[^\[\]]+,\s*Page\s+\d+\]", answer_text, re.IGNORECASE)
    )
    insufficient = (
        prepared.insufficient_evidence
        or had_marker
        or (bool(INSUFFICIENT_REGEX.search(answer_text)) and not has_inline_citation)
    )
    assistant_message = persist_message(
        db,
        conversation_id=prepared.conversation.id,
        role="assistant",
        content=answer_text.strip(),
        mode=prepared.mode,
        citations=[] if insufficient else prepared.citations,
        insufficient_evidence=insufficient,
    )

    return {
        "conversation": prepared.conversation.model_dump(),
        "userMessage": prepared.user_message.model_dump(),
        "assistantMessage": assistant_message.model_dump(),
    }


def _sse_event(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_chat_turn_sse(
    db: Session,
    *,
    provider: LLMProvider,
    vector_store: ChromaVectorStore,
    prepared: PreparedChatTurn,
) -> Iterator[str]:
    yield _sse_event(
        "meta",
        {
            "conversation": prepared.conversation.model_dump(),
            "userMessage": prepared.user_message.model_dump(),
            "mode": prepared.mode,
            "citations": []
            if prepared.insufficient_evidence
            else [c.model_dump() for c in prepared.citations],
        },
    )

    chunks_collected: list[str] = []
    had_marker = False

    if prepared.direct_answer is not None:
        chunks_collected.append(prepared.direct_answer)
        yield _sse_event("delta", {"text": prepared.direct_answer})
    else:
        # Hold leading characters briefly only if they could be the [[INSUFFICIENT_EVIDENCE]] marker
        prefix_buffer = ""
        prefix_resolved = False
        marker_len = len(INSUFFICIENT_MARKER)

        try:
            for piece in provider.stream_text(
                user_prompt=prepared.user_prompt,
                system_instruction=prepared.system_instruction,
                history=prepared.history,
            ):
                if not prefix_resolved:
                    prefix_buffer += piece
                    stripped_prefix = prefix_buffer.lstrip()
                    if not stripped_prefix:
                        continue
                    if stripped_prefix.startswith(INSUFFICIENT_MARKER):
                        had_marker = True
                        prefix_resolved = True
                        remainder = stripped_prefix[marker_len:].lstrip(" \n:-")
                        if remainder:
                            chunks_collected.append(remainder)
                            yield _sse_event("delta", {"text": remainder})
                    elif INSUFFICIENT_MARKER.startswith(stripped_prefix):
                        # Still potentially receiving the marker prefix; wait for next chunk
                        continue
                    else:
                        prefix_resolved = True
                        chunks_collected.append(prefix_buffer)
                        yield _sse_event("delta", {"text": prefix_buffer})
                else:
                    chunks_collected.append(piece)
                    yield _sse_event("delta", {"text": piece})

            if not prefix_resolved and prefix_buffer:
                cleaned_buf, marker_hit = _strip_insufficient_marker(prefix_buffer)
                had_marker = had_marker or marker_hit
                if cleaned_buf:
                    chunks_collected.append(cleaned_buf)
                    yield _sse_event("delta", {"text": cleaned_buf})
        except LLMProviderError as exc:
            yield _sse_event("error", {"code": exc.code, "message": exc.message})
            return
        except Exception:
            yield _sse_event(
                "error",
                {
                    "code": "STREAM_ERROR",
                    "message": "An unexpected error occurred while streaming the AI response.",
                },
            )
            return

    full_text = "".join(chunks_collected).strip()
    has_inline_citation = bool(
        re.search(r"\[[^\[\]]+,\s*Page\s+\d+\]", full_text, re.IGNORECASE)
    )
    insufficient = (
        prepared.insufficient_evidence
        or had_marker
        or (bool(INSUFFICIENT_REGEX.search(full_text)) and not has_inline_citation)
    )

    assistant_message = persist_message(
        db,
        conversation_id=prepared.conversation.id,
        role="assistant",
        content=full_text,
        mode=prepared.mode,
        citations=[] if insufficient else prepared.citations,
        insufficient_evidence=insufficient,
    )

    yield _sse_event(
        "done",
        {
            "conversation": prepared.conversation.model_dump(),
            "assistantMessage": assistant_message.model_dump(),
        },
    )
