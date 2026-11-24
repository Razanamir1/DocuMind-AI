"""Comprehensive automated pytest suite for the DocuMind AI Python/FastAPI/PyMuPDF/ChromaDB backend."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.api.routes import get_provider_dep, get_vector_store_dep
from app.core.config import get_settings
from app.core.errors import DocumentError, LLMProviderError
from app.db.session import Base, create_sqlite_engine, get_db
from app.main import create_app
from app.models.entities import WorkspaceSettingsORM
from app.providers.llm import HistoryTurn, LLMProvider, classify_provider_exception
from app.rag.chunker import chunk_pages
from app.rag.memory import (
    build_contextual_retrieval_query,
    generate_conversation_title,
    select_conversation_window,
)
from app.rag.pdf_processor import (
    PageText,
    create_sample_pdf_bytes,
    extract_pdf_pages,
    sanitize_filename,
)
from app.rag.vector_store import ChromaVectorStore


class DeterministicMockLLMProvider(LLMProvider):
    """Deterministic local provider for pytest: real ChromaDB cosine math with zero API quota usage."""

    provider_name = "MockGeminiTestProvider"
    chat_model = "mock-gemini-chat"
    embedding_model = "mock-gemini-embedding"

    def __init__(
        self,
        *,
        reply: str = "Based on the report, Q3 revenue was $148.4 million [financial_report.pdf, Page 1].",
        fail_generate_exc: Exception | None = None,
        fail_embed_exc: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.fail_generate_exc = fail_generate_exc
        self.fail_embed_exc = fail_embed_exc
        self.last_history: list[HistoryTurn] = []

    def is_configured(self) -> bool:
        return True

    def generate_text(
        self,
        *,
        user_prompt: str,
        system_instruction: str | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> str:
        self.last_history = list(history or [])
        if self.fail_generate_exc:
            raise self.fail_generate_exc
        return self.reply

    def stream_text(
        self,
        *,
        user_prompt: str,
        system_instruction: str | None = None,
        history: list[HistoryTurn] | None = None,
    ) -> Iterator[str]:
        self.last_history = list(history or [])
        if self.fail_generate_exc:
            raise self.fail_generate_exc
        midpoint = max(1, len(self.reply) // 2)
        yield self.reply[:midpoint]
        yield self.reply[midpoint:]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail_embed_exc:
            raise self.fail_embed_exc
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if any(w in lower for w in ("revenue", "arr", "margin", "financial", "q3")) else 0.0,
                    1.0 if any(w in lower for w in ("latency", "vector", "hnsw", "millisecond")) else 0.0,
                    1.0 if any(w in lower for w in ("governance", "policy", "aes-256", "deletion", "sla")) else 0.0,
                    1.0 if any(w in lower for w in ("quantum", "sourdough", "mars", "brazil", "parental")) else 0.0,
                    0.05,
                ]
            )
        return vectors


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch):
    """Creates an isolated SQLite DB, persistent ChromaDB directory, and upload folder per test."""
    db_path = tmp_path / "test.sqlite"
    chroma_dir = tmp_path / "chroma"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    engine = create_sqlite_engine(f"sqlite:///{db_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
    with TestSession() as db:
        db.add(WorkspaceSettingsORM(id=1))
        db.commit()

    vector_store = ChromaVectorStore(chroma_dir, collection_name="test_documind_chunks")
    provider = DeterministicMockLLMProvider()

    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(chroma_dir))

    app = create_app()

    def override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_provider_dep] = lambda: provider
    app.dependency_overrides[get_vector_store_dep] = lambda: vector_store

    # Avoid triggering global lifespan seeder against production paths during unit tests
    client = TestClient(app)
    yield {
        "client": client,
        "provider": provider,
        "vector_store": vector_store,
        "upload_dir": upload_dir,
        "session_factory": TestSession,
    }
    engine.dispose()


def _make_blank_image_only_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


# ==================== 1. PDF Extraction & Security Tests ====================


def test_pymupdf_extracts_multipage_text_preserving_page_numbers():
    pdf_bytes = create_sample_pdf_bytes(
        "PyMuPDF Unit Test",
        [
            {
                "heading": "Page One Revenue",
                "lines": ["Q3 revenue reached $148.4 million with 79.4% gross margin."],
            },
            {
                "heading": "Page Two Governance",
                "lines": ["All files are encrypted with AES-256 and purged within 5 seconds on deletion."],
            },
        ],
    )
    extracted = extract_pdf_pages(pdf_bytes)
    assert extracted.page_count == 2
    assert extracted.pages[0].page_number == 1
    assert "$148.4 million" in extracted.pages[0].text
    assert extracted.pages[1].page_number == 2
    assert "AES-256" in extracted.pages[1].text


def test_pymupdf_rejects_invalid_corrupted_and_scanned_pdfs():
    with pytest.raises(DocumentError) as invalid_err:
        extract_pdf_pages(b"plain text disguised as pdf")
    assert invalid_err.value.code == "INVALID_PDF"

    with pytest.raises(DocumentError) as corrupt_err:
        extract_pdf_pages(b"%PDF-1.4\n%%%corrupted_payload_bytes%%%")
    assert corrupt_err.value.code == "CORRUPTED_PDF"

    with pytest.raises(DocumentError) as scanned_err:
        extract_pdf_pages(_make_blank_image_only_pdf())
    assert scanned_err.value.code == "SCANNED_PDF"


def test_filename_sanitizer_prevents_path_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd.pdf"
    assert sanitize_filename("Q3 Report: Final (2026).pdf") == "Q3_Report_Final_(2026).pdf"


# ==================== 2. Chunking & Conversation Memory Tests ====================


def test_chunking_creates_overlapping_chunks_with_page_metadata():
    long_page = " ".join(
        f"Sentence {i} verifies overlapping PDF text splitting with exact page numbers."
        for i in range(1, 24)
    )
    chunks = chunk_pages(
        [
            PageText(page_number=1, text=long_page),
            PageText(page_number=2, text="Page two reports 18 ms median HNSW vector retrieval latency."),
        ],
        chunk_size=360,
        chunk_overlap=80,
    )
    assert len(chunks) >= 4
    assert chunks[0].page_number == 1
    assert chunks[1].char_start < chunks[0].char_end
    assert chunks[-1].page_number == 2


def test_conversation_memory_windowing_and_followup_resolution():
    turns = [
        ("user" if i % 2 == 0 else "assistant", f"Turn {i}: " + ("x" * 500))
        for i in range(16)
    ]
    window = select_conversation_window(turns, max_tokens=600, max_messages=10)
    assert len(window) <= 5
    assert "Turn 15" in window[-1].text

    followup = build_contextual_retrieval_query(
        "What about Q4?", ["What was Meridian Q3 revenue?"]
    )
    assert "Meridian Q3 revenue" in followup

    standalone = build_contextual_retrieval_query(
        "Explain quantum physics wave-particle duality in full detail.",
        ["What was Meridian Q3 revenue?"],
    )
    assert "Meridian Q3 revenue" not in standalone
    assert generate_conversation_title("  Analyze Q3 ARR growth  ") == "Analyze Q3 ARR growth"


# ==================== 3. Real ChromaDB RAG, Citations, Hallucination & Deletion ====================


def test_upload_chromadb_indexing_grounded_citations_and_hallucination_guard(isolated_env):
    client: TestClient = isolated_env["client"]
    vector_store: ChromaVectorStore = isolated_env["vector_store"]

    pdf_bytes = create_sample_pdf_bytes(
        "Financial Report",
        [
            {
                "heading": "Q3 Revenue Summary",
                "lines": [
                    "Q3 2026 total revenue was $148.4 million with 79.4% gross margin and $612 million ARR."
                ],
            },
            {
                "heading": "Infrastructure Latency",
                "lines": [
                    "Median vector retrieval latency dropped to 18 milliseconds via HNSW quantization."
                ],
            },
        ],
    )

    # 1. Upload & verify ChromaDB actually stores vectors
    up_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("financial_report.pdf", pdf_bytes, "application/pdf")},
    )
    assert up_res.status_code == 201
    doc = up_res.json()["document"]
    assert doc["status"] == "ready"
    assert doc["embeddingStatus"] == "vector"
    assert doc["pageCount"] == 2
    assert vector_store.count() == doc["chunkCount"]
    assert vector_store.has_document_vectors(doc["id"]) is True

    # 2. Duplicate upload rejected with 409
    dup_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("financial_report_copy.pdf", pdf_bytes, "application/pdf")},
    )
    assert dup_res.status_code == 409
    assert dup_res.json()["error"]["code"] == "DUPLICATE_DOCUMENT"

    # 3. Grounded question retrieves Page 1 from ChromaDB and returns structured citation
    chat_res = client.post(
        "/api/v1/chat",
        json={
            "message": "What was Q3 2026 total revenue and ARR?",
            "mode": "documents",
            "stream": False,
        },
    )
    assert chat_res.status_code == 200
    body = chat_res.json()
    convo_id = body["conversation"]["id"]
    assistant = body["assistantMessage"]
    assert assistant["insufficientEvidence"] is False
    assert len(assistant["citations"]) >= 1
    assert assistant["citations"][0]["documentId"] == doc["id"]
    assert assistant["citations"][0]["filename"] == "financial_report.pdf"
    assert assistant["citations"][0]["pageNumber"] == 1

    # 4. Page endpoint & PDF binary endpoint return exact page text and PDF bytes
    page_res = client.get(f"/api/v1/documents/{doc['id']}/pages/2")
    assert page_res.status_code == 200
    assert "18 milliseconds" in page_res.json()["page"]["content"]

    file_res = client.get(f"/api/v1/documents/{doc['id']}/file")
    assert file_res.status_code == 200
    assert file_res.headers["content-type"] == "application/pdf"
    assert file_res.content.startswith(b"%PDF-")

    # 5. Unsupported out-of-scope question refuses honestly with zero fabricated citations
    unsup_res = client.post(
        "/api/v1/chat",
        json={
            "conversationId": convo_id,
            "message": "What is the Brazil parental leave policy for astronauts on Mars?",
            "mode": "documents",
            "stream": False,
        },
    )
    assert unsup_res.status_code == 200
    unsup_msg = unsup_res.json()["assistantMessage"]
    assert unsup_msg["insufficientEvidence"] is True
    assert unsup_msg["citations"] == []

    # 6. Deleting document purges SQLite row, disk binary, AND ChromaDB vectors
    del_res = client.delete(f"/api/v1/documents/{doc['id']}")
    assert del_res.status_code == 200
    assert vector_store.count() == 0
    assert vector_store.has_document_vectors(doc["id"]) is False
    assert client.get(f"/api/v1/documents/{doc['id']}").status_code == 404


# ==================== 4. SSE Streaming, Conversation Persistence & Settings ====================


def test_sse_streaming_conversation_persistence_rename_clear_delete(isolated_env):
    client: TestClient = isolated_env["client"]

    with client.stream(
        "POST",
        "/api/v1/chat",
        json={"message": "Explain RAG chunk overlap briefly.", "mode": "general", "stream": True},
    ) as stream_res:
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]
        raw = "".join(stream_res.iter_text())

    assert "event: meta" in raw
    assert "event: delta" in raw
    assert "event: done" in raw

    # Extract conversation ID from persisted list
    list_res = client.get("/api/v1/conversations")
    assert list_res.status_code == 200
    convos = list_res.json()["conversations"]
    assert len(convos) == 1
    convo_id = convos[0]["id"]

    # Verify history persisted across requests
    detail_res = client.get(f"/api/v1/conversations/{convo_id}")
    assert detail_res.status_code == 200
    assert len(detail_res.json()["messages"]) == 2

    # Rename conversation
    patch_res = client.patch(
        f"/api/v1/conversations/{convo_id}", json={"title": "RAG Chunking Notes"}
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["conversation"]["title"] == "RAG Chunking Notes"

    # Clear & delete conversation
    assert client.post(f"/api/v1/conversations/{convo_id}/clear").status_code == 200
    assert len(client.get(f"/api/v1/conversations/{convo_id}").json()["messages"]) == 0
    assert client.delete(f"/api/v1/conversations/{convo_id}").status_code == 200
    assert client.get(f"/api/v1/conversations/{convo_id}").status_code == 404

    # Settings persistence
    s_patch = client.patch("/api/v1/settings", json={"topK": 7, "minRelevanceScore": 0.4})
    assert s_patch.status_code == 200
    assert s_patch.json()["settings"]["topK"] == 7
    assert client.get("/api/v1/settings").json()["settings"]["topK"] == 7


# ==================== 5. Validation, Failed Uploads & Provider Errors ====================


def test_api_validation_and_failed_uploads(isolated_env):
    client: TestClient = isolated_env["client"]

    # Blank prompt rejected by Pydantic
    blank_res = client.post(
        "/api/v1/chat", json={"message": "   ", "mode": "general", "stream": False}
    )
    assert blank_res.status_code == 422
    assert blank_res.json()["error"]["code"] == "VALIDATION_ERROR"

    # Non-PDF rejected
    txt_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert txt_res.status_code == 415
    assert txt_res.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"

    # Scanned image-only PDF rejected clearly without claiming OCR
    scan_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("scan.pdf", _make_blank_image_only_pdf(), "application/pdf")},
    )
    assert scan_res.status_code == 422
    assert scan_res.json()["error"]["code"] == "SCANNED_PDF"

    # Invalid conversation ID returns 404
    missing_convo = client.post(
        "/api/v1/chat",
        json={
            "conversationId": "conv-does-not-exist",
            "message": "Hello",
            "mode": "general",
            "stream": False,
        },
    )
    assert missing_convo.status_code == 404


def test_gemini_provider_errors_are_surfaced_not_fabricated(isolated_env):
    client: TestClient = isolated_env["client"]
    provider: DeterministicMockLLMProvider = isolated_env["provider"]

    # 1. Missing API key / quota error on generation returns structured HTTP error
    provider.fail_generate_exc = LLMProviderError(
        code="RATE_LIMITED",
        message="Quota exceeded.",
        status_code=429,
    )
    res = client.post(
        "/api/v1/chat",
        json={"message": "Hello world", "mode": "general", "stream": False},
    )
    assert res.status_code == 429
    assert res.json()["error"]["code"] == "RATE_LIMITED"

    # 2. Embedding failure on PDF upload fails cleanly without storing fake vectors
    provider.fail_embed_exc = LLMProviderError(
        code="MISSING_API_KEY",
        message="GEMINI_API_KEY missing.",
        status_code=503,
    )
    pdf_bytes = create_sample_pdf_bytes(
        "Upload Test", [{"heading": "Test", "lines": ["Real text inside PDF document."]}]
    )
    up_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("fail_embed.pdf", pdf_bytes, "application/pdf")},
    )
    assert up_res.status_code == 503
    assert up_res.json()["error"]["code"] == "MISSING_API_KEY"
    assert isolated_env["vector_store"].count() == 0

    # 3. Verify provider exception classification helper covers all key error classes
    assert classify_provider_exception(RuntimeError("401 API_KEY_INVALID")).code == "INVALID_CREDENTIALS"
    assert classify_provider_exception(RuntimeError("429 RESOURCE_EXHAUSTED")).code == "RATE_LIMITED"
    assert classify_provider_exception(RuntimeError("404 NOT_FOUND"), "bad-model").code == "MODEL_UNAVAILABLE"
    assert classify_provider_exception(TimeoutError("Request timed out")).code == "TIMEOUT"
    assert classify_provider_exception(ConnectionError("DNS connecterror")).code == "NETWORK_ERROR"


def test_chromadb_enforces_consistent_embedding_dimensionality(tmp_path):
    store = ChromaVectorStore(tmp_path / "chroma_dim_test")
    chunk = chunk_pages([PageText(page_number=1, text="Vector dimensionality check.")])[0]

    store.upsert_document_chunks(
        document_id="doc-dim-1",
        filename="dim_test.pdf",
        chunks=[chunk],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
    )
    assert store.get_embedding_dimension() == 4

    with pytest.raises(LLMProviderError) as mismatch_upsert:
        store.upsert_document_chunks(
            document_id="doc-dim-2",
            filename="dim_mismatch.pdf",
            chunks=[chunk],
            embeddings=[[0.1, 0.2]],
        )
    assert mismatch_upsert.value.code == "EMBEDDING_DIMENSION_MISMATCH"

    with pytest.raises(LLMProviderError) as mismatch_query:
        store.query(query_embedding=[0.1, 0.2, 0.3])
    assert mismatch_query.value.code == "EMBEDDING_DIMENSION_MISMATCH"

    store.delete_document("doc-dim-1")
    assert store.count() == 0
    assert store.get_embedding_dimension() is None


def test_gemini_provider_falls_back_on_rate_limit(monkeypatch):
    from types import SimpleNamespace
    from app.providers.llm import GeminiProvider

    provider = GeminiProvider()
    provider.chat_model = "gemini-4-hydrogen"
    provider.fallback_chat_model = "gemini-3.1-flash-lite"

    calls: list[str] = []

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append(f"gen:{model}")
            if model == "gemini-4-hydrogen":
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return SimpleNamespace(text="Fallback answer ok")

        def generate_content_stream(self, *, model, contents, config):
            calls.append(f"stream:{model}")
            if model == "gemini-4-hydrogen":
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return iter([SimpleNamespace(text="Stream "), SimpleNamespace(text="ok")])

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(provider, "_create_client", lambda timeout_seconds=None: fake_client)
    monkeypatch.setattr("app.providers.llm.time.sleep", lambda _s: None)

    assert provider.generate_text(user_prompt="Hello") == "Fallback answer ok"
    assert "gen:gemini-4-hydrogen" in calls
    assert "gen:gemini-3.1-flash-lite" in calls

    # Cooldown active: _candidate_chat_models() excludes cooling primary so fallback runs as final active model
    assert provider._candidate_chat_models() == ["gemini-3.1-flash-lite"]
    calls.clear()
    assert "".join(provider.stream_text(user_prompt="Hello")) == "Stream ok"
    assert calls == ["stream:gemini-3.1-flash-lite"]

    # After cooldown expires/clears, stream tries primary first then fails over cleanly
    provider._model_cooldown_until.clear()
    calls.clear()
    assert "".join(provider.stream_text(user_prompt="Hello")) == "Stream ok"
    assert calls == ["stream:gemini-4-hydrogen", "stream:gemini-3.1-flash-lite"]

    # Also verify failover on 503 UNAVAILABLE / 500 INTERNAL / TIMEOUT
    for transient_msg in ("503 UNAVAILABLE high demand", "500 INTERNAL error", "504 deadline exceeded"):
        provider._model_cooldown_until.clear()

        class TransientFailModels:
            def generate_content(self, *, model, contents, config):
                if model == "gemini-4-hydrogen":
                    raise RuntimeError(transient_msg)
                return SimpleNamespace(text="Recovered ok")

            def generate_content_stream(self, *, model, contents, config):
                if model == "gemini-4-hydrogen":
                    raise RuntimeError(transient_msg)
                return iter([SimpleNamespace(text="Recovered stream")])

        transient_client = SimpleNamespace(models=TransientFailModels())
        monkeypatch.setattr(provider, "_create_client", lambda timeout_seconds=None, c=transient_client: c)
        assert provider.generate_text(user_prompt="Hi") == "Recovered ok"
        assert "".join(provider.stream_text(user_prompt="Hi")) == "Recovered stream"

    # Verify stream_text switches directly to unary generate_content after 1 stream ReadTimeout (no redundant stream retries)
    provider._model_cooldown_until = {"gemini-4-hydrogen": 1e18}
    timeout_calls: list[str] = []

    class StreamTimeoutUnaryOkModels:
        def generate_content_stream(self, *, model, contents, config):
            timeout_calls.append(f"stream:{model}")
            raise TimeoutError("The read operation timed out")

        def generate_content(self, *, model, contents, config):
            timeout_calls.append(f"unary:{model}")
            return SimpleNamespace(text="Unary recovery after stream timeout")

    timeout_client = SimpleNamespace(models=StreamTimeoutUnaryOkModels())
    monkeypatch.setattr(provider, "_create_client", lambda timeout_seconds=None: timeout_client)
    assert "".join(provider.stream_text(user_prompt="Hi")) == "Unary recovery after stream timeout"
    assert timeout_calls == ["stream:gemini-3.1-flash-lite", "unary:gemini-3.1-flash-lite"]

    # Verify fallback 404 MODEL_UNAVAILABLE does not mask an earlier 429 RATE_LIMITED error
    provider._model_cooldown_until.clear()

    class RateLimitThen404Models:
        def generate_content(self, *, model, contents, config):
            if model == "gemini-4-hydrogen":
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            raise RuntimeError("404 NOT_FOUND")

    masked_client = SimpleNamespace(models=RateLimitThen404Models())
    monkeypatch.setattr(provider, "_create_client", lambda timeout_seconds=None: masked_client)
    with pytest.raises(LLMProviderError) as err_info:
        provider.generate_text(user_prompt="Hi")
    assert err_info.value.code == "RATE_LIMITED"


def test_idempotent_reindex_handles_errors_retries_safely_and_prevents_duplicates(isolated_env):
    from app.db.seed import seed_initial_workspace
    from app.services.document_service import reset_indexing_coordinator_state

    reset_indexing_coordinator_state()
    client: TestClient = isolated_env["client"]
    provider: DeterministicMockLLMProvider = isolated_env["provider"]
    vector_store: ChromaVectorStore = isolated_env["vector_store"]
    SessionFactory = isolated_env["session_factory"]
    upload_dir: Path = isolated_env["upload_dir"]

    # Seed workspace with deferred embeddings (simulating non-blocking Cloud Run cold start)
    with SessionFactory() as db:
        seed_initial_workspace(
            db,
            provider=provider,
            vector_store=vector_store,
            upload_dir=upload_dir,
            defer_embeddings=True,
        )

    # 1. Health reports pending state accurately before indexing
    health_pending = client.get("/api/v1/health").json()["stats"]
    assert health_pending["documentsReady"] == 2
    assert health_pending["documentsIndexed"] == 0
    assert health_pending["documentsPending"] == 2
    assert health_pending["indexingState"] == "pending"
    assert health_pending["storageMode"] == "ephemeral-local"
    assert health_pending["chromaVectorCount"] == 0

    # 2. Simulate transient Gemini embedding error during reindex: surfaces error & preserves pending work
    provider.fail_embed_exc = LLMProviderError("RATE_LIMITED", "Rate limit exceeded", 429)
    fail_resp = client.post("/api/v1/documents/reindex")
    assert fail_resp.status_code == 429
    assert fail_resp.json()["error"]["code"] == "RATE_LIMITED"

    health_degraded = client.get("/api/v1/health").json()["stats"]
    assert health_degraded["documentsPending"] == 2
    assert health_degraded["indexingState"] == "degraded"
    assert health_degraded["lastIndexingError"] == "RATE_LIMITED"

    # 3. Provider recovers -> retry succeeds cleanly and indexes all chunks
    provider.fail_embed_exc = None
    ok_resp = client.post("/api/v1/documents/reindex")
    assert ok_resp.status_code == 200
    first_count = ok_resp.json()["chromaVectorCount"]
    assert first_count > 0
    assert ok_resp.json()["documentsIndexed"] == 2
    assert ok_resp.json()["documentsPending"] == 0
    assert ok_resp.json()["indexingState"] == "ready"

    # 4. Retrying reindex again is completely idempotent (zero duplicate vectors)
    retry_resp = client.post("/api/v1/documents/reindex")
    assert retry_resp.status_code == 200
    assert retry_resp.json()["chromaVectorCount"] == first_count

    health_ready = client.get("/api/v1/health").json()["stats"]
    assert health_ready["indexingState"] == "ready"
    assert health_ready["lastIndexingError"] is None
    assert health_ready["documentsIndexed"] == 2
    assert health_ready["documentsPending"] == 0


def test_balanced_retriever_handles_summaries_paraphrases_cross_lingual_and_blocks_unrelated():
    from app.rag.retriever import (
        compute_lexical_score,
        has_script_mismatch,
        is_overview_query,
        rank_and_filter_matches,
        tokenize_for_search,
    )
    from app.rag.vector_store import VectorMatch

    # 1. Overview / summary intent detection (English + Arabic)
    assert is_overview_query("Summarize this PDF") is True
    assert is_overview_query("What is this document about?") is True
    assert is_overview_query("Give me the key takeaways") is True
    assert is_overview_query("لخص هذا الملف") is True
    assert is_overview_query("ما هو محتوى المستند؟") is True
    assert is_overview_query("What was Q3 2026 revenue?") is False

    # 2. Unicode tokenization preserves Arabic and accented terms
    arabic_tokens = tokenize_for_search("ما هي مهارات الذكاء الاصطناعي؟")
    assert "مهارات" in arabic_tokens
    assert "الذكاء" in arabic_tokens
    assert has_script_mismatch("ما هي مهارات شهد؟", "Python, SQL, and Cloud Architecture experience.") is True
    assert has_script_mismatch("What are the skills?", "Python, SQL, and Cloud Architecture experience.") is False

    # 3. Paraphrased / synonym query (zero lexical overlap, moderate 0.44 cosine score) passes balanced gate
    paraphrase_match = VectorMatch(
        chunk_id="doc-resume-chunk-0",
        document_id="doc-resume",
        filename="candidate_profile.pdf",
        page_number=1,
        chunk_index=0,
        content="AI, Data Science & Cloud Computing Trainee at DEBI. Developed ETL pipelines and cloud deployments.",
        vector_score=0.44,
    )
    paraphrase_ranked = rank_and_filter_matches(
        query="What is their work history?",
        matches=[paraphrase_match],
        min_relevance_score=0.34,
    )
    assert len(paraphrase_ranked) == 1
    assert paraphrase_ranked[0].combined_score == 0.44

    # 4. Cross-lingual query (Arabic question vs English PDF, 0.35 cosine score) passes via script mismatch
    cross_match = VectorMatch(
        chunk_id="doc-resume-chunk-1",
        document_id="doc-resume",
        filename="candidate_profile.pdf",
        page_number=1,
        chunk_index=1,
        content="Technical Skills: Python, FastAPI, SQL, PyTorch, and Google Cloud Platform.",
        vector_score=0.35,
    )
    cross_ranked = rank_and_filter_matches(
        query="ما هي المهارات التقنية؟",
        matches=[cross_match],
        min_relevance_score=0.34,
    )
    assert len(cross_ranked) == 1

    # 5. Filename-aware lexical boost helps queries referencing an uploaded PDF's name
    filename_lex = compute_lexical_score(
        "Tell me about Shahd qualifications",
        "B.Sc. in Computer Science with machine learning internships.",
        filename="Shahd_Mohamed_Resume.pdf",
    )
    assert filename_lex > 0.0

    # 6. Truly unrelated out-of-scope query (low cosine score 0.25, zero lexical match) is still rejected
    unrelated_match = VectorMatch(
        chunk_id="doc-fin-chunk-0",
        document_id="doc-fin",
        filename="financial_report.pdf",
        page_number=1,
        chunk_index=0,
        content="Q3 2026 total revenue was $148.4 million with 79.4% gross margin.",
        vector_score=0.25,
    )
    unrelated_ranked = rank_and_filter_matches(
        query="What is the parental leave policy in Brazil?",
        matches=[unrelated_match],
        min_relevance_score=0.34,
    )
    assert unrelated_ranked == []


def test_overview_summary_and_streaming_insufficient_marker_end_to_end(isolated_env):
    client: TestClient = isolated_env["client"]
    provider: DeterministicMockLLMProvider = isolated_env["provider"]

    pdf_bytes = create_sample_pdf_bytes(
        "Resume Overview Test",
        [
            {
                "heading": "Profile Summary",
                "lines": ["AI and Cloud Engineering candidate experienced with Python, FastAPI, and ChromaDB."],
            },
            {
                "heading": "Education & Projects",
                "lines": ["Built an enterprise PDF RAG workspace with citation grounding and SSE streaming."],
            },
        ],
    )
    up_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("candidate_resume.pdf", pdf_bytes, "application/pdf")},
    )
    assert up_res.status_code == 201
    doc = up_res.json()["document"]

    # 1. Broad summary question retrieves representative pages from the newly uploaded PDF (no false refusal)
    provider.reply = "This resume highlights Python, FastAPI, and ChromaDB experience [candidate_resume.pdf, Page 1]."
    summary_res = client.post(
        "/api/v1/chat",
        json={
            "message": "Summarize this PDF",
            "mode": "documents",
            "documentIds": [doc["id"]],
            "stream": False,
        },
    )
    assert summary_res.status_code == 200
    summary_msg = summary_res.json()["assistantMessage"]
    assert summary_msg["insufficientEvidence"] is False
    assert len(summary_msg["citations"]) >= 1
    assert summary_msg["citations"][0]["filename"] == "candidate_resume.pdf"
    assert summary_msg["citations"][0]["pageNumber"] == 1

    # 2. Arabic summary question also samples representative pages cleanly
    provider.reply = "يتناول هذا الملف خبرة في Python وFastAPI [candidate_resume.pdf, Page 1]."
    ar_res = client.post(
        "/api/v1/chat",
        json={
            "message": "لخص هذا الملف",
            "mode": "documents",
            "documentIds": [doc["id"]],
            "stream": False,
        },
    )
    assert ar_res.status_code == 200
    assert ar_res.json()["assistantMessage"]["insufficientEvidence"] is False
    assert len(ar_res.json()["assistantMessage"]["citations"]) >= 1

    # 3. Streaming [[INSUFFICIENT_EVIDENCE]] marker is stripped from user text and clears citations
    provider.reply = "[[INSUFFICIENT_EVIDENCE]] لا تحتوي المستندات المرفوعة على معلومات كافية للإجابة."
    with client.stream(
        "POST",
        "/api/v1/chat",
        json={
            "message": "Summarize this PDF",
            "mode": "documents",
            "documentIds": [doc["id"]],
            "stream": True,
        },
    ) as stream_res:
        assert stream_res.status_code == 200
        raw_sse = "".join(stream_res.iter_text())

    assert "[[INSUFFICIENT_EVIDENCE]]" not in raw_sse
    done_blocks = [
        blk for blk in raw_sse.strip().split("\n\n") if blk.startswith("event: done")
    ]
    assert len(done_blocks) == 1
    done_data_line = next(
        line[5:].strip() for line in done_blocks[0].split("\n") if line.startswith("data:")
    )
    done_payload = json.loads(done_data_line)
    done_msg = done_payload["assistantMessage"]
    assert done_msg["content"] == "لا تحتوي المستندات المرفوعة على معلومات كافية للإجابة."
    assert done_msg["insufficientEvidence"] is True
    assert done_msg["citations"] == []


