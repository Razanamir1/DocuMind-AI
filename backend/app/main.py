"""DocuMind AI — Primary Python FastAPI Application Entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_v1_router
from app.core.config import PROJECT_ROOT, get_settings
from app.core.errors import AppError
from app.db.seed import seed_initial_workspace
from app.db.session import SessionLocal, engine, init_db
from app.providers.llm import get_llm_provider
from app.rag.vector_store import get_vector_store

logger = logging.getLogger("documind.main")


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    upload_path = settings.upload_path
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    init_db()

    provider = get_llm_provider()
    vector_store = get_vector_store()
    application.state.provider = provider
    application.state.vector_store = vector_store

    # Seed sample PDFs & SQLite rows locally without external network calls so HTTP
    # startup binds PORT immediately. Pending seeded vectors are indexed idempotently
    # on demand during the first RAG request or via POST /api/v1/documents/reindex
    # (never relying on an in-process daemon thread that Cloud Run may CPU-throttle).
    with SessionLocal() as db:
        seed_initial_workspace(
            db,
            provider=provider,
            vector_store=vector_store,
            upload_dir=upload_path,
            defer_embeddings=True,
        )
    try:
        yield
    finally:
        engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "DocuMind AI — Enterprise AI Workspace & Document Intelligence API powered by "
            "FastAPI, SQLAlchemy/SQLite, PyMuPDF, persistent ChromaDB, and Google Gemini."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_origin_regex=r"^https://.*\.run\.app$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @application.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError):
        first_msg = "Invalid request payload."
        errors = exc.errors()
        if errors:
            first_msg = str(errors[0].get("msg") or first_msg)
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "VALIDATION_ERROR", "message": first_msg}},
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected server error occurred. Please try again.",
                }
            },
        )

    application.include_router(api_v1_router)

    # Production mode: serve compiled React dist/ SPA directly from FastAPI when present
    dist_dir = PROJECT_ROOT / "dist"
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            application.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        src_assets_dir = dist_dir / "src" / "assets"
        if src_assets_dir.exists():
            application.mount(
                "/src/assets",
                StaticFiles(directory=str(src_assets_dir)),
                name="src_assets",
            )

        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            if full_path.startswith("api/") or full_path in {"docs", "redoc", "openapi.json"}:
                return JSONResponse(
                    status_code=404,
                    content={"error": {"code": "NOT_FOUND", "message": "Endpoint not found."}},
                )
            candidate = (dist_dir / full_path).resolve()
            try:
                candidate.relative_to(dist_dir.resolve())
                if candidate.is_file():
                    return FileResponse(candidate)
            except ValueError:
                pass
            return FileResponse(dist_dir / "index.html")

    return application


app = create_app()
