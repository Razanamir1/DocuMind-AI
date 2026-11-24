"""Environment-based application configuration for DocuMind AI FastAPI backend."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


def _ensure_writable_dir(preferred: Path, fallback_sub: str) -> Path:
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "documind-data" / fallback_sub
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# Load root .env first, then backend/.env (backend/.env overrides root if present)
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(BACKEND_ROOT / ".env", override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    app_name: str = "DocuMind AI API"
    app_version: str = "2.0.0"

    # AI Provider settings (server-side only; never exposed to browser)
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-4-hydrogen", alias="GEMINI_MODEL")
    gemini_fallback_model: str = Field(
        default="gemini-3.1-flash-lite", alias="GEMINI_FALLBACK_MODEL"
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-2", alias="GEMINI_EMBEDDING_MODEL"
    )
    provider_timeout_seconds: float = Field(default=60.0, alias="PROVIDER_TIMEOUT_SECONDS")

    # Storage paths
    database_url: str = Field(
        default=f"sqlite:///{(BACKEND_ROOT / 'data' / 'documind.sqlite').as_posix()}",
        alias="DATABASE_URL",
    )
    chroma_persist_dir: str = Field(
        default=str(BACKEND_ROOT / "data" / "chroma_db"),
        alias="CHROMA_PERSIST_DIR",
    )
    upload_dir: str = Field(
        default=str(BACKEND_ROOT / "data" / "uploads"),
        alias="UPLOAD_DIR",
    )
    max_file_size_mb: int = Field(default=25, alias="MAX_FILE_SIZE_MB")

    # CORS origins
    allowed_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
        alias="ALLOWED_ORIGINS",
    )
    app_url: str = Field(default="", alias="APP_URL")

    @property
    def resolved_database_url(self) -> str:
        url = self.database_url
        if url.startswith("sqlite:///") and ":memory:" not in url:
            raw_path = Path(url.replace("sqlite:///", "", 1))
            if not raw_path.is_absolute():
                raw_path = (BACKEND_ROOT / raw_path).resolve()
            writable_parent = _ensure_writable_dir(raw_path.parent, "sqlite")
            return f"sqlite:///{(writable_parent / raw_path.name).as_posix()}"
        return url

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        if not p.is_absolute():
            p = (BACKEND_ROOT / p).resolve()
        return _ensure_writable_dir(p, "uploads")

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        if not p.is_absolute():
            p = (BACKEND_ROOT / p).resolve()
        return _ensure_writable_dir(p, "chroma_db")

    @property
    def allowed_origins(self) -> list[str]:
        origins = [
            o.strip().rstrip("/")
            for o in self.allowed_origins_raw.split(",")
            if o.strip()
        ]
        if self.app_url and self.app_url != "MY_APP_URL":
            origins.append(self.app_url.strip().rstrip("/"))
        return list(dict.fromkeys(origins))

    @property
    def effective_gemini_api_key(self) -> str:
        # Always check live os.environ as well so runtime secret injection works seamlessly
        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        key = env_key or self.gemini_api_key.strip()
        if not key or key in {"MY_GEMINI_API_KEY", "your_gemini_api_key_here"}:
            return ""
        return key

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
