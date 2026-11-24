"""SQLAlchemy SQLite engine and session management with WAL mode and FK enforcement."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import get_settings

Base = declarative_base()


def _configure_sqlite_pragmas(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.close()


def create_sqlite_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
        db_file = Path(database_url.replace("sqlite:///", "", 1))
        db_file.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
        future=True,
    )
    event.listen(engine, "connect", _configure_sqlite_pragmas)
    return engine


_settings = get_settings()
engine = create_sqlite_engine(_settings.resolved_database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def init_db(target_engine: Engine | None = None) -> None:
    from app.models import entities  # noqa: F401

    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)

    # Ensure singleton workspace_settings row exists
    local_session = sessionmaker(autocommit=False, autoflush=False, bind=eng, future=True)
    with local_session() as db:
        from app.models.entities import WorkspaceSettingsORM

        existing = db.get(WorkspaceSettingsORM, 1)
        if existing is None:
            db.add(WorkspaceSettingsORM(id=1))
            db.commit()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
