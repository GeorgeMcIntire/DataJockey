import os
import time
from dataclasses import dataclass
from functools import lru_cache
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

load_dotenv()  # reads .env if present (won’t override existing env vars)

T = TypeVar("T")

# ---- Minimal, typed config (no pydantic) ------------------------------------
@dataclass(frozen=True)
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    pool_size: int = 5
    max_overflow: int = 5
    pool_recycle_seconds: int = 1800
    connect_timeout_seconds: int = 10
    statement_timeout_ms: int = 60_000    # server-side timeout
    application_name: str = "jaage_etl"
    sslmode: str = "prefer"               # use "require" in cloud if needed

    @staticmethod
    def from_env() -> "DBConfig":
        def need(name: str) -> str:
            val = os.getenv(name)
            if not val:
                raise RuntimeError(f"Missing required env var: {name}")
            return val

        return DBConfig(
            host=need("POSTGRES_HOST"),
            port=int(need("POSTGRES_PORT")),
            user=need("POSTGRES_USER"),
            password=need("POSTGRES_PASSWORD"),
            database=need("POSTGRES_DB"),
            # tweakables via env (optional)
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "5")),
            pool_recycle_seconds=int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800")),
            connect_timeout_seconds=int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10")),
            statement_timeout_ms=int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "60000")),
            application_name=os.getenv("DB_APP_NAME", "jaage_etl"),
            sslmode=os.getenv("DB_SSLMODE", "prefer"),
        )

    def dsn(self) -> str:
        # Use psycopg (v3) driver with SQLAlchemy 2.x
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.sslmode}&application_name={self.application_name}"
        )


# ---- Engine & session helpers ------------------------------------------------
@lru_cache(maxsize=1)
def get_engine(cfg: Optional[DBConfig] = None) -> Engine:
    """Singleton engine for the process."""
    cfg = cfg or DBConfig.from_env()
    engine = create_engine(
        cfg.dsn(),
        pool_size=cfg.pool_size,
        max_overflow=cfg.max_overflow,
        pool_recycle=cfg.pool_recycle_seconds,
        pool_pre_ping=True,  # validate pooled conns
        connect_args={
            "connect_timeout": cfg.connect_timeout_seconds,
            # Set a per-connection statement timeout (server-side)
            "options": f"-c statement_timeout={cfg.statement_timeout_ms}",
        },
        future=True,
        echo=False,
    )
    _assert_db_ready(engine)
    return engine


def _assert_db_ready(engine: Engine, retries: int = 5, backoff_base: float = 0.5) -> None:
    """Ping with exponential backoff; raise if not reachable."""
    err: Optional[Exception] = None
    for i in range(retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception as e:
            err = e
            time.sleep(backoff_base * (2 ** i))
    raise RuntimeError(f"DB connectivity check failed: {err}") from err


# Global sessionmaker (bound lazily)
_Session: Optional[sessionmaker] = None

def get_sessionmaker() -> sessionmaker:
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)
    return _Session


@contextmanager
def session_scope() -> Iterator:
    """
    Context-managed session with commit/rollback semantics.

    Usage:
        from sqlalchemy import text
        with session_scope() as s:
            s.execute(text("SELECT 1"))
    """
    Session = get_sessionmaker()
    s = Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


class Databaser:
    """
    Lightweight helper around SQLAlchemy 2.x sessions.
    - No long-lived session; always context-managed.
    - Convenience CRUD and Postgres upsert (ON CONFLICT).
    """

    def __init__(self, engine: Optional[Engine] = None):
        self.engine: Engine = engine or get_engine()  # define/import get_engine in your project
        self.SessionLocal: sessionmaker[Session] = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )

    # ---- session/transaction scope -----------------------------------------
    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ---- convenience methods ------------------------------------------------
    def get(self, model: type[T], pk: Any) -> Optional[T]:
        with self.session() as s:
            return s.get(model, pk)

    def add(self, obj: T) -> T:
        with self.session() as s:
            s.add(obj)
            return obj

    def add_all(self, objs: Iterable[T]) -> int:
        objs = list(objs)
        if not objs:
            return 0
        with self.session() as s:
            s.add_all(objs)
        return len(objs)

    def execute(self, stmt):
        with self.session() as s:
            return s.execute(stmt)

    def scalar(self, stmt):
        with self.session() as s:
            return s.scalar(stmt)

    # ---- Postgres upsert helpers (prefer these over pre-checks) -------------
    def upsert_one(
        self,
        model: type[T],
        values: Mapping[str, Any],
        conflict_cols: Sequence[str],
        update_cols: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Insert one row; if key conflict, update specified columns.
        """
        stmt = pg_insert(model).values(values)
        if update_cols is None:
            update_cols = [c for c in values.keys() if c not in conflict_cols]
        stmt = stmt.on_conflict_do_update(
            index_elements=list(conflict_cols),
            set_={c: getattr(stmt.excluded, c) for c in update_cols},
        )
        with self.session() as s:
            s.execute(stmt)

    def upsert_many(
        self,
        model: type[T],
        rows: Iterable[Mapping[str, Any]],
        conflict_cols: Sequence[str],
        update_cols: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
    ) -> int:
        """
        Batch upsert (efficient). Returns number of rows attempted.
        """
        total = 0
        batch: list[Mapping[str, Any]] = []
        for r in rows:
            batch.append(r)
            if len(batch) >= batch_size:
                total += self._upsert_batch(model, batch, conflict_cols, update_cols)
                batch.clear()
        if batch:
            total += self._upsert_batch(model, batch, conflict_cols, update_cols)
        return total

    def _upsert_batch(
        self,
        model: type[T],
        rows: list[Mapping[str, Any]],
        conflict_cols: Sequence[str],
        update_cols: Optional[Sequence[str]],
    ) -> int:
        stmt = pg_insert(model).values(rows)
        if update_cols is None:
            # union of keys across batch, minus conflict columns
            keys = set().union(*(r.keys() for r in rows))
            update_cols = [c for c in keys if c not in conflict_cols]
        stmt = stmt.on_conflict_do_update(
            index_elements=list(conflict_cols),
            set_={c: getattr(stmt.excluded, c) for c in update_cols},
        )
        with self.session() as s:
            s.execute(stmt)
        return len(rows)

    # ---- query helpers ------------------------------------------------------
    def one_or_none_by(self, model: type[T], **filters) -> Optional[T]:
        stmt = select(model).filter_by(**filters).limit(1)
        with self.session() as s:
            return s.scalars(stmt).first()