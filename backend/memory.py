"""
memory.py
---------
Singleton AsyncPostgresSaver checkpointer for the entire multi-agent system.

Why a singleton?
  AsyncPostgresSaver holds an async connection pool. Creating one per request
  exhausts connections fast. A single shared instance is the correct pattern.

Setup:
  1. pip install langgraph-checkpoint-postgres psycopg[binary] psycopg-pool
  2. Add to .env:
       DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname

  On first startup, setup() is called automatically — it creates the
  `checkpoints`, `checkpoint_writes`, and `checkpoint_blobs` tables in your DB.
  You do NOT need to run any migrations manually.

Usage:
    from backend.memory import get_checkpointer, close_checkpointer
"""

from __future__ import annotations

import os
import asyncio
import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
load_dotenv()  # Load .env file for DATABASE_URL
log = structlog.get_logger(__name__)

# Read from .env — example:
# DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/financial_ai
_DATABASE_URL = os.getenv("DATABASE_URL", "")

# Module-level singleton
_checkpointer: AsyncPostgresSaver | None = None
_lock = asyncio.Lock()

_saver_cm = None
_saver = None

async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Return the shared AsyncPostgresSaver instance, creating it on first call.

    - Opens the async connection pool
    - Calls setup() to create LangGraph tables if they don't exist yet
    - Thread-safe via asyncio.Lock

    Call once at startup (via lifespan) and reuse everywhere.
    """
    global _checkpointer
    global _saver, _saver_cm

    if _checkpointer is not None:
        return _checkpointer

    async with _lock:
        if _checkpointer is not None:
            return _checkpointer

        if not _DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Add it to .env: postgresql+psycopg://user:pass@host:5432/dbname"
            )

        log.info("memory.checkpointer.init", url=_DATABASE_URL.split("@")[-1])  # log host only

        if _saver is None:
            _saver_cm = AsyncPostgresSaver.from_conn_string(_DATABASE_URL)
            _saver = await _saver_cm.__aenter__()   # enter properly
            await _saver.setup()
   
        _checkpointer = _saver
        log.info("memory.checkpointer.ready")
        return _checkpointer


async def close_checkpointer() -> None:
    """
    Gracefully close the Postgres connection pool.
    Call this in the FastAPI lifespan shutdown handler.
    """
    global _checkpointer
    if _checkpointer is not None:
        try:
            await _checkpointer.__aexit__(None, None, None)
            log.info("memory.checkpointer.closed")
        except Exception as exc:
            log.warning("memory.checkpointer.close_error", error=str(exc))
        finally:
            _checkpointer = None