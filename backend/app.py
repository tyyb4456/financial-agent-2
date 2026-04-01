"""
app.py
------
Main FastAPI application for the Financial AI multi-agent system.

Endpoints:
  POST   /chat                          — conversational endpoint (with memory)
  GET    /chat/threads                  — list all active thread IDs
  GET    /chat/threads/{thread_id}/history — fetch message history for a thread
  DELETE /chat/threads/{thread_id}      — clear a conversation thread
  POST   /analyze/{symbol}              — quick full stock analysis (fire-and-forget)
  GET    /health                        — health check

Memory:
  Uses AsyncPostgresSaver (LangGraph) backed by Postgres.
  Set DATABASE_URL in .env — see below.

.env required keys:
  DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname
  GOOGLE_API_KEY=...
  (optional) CHECKPOINT_SCHEMA=langgraph   # Postgres schema for checkpoint tables
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memory import get_checkpointer, close_checkpointer
from agents.superior import get_main_agent, _extract_text
from agents.financial_reporting import financial_reporter_agent
from agents.investment_advise import investment_advisor_agent
from agents.technical_analysis import technical_analysis_agent

log = structlog.get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open DB connection + build agents at startup. Close cleanly on shutdown."""
    log.info("app.startup")
    await get_checkpointer()   # opens Postgres pool + runs setup()
    await get_main_agent()     # compiles agent with checkpointer
    log.info("app.ready")
    yield
    log.info("app.shutdown")
    await close_checkpointer()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial AI API",
    version="2.0.0",
    description=(
        "Multi-agent financial analysis system with persistent conversation memory. "
        "Use `thread_id` to maintain context across turns."
    ),
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(
        description="The user's message or financial question.",
        examples=["Analyze NVDA for me", "Compare it to AMD"]
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation thread ID. Pass the same value to continue a previous session. "
            "Omit to start a fresh conversation — a new ID will be generated and returned."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Give me a full analysis of AAPL", "thread_id": None},
                {"message": "Now compare it to MSFT", "thread_id": "user-42-abc123"},
            ]
        }
    }


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class MessageOut(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[MessageOut]


class ThreadListResponse(BaseModel):
    threads: list[str]
    count: int


class AnalyzeResponse(BaseModel):
    symbol: str
    financial: str
    investment: str
    technical: str


class HealthResponse(BaseModel):
    status: str
    version: str
    memory_backend: str


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health() -> HealthResponse:
    """Returns service status and memory backend info."""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        memory_backend="postgres",
    )


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Send a message to the financial assistant",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message and get a response from the multi-agent financial assistant.

    The assistant routes your query to specialist sub-agents (financial analysis,
    investment advice, technical analysis, news, Reddit sentiment, etc.) and
    synthesises their outputs into a single coherent response.

    Pass the same `thread_id` across turns to maintain full conversation context —
    the assistant remembers previously analysed stocks, follow-up questions, and
    prior recommendations within the same thread.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    log.info("chat.request", thread_id=thread_id[:8], msg=request.message[:80])

    try:
        agent = await get_main_agent()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config,
        )
        reply = _extract_text(result["messages"][-1].content)
        log.info("chat.ok", thread_id=thread_id[:8], reply_len=len(reply))
        return ChatResponse(reply=reply, thread_id=thread_id)

    except Exception as exc:
        log.error("chat.error", thread_id=thread_id[:8], error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── Threads ────────────────────────────────────────────────────────────────────

@app.get(
    "/chat/threads",
    response_model=ThreadListResponse,
    tags=["Threads"],
    summary="List all active conversation threads",
)
async def list_threads(
    limit: int = Query(default=50, ge=1, le=200, description="Max threads to return"),
) -> ThreadListResponse:
    """
    Returns a list of all thread IDs that have at least one saved checkpoint.
    Useful for building a conversation sidebar in the UI.
    """
    try:
        checkpointer = await get_checkpointer()

        # AsyncPostgresSaver.alist() streams all checkpoints — collect unique thread IDs
        seen: set[str] = set()
        async for item in checkpointer.alist(None):   # None = no filter
            tid = item.config["configurable"].get("thread_id")
            if tid:
                seen.add(tid)
            if len(seen) >= limit:
                break

        threads = sorted(seen)
        return ThreadListResponse(threads=threads, count=len(threads))

    except Exception as exc:
        log.error("list_threads.error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/chat/threads/{thread_id}/history",
    response_model=HistoryResponse,
    tags=["Threads"],
    summary="Get message history for a thread",
)
async def get_thread_history(
    thread_id: str = Path(description="The conversation thread ID"),
) -> HistoryResponse:
    """
    Returns the full message history for a conversation thread.

    Filters out internal tool messages — only user and assistant messages are returned.
    Useful for rebuilding the conversation UI after a page refresh.
    """
    try:
        checkpointer = await get_checkpointer()
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await checkpointer.aget(config)

        if checkpoint is None:
            return HistoryResponse(thread_id=thread_id, messages=[])

        raw_messages = checkpoint.get("channel_values", {}).get("messages", [])
        history: list[MessageOut] = []

        for msg in raw_messages:
            cls = msg.__class__.__name__
            if cls == "ToolMessage":
                continue   # skip internal tool call/result pairs

            content = _extract_text(msg.content)
            if not content.strip():
                continue   # skip empty assistant messages mid-tool-call

            role = "assistant" if cls == "AIMessage" else "user"
            history.append(MessageOut(role=role, content=content))

        return HistoryResponse(thread_id=thread_id, messages=history)

    except Exception as exc:
        log.error("get_history.error", thread_id=thread_id[:8], error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete(
    "/chat/threads/{thread_id}",
    tags=["Threads"],
    summary="Clear a conversation thread",
)
async def delete_thread(
    thread_id: str = Path(description="The conversation thread ID to clear"),
) -> JSONResponse:
    """
    Deletes all checkpoints for the given thread from the Postgres database,
    effectively clearing the conversation history.

    After deletion, the same `thread_id` can be reused for a fresh conversation,
    or the client can generate a new one.
    """
    try:
        checkpointer = await get_checkpointer()

        # Delete all checkpoints for this thread via raw SQL on the pool
        async with checkpointer.conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (thread_id,),
            )
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (thread_id,),
            )
        await checkpointer.conn.commit()

        log.info("delete_thread.ok", thread_id=thread_id[:8])
        return JSONResponse({
            "thread_id": thread_id,
            "status": "deleted",
            "message": "Conversation history cleared. You can reuse this thread_id or start with a new one.",
        })

    except Exception as exc:
        log.error("delete_thread.error", thread_id=thread_id[:8], error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── Quick stock analysis ───────────────────────────────────────────────────────

@app.post(
    "/analyze/{symbol}",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Full stock analysis — financial + investment + technical",
)
async def analyze_symbol(
    symbol: str = Path(description="Stock ticker symbol, e.g. AAPL, TSLA, NVDA"),
) -> AnalyzeResponse:
    """
    Runs financial reporting, investment advice, and technical analysis on a stock
    **in parallel** and returns all three reports in one response.

    This is a stateless endpoint — results are NOT saved to any conversation thread.
    Use `POST /chat` if you want the analysis to be part of a persistent conversation.
    """
    symbol = symbol.upper().strip()
    log.info("analyze.start", symbol=symbol)

    import asyncio

    async def _financial():
        try:
            result = await financial_reporter_agent.ainvoke({
                "messages": [{"role": "user", "content": symbol}]
            })
            return _extract_text(result["messages"][-1].content)
        except Exception as exc:
            return f"Error: {exc}"

    async def _investment():
        try:
            result = await investment_advisor_agent.ainvoke({
                "messages": [{"role": "user", "content": symbol}]
            })
            return _extract_text(result["messages"][-1].content)
        except Exception as exc:
            return f"Error: {exc}"

    async def _technical():
        try:
            result = await technical_analysis_agent.ainvoke({
                "messages": [{"role": "user", "content": symbol}]
            })
            return _extract_text(result["messages"][-1].content)
        except Exception as exc:
            return f"Error: {exc}"

    financial, investment, technical = await asyncio.gather(
        _financial(), _investment(), _technical()
    )

    log.info("analyze.done", symbol=symbol)
    return AnalyzeResponse(
        symbol=symbol,
        financial=financial,
        investment=investment,
        technical=technical,
    )