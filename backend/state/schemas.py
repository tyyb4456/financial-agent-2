"""
State schemas
-------------
Every LangGraph StateGraph needs a TypedDict that defines what data
flows through its nodes.  All schemas live here so imports are clean:

    from state.schemas import FinancialState, RedditState, ...

Design rules:
- Input fields come in from the API (required).
- Intermediate fields are filled in by nodes.
- `result` is always the final human-readable output string.
- All fields default to None so nodes can be run independently in tests.
"""

from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict


# ── Financial report ──────────────────────────────────────────────────────────

class FinancialState(TypedDict):
    # inputs
    symbol: str
    # intermediate
    raw_financials: Optional[dict]
    # output
    result: Optional[str]


# ── Investment advisor ────────────────────────────────────────────────────────

class InvestmentState(TypedDict):
    # inputs
    symbol: str
    # intermediate
    raw_financials: Optional[dict]
    raw_investment:  Optional[dict]
    # output
    result: Optional[str]


# ── Financial Q&A ─────────────────────────────────────────────────────────────

class QueryState(TypedDict):
    # inputs
    symbol: str
    query:  str
    # intermediate
    raw_financials: Optional[dict]
    raw_investment:  Optional[dict]
    # output
    result: Optional[str]
