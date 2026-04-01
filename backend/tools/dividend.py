"""
tools/dividend_capture.py
--------------------------
Dividend capture strategy tools for identifying and executing dividend opportunities.

Three async tools exposed:
  - fetch_dividend_calendar    → upcoming ex-dividend dates with yields
  - calculate_capture_profit   → expected profit/loss analysis for a trade
  - screen_capture_opportunities → filter best dividend capture candidates
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Literal
import structlog
import yfinance as yf
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

_retry = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


# ── Arg schemas ───────────────────────────────────────────────────────────────

class DividendCalendarInput(BaseModel):
    symbols: list[str] = Field(
        description="List of stock tickers to check for upcoming dividends, e.g. ['AAPL', 'MSFT', 'KO']."
    )
    days_ahead: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Number of days to look ahead for ex-dividend dates (7-90).",
    )


class CaptureProfitInput(BaseModel):
    symbol: str = Field(description="Stock ticker, e.g. 'KO' or 'T'.")
    entry_price: float = Field(description="Expected entry price per share, e.g. 71.23")
    shares: int = Field(description="Number of shares to buy, e.g. 100")
    dividend_per_share: float = Field(description="Expected dividend per share, e.g. 0.485")
    expected_ex_drop_pct: float = Field(
        default=100.0,
        description="Expected price drop on ex-date as % of dividend (default 100% = full dividend drop).",
    )
    hold_days: int = Field(
        default=1,
        ge=1,
        le=7,
        description="Days to hold after ex-date before selling (default 1).",
    )
    tax_rate: float = Field(
        default=0.22,
        description="Ordinary income tax rate for short-term dividends (default 22%).",
    )
    commission_per_trade: float = Field(
        default=0.0,
        description="Brokerage commission per trade (default $0 for zero-commission brokers).",
    )


class ScreenOpportunitiesInput(BaseModel):
    min_yield: float = Field(
        default=0.015,
        description="Minimum dividend yield to consider (e.g. 0.015 = 1.5%).",
    )
    min_volume: int = Field(
        default=1_000_000,
        description="Minimum average daily volume for liquidity.",
    )
    max_volatility_pct: float = Field(
        default=3.0,
        description="Maximum daily volatility % (ATR/price) to avoid high-risk stocks.",
    )
    days_ahead: int = Field(
        default=14,
        ge=7,
        le=30,
        description="Days ahead to scan for ex-dividend dates.",
    )
    universe: Literal["sp500", "dow30", "nasdaq100", "custom"] = Field(
        default="sp500",
        description="Stock universe to screen: sp500, dow30, nasdaq100, or custom list.",
    )
    custom_symbols: list[str] | None = Field(
        default=None,
        description="Custom ticker list if universe='custom'.",
    )


# ── Shared helpers ────────────────────────────────────────────────────────────

@_retry
def _get_ticker_info(symbol: str) -> dict:
    """Fetch yfinance .info + dividend data."""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if not info:
        raise ValueError(f"No data for '{symbol}'")
    
    # Get dividend history (last 2 years)
    try:
        divs = ticker.dividends
        if divs is not None and not divs.empty:
            divs = divs[divs.index > (datetime.now() - timedelta(days=730))]
        else:
            divs = pd.Series(dtype=float)
    except Exception:
        divs = pd.Series(dtype=float)
    
    return {"info": info, "dividends": divs}


def _safe_float(val, decimals=4) -> float | None:
    """Convert to float, None if NaN/inf."""
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _estimate_ex_date(last_div_date: pd.Timestamp, frequency: str) -> datetime | None:
    """Estimate next ex-dividend date based on frequency."""
    if pd.isna(last_div_date):
        return None
    
    freq_map = {
        "quarterly": 90,
        "monthly": 30,
        "annual": 365,
        "semi-annual": 180,
    }
    
    days = freq_map.get(frequency.lower(), 90)
    return last_div_date + timedelta(days=days)


# ── S&P 500 / Dow 30 / Nasdaq 100 tickers (subset for demo) ───────────────────

_SP500_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "JNJ",
    "WMT", "JPM", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "KO",
    "PFE", "ABBV", "AVGO", "MRK", "CVX", "PEP", "COST", "TMO", "CSCO", "ABT",
    "ACN", "LLY", "NKE", "DHR", "VZ", "ADBE", "NEE", "MCD", "BMY", "TXN",
    "PM", "T", "HON", "UNP", "UPS", "IBM", "QCOM", "LOW", "RTX", "SPGI",
]

_DOW30 = [
    "AAPL", "MSFT", "UNH", "JNJ", "V", "JPM", "WMT", "PG", "HD", "CVX",
    "MRK", "DIS", "KO", "MCD", "CSCO", "VZ", "AXP", "IBM", "AMGN", "HON",
    "CAT", "BA", "GS", "NKE", "MMM", "TRV", "CRM", "INTC", "WBA", "DOW",
]

_NASDAQ100_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST", "ASML",
    "PEP", "ADBE", "CSCO", "NFLX", "CMCSA", "INTC", "AMD", "QCOM", "INTU", "TXN",
    "AMGN", "HON", "SBUX", "AMAT", "ISRG", "BKNG", "GILD", "ADP", "ADI", "VRTX",
]


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("fetch_dividend_calendar", args_schema=DividendCalendarInput)
async def fetch_dividend_calendar(
    symbols: list[str],
    days_ahead: int = 30,
) -> list[dict]:
    """
    Fetch upcoming ex-dividend dates and dividend amounts for a list of stocks.

    Returns a calendar of dividend opportunities with: symbol, next ex-date,
    dividend amount, yield, frequency, and current price. Use this to plan
    dividend capture trades.
    """
    log.info("tool.fetch_dividend_calendar", symbols=symbols, days=days_ahead)

    async def _fetch_one(symbol: str) -> dict | None:
        try:
            data = await asyncio.to_thread(_get_ticker_info, symbol)
            info = data["info"]
            divs = data["dividends"]

            if divs.empty:
                return None  # No dividend history

            # Most recent dividend
            last_div_date = divs.index[-1]
            last_div_amount = float(divs.iloc[-1])

            # Estimate frequency (quarterly if ~3 months between last 2 payments)
            if len(divs) >= 2:
                days_between = (divs.index[-1] - divs.index[-2]).days
                if 80 <= days_between <= 100:
                    frequency = "quarterly"
                elif 25 <= days_between <= 35:
                    frequency = "monthly"
                elif days_between > 300:
                    frequency = "annual"
                else:
                    frequency = "semi-annual"
            else:
                frequency = "quarterly"

            # Estimate next ex-date
            next_ex_date = _estimate_ex_date(last_div_date, frequency)
            if not next_ex_date:
                return None

            # Check if within days_ahead window
            days_until_ex = (next_ex_date - datetime.now()).days
            if days_until_ex < 0 or days_until_ex > days_ahead:
                return None

            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not current_price:
                return None

            div_yield = last_div_amount / current_price

            return {
                "symbol": symbol.upper(),
                "ex_date": next_ex_date.strftime("%Y-%m-%d"),
                "days_until_ex": days_until_ex,
                "dividend_amount": _safe_float(last_div_amount, 3),
                "current_price": _safe_float(current_price, 2),
                "dividend_yield": _safe_float(div_yield, 4),
                "frequency": frequency,
                "avg_volume": info.get("averageVolume"),
            }

        except Exception as exc:
            log.warning("tool.fetch_dividend_calendar.single_error",
                        symbol=symbol, error=str(exc))
            return None

    # Fetch all symbols in parallel
    tasks = [_fetch_one(s) for s in symbols]
    results = await asyncio.gather(*tasks)

    # Filter out None and sort by days_until_ex
    calendar = [r for r in results if r is not None]
    calendar.sort(key=lambda x: x["days_until_ex"])

    log.info("tool.fetch_dividend_calendar.done", found=len(calendar))
    return calendar


@tool("calculate_capture_profit", args_schema=CaptureProfitInput)
async def calculate_capture_profit(
    symbol: str,
    entry_price: float,
    shares: int,
    dividend_per_share: float,
    expected_ex_drop_pct: float = 100.0,
    hold_days: int = 1,
    tax_rate: float = 0.22,
    commission_per_trade: float = 0.0,
) -> dict:
    """
    Calculate expected profit/loss for a dividend capture trade.

    Models the economics: entry cost, dividend income (after tax), price drop on ex-date,
    recovery over hold period, exit proceeds, and net profit. Use this to evaluate
    whether a specific trade is worth executing.
    """
    log.info("tool.calculate_capture_profit", symbol=symbol, shares=shares)

    try:
        # ── Costs ──
        entry_cost = entry_price * shares + commission_per_trade
        exit_commission = commission_per_trade
        entry_commission = entry_cost - (entry_price * shares)

        # ── Dividend income (after tax) ──
        gross_dividend = dividend_per_share * shares
        tax_on_dividend = gross_dividend * tax_rate
        net_dividend = gross_dividend - tax_on_dividend

        # ── Price movement simulation ──
        # Ex-date drop = dividend * (expected_ex_drop_pct / 100)
        ex_drop = dividend_per_share * (expected_ex_drop_pct / 100.0)
        ex_price = entry_price - ex_drop

        # Recovery: assume 0.5% daily price recovery (market drift + reversion)
        # This is a VERY optimistic assumption — real recovery varies
        recovery_per_day = entry_price * 0.005
        exit_price = ex_price + (recovery_per_day * hold_days)

        # Cap exit price at entry (can't recover above entry in short period)
        exit_price = min(exit_price, entry_price - 0.01)

        # ── Exit proceeds ──
        exit_proceeds = exit_price * shares - exit_commission

        # ── Net profit ──
        net_profit = exit_proceeds + net_dividend - entry_cost
        net_profit_pct = (net_profit / entry_cost) * 100

        # ── Breakeven analysis ──
        # What exit price is needed to break even?
        breakeven_exit_price = (entry_cost - net_dividend + exit_commission) / shares

        output = {
            "symbol": symbol.upper(),
            "shares": shares,
            "entry_price": _safe_float(entry_price, 2),
            "entry_cost": _safe_float(entry_cost, 2),
            "dividend_gross": _safe_float(gross_dividend, 2),
            "dividend_after_tax": _safe_float(net_dividend, 2),
            "tax_paid": _safe_float(tax_on_dividend, 2),
            "ex_date_drop": _safe_float(ex_drop, 3),
            "ex_date_price": _safe_float(ex_price, 2),
            "expected_exit_price": _safe_float(exit_price, 2),
            "exit_proceeds": _safe_float(exit_proceeds, 2),
            "net_profit": _safe_float(net_profit, 2),
            "net_profit_pct": _safe_float(net_profit_pct, 2),
            "breakeven_exit_price": _safe_float(breakeven_exit_price, 2),
            "hold_days": hold_days,
            "commissions_total": _safe_float(entry_commission + exit_commission, 2),
            "verdict": (
                "PROFITABLE" if net_profit > 0
                else "BREAK-EVEN" if abs(net_profit) < 1
                else "LOSS"
            ),
        }

        log.info("tool.calculate_capture_profit.done", symbol=symbol,
                 profit=_safe_float(net_profit, 2))
        return output

    except Exception as exc:
        log.error("tool.calculate_capture_profit.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}


@tool("screen_capture_opportunities", args_schema=ScreenOpportunitiesInput)
async def screen_capture_opportunities(
    min_yield: float = 0.015,
    min_volume: int = 1_000_000,
    max_volatility_pct: float = 3.0,
    days_ahead: int = 14,
    universe: Literal["sp500", "dow30", "nasdaq100", "custom"] = "sp500",
    custom_symbols: list[str] | None = None,
) -> list[dict]:
    """
    Screen for high-quality dividend capture opportunities.

    Filters stocks by: upcoming ex-date, minimum yield, liquidity (volume),
    and volatility. Returns ranked list of candidates with scores. Use this
    to find the best dividend capture trades each week.
    """
    log.info("tool.screen_capture_opportunities", universe=universe, days=days_ahead)

    try:
        # Select universe
        if universe == "sp500":
            symbols = _SP500_SAMPLE
        elif universe == "dow30":
            symbols = _DOW30
        elif universe == "nasdaq100":
            symbols = _NASDAQ100_SAMPLE
        elif universe == "custom":
            symbols = custom_symbols or []
        else:
            raise ValueError(f"Invalid universe: {universe}")

        if not symbols:
            raise ValueError("No symbols to screen")

  
        # Step 1: Get dividend calendar
        calendar = await fetch_dividend_calendar.ainvoke({
            "symbols": symbols,
            "days_ahead": days_ahead
        })

        if not calendar:
            return []

        # Step 2: Filter by yield
        candidates = [c for c in calendar if (c.get("dividend_yield") or 0) >= min_yield]

        # Step 3: Fetch volatility and filter
        async def _add_volatility(candidate: dict) -> dict | None:
            try:
                sym = candidate["symbol"]
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1mo")
                
                if hist is None or hist.empty or len(hist) < 10:
                    return None
                
                # ATR (14-day)
                high = hist["High"]
                low = hist["Low"]
                close = hist["Close"]
                
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ], axis=1).max(axis=1)
                
                atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
                current_price = candidate["current_price"]
                volatility_pct = (atr / current_price) * 100 if current_price else 99.0
                
                # Volume check
                avg_vol = candidate.get("avg_volume", 0)
                
                if volatility_pct > max_volatility_pct or avg_vol < min_volume:
                    return None
                
                # Score: yield + inverse_volatility
                # Higher yield = good, lower volatility = good
                score = (candidate["dividend_yield"] * 100) + (1 / volatility_pct)
                
                return {
                    **candidate,
                    "volatility_pct": _safe_float(volatility_pct, 2),
                    "score": _safe_float(score, 2),
                }
                
            except Exception:
                return None

        # Fetch volatility in parallel
        tasks = [_add_volatility(c) for c in candidates]
        enriched = await asyncio.gather(*tasks)
        
        # Filter None and sort by score (descending)
        opportunities = [e for e in enriched if e is not None]
        opportunities.sort(key=lambda x: x["score"], reverse=True)

        log.info("tool.screen_capture_opportunities.done", count=len(opportunities))
        return opportunities[:20]  # Top 20

    except Exception as exc:
        log.error("tool.screen_capture_opportunities.error", error=str(exc))
        return [{"error": str(exc)}]