"""
tools/benchmarking.py
---------------------
Competitor benchmarking tools using yfinance.

Two tools exposed (async):
  - fetch_peer_group       → resolves ticker → sector → peer tickers
  - fetch_peer_metrics     → fetches valuation + financial metrics for a list of tickers
"""

from __future__ import annotations

import asyncio
import structlog
from pydantic import BaseModel, Field
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import yfinance as yf

log = structlog.get_logger(__name__)

_retry = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)

# ── Peer group map ────────────────────────────────────────────────────────────
# Curated peer groups by sector/industry. The agent resolves a ticker's
# industry via yfinance .info["industry"] and picks the right peer list.
# Extend this as needed — it is the core knowledge of this tool.

_PEER_GROUPS: dict[str, list[str]] = {
    # Technology
    "Software—Application":          ["MSFT", "ORCL", "SAP",  "CRM",  "NOW",  "ADBE"],
    "Software—Infrastructure":       ["MSFT", "AMZN", "GOOGL", "IBM",  "CSCO", "VMW"],
    "Semiconductors":                ["NVDA", "AMD",  "INTC", "QCOM", "AVGO", "TSM"],
    "Consumer Electronics":          ["AAPL", "SONY", "SMSN", "HPQ",  "DELL", "LNV"],
    "Internet Content & Information":["GOOGL","META", "SNAP", "PINS", "TWTR", "BIDU"],
    "Computer Hardware":             ["AAPL", "DELL", "HPQ",  "LNVGY","MSFT", "NTAP"],
    # Finance
    "Banks—Diversified":             ["JPM",  "BAC",  "WFC",  "C",    "GS",   "MS"],
    "Banks—Regional":                ["USB",  "PNC",  "TFC",  "FITB", "RF",   "CFG"],
    "Asset Management":              ["BLK",  "SCHW", "MS",   "GS",   "BAM",  "APO"],
    "Insurance—Life":                ["MET",  "PRU",  "LNC",  "AFL",  "AIG",  "PFG"],
    "Credit Services":               ["V",    "MA",   "AXP",  "DFS",  "COF",  "SYF"],
    # Healthcare
    "Drug Manufacturers—General":    ["JNJ",  "PFE",  "MRK",  "ABBV", "LLY",  "BMY"],
    "Biotechnology":                 ["AMGN", "GILD", "REGN", "VRTX", "BIIB", "MRNA"],
    "Medical Devices":               ["MDT",  "ABT",  "SYK",  "BSX",  "EW",   "ZBH"],
    "Health Care Plans":             ["UNH",  "CVS",  "CI",   "HUM",  "CNC",  "MOH"],
    # Consumer
    "Specialty Retail":              ["AMZN", "HD",   "LOW",  "TGT",  "COST", "WMT"],
    "Restaurants":                   ["MCD",  "SBUX", "YUM",  "QSR",  "DPZ",  "CMG"],
    "Beverages—Non-Alcoholic":       ["KO",   "PEP",  "MNST", "CELH", "FIZZ", "NRGV"],
    "Apparel Retail":                ["NKE",  "LULU", "GPS",  "PVH",  "RL",   "VFC"],
    # Energy
    "Oil & Gas Integrated":          ["XOM",  "CVX",  "COP",  "BP",   "SHEL", "TTE"],
    "Oil & Gas E&P":                 ["EOG",  "PXD",  "DVN",  "FANG", "MRO",  "APA"],
    "Oil & Gas Midstream":           ["ET",   "EPD",  "MMP",  "WMB",  "KMI",  "OKE"],
    # Industrials
    "Aerospace & Defense":           ["LMT",  "RTX",  "NOC",  "GD",   "BA",   "HII"],
    "Industrial Conglomerates":      ["GE",   "HON",  "MMM",  "EMR",  "ITW",  "ETN"],
    "Airlines":                      ["DAL",  "UAL",  "AAL",  "LUV",  "ALK",  "JBLU"],
    # Real Estate
    "REIT—Diversified":              ["O",    "WPC",  "NNN",  "SRC",  "STOR", "ADC"],
    "REIT—Retail":                   ["SPG",  "MAC",  "TCO",  "PEI",  "WPG",  "CBL"],
}

_FALLBACK_PEERS = 5   # max peers when no curated list found — use same-sector ETF components


# ── Arg schemas ───────────────────────────────────────────────────────────────

class PeerGroupInput(BaseModel):
    """Input for resolving a ticker's peer group."""
    symbol: str = Field(description="Stock ticker to find peers for, e.g. 'AAPL'.")

class PeerMetricsInput(BaseModel):
    """Input for fetching metrics across a list of tickers."""
    symbols: list[str] = Field(description="List of stock tickers to fetch metrics for.")


# ── Shared helpers ────────────────────────────────────────────────────────────

@_retry
def _get_info(symbol: str) -> dict:
    """Fetch yfinance .info with retry."""
    info = yf.Ticker(symbol).info
    if not info:
        raise ValueError(f"No data for '{symbol}'")
    return info


def _safe(info: dict, key: str, default=None):
    """Return info[key] or default — never raises."""
    val = info.get(key, default)
    return val if val not in (None, "", "N/A", "nan") else default


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("fetch_peer_group", args_schema=PeerGroupInput)
async def fetch_peer_group(symbol: str) -> dict:
    """
    Resolve a stock ticker to its peer group.

    Returns the target company's name, sector, industry, and a list of
    peer ticker symbols drawn from a curated industry map. Call this first,
    then pass the peer list to fetch_peer_metrics.
    """
    log.info("tool.fetch_peer_group", symbol=symbol)

    try:
        info     = await asyncio.to_thread(_get_info, symbol)
        industry = _safe(info, "industry", "")
        sector   = _safe(info, "sector",   "")
        name     = _safe(info, "longName", symbol)

        # Look up curated peer list, exclude the target ticker itself
        raw_peers = _PEER_GROUPS.get(industry, [])
        peers     = [t for t in raw_peers if t.upper() != symbol.upper()]

        # Fallback: if industry not mapped, pull same-sector peers from a
        # broad list by scanning all curated groups for matching sector tag.
        if not peers:
            log.warning("tool.fetch_peer_group.no_curated_list",
                        symbol=symbol, industry=industry)
            seen: set[str] = set()
            for tickers in _PEER_GROUPS.values():
                for t in tickers:
                    if t.upper() != symbol.upper():
                        seen.add(t)
            # Can't do sector filter without fetching every ticker — just
            # return an empty list and let the agent handle it gracefully.
            peers = []

        result = {
            "target": {
                "symbol":   symbol.upper(),
                "name":     name,
                "sector":   sector,
                "industry": industry,
            },
            "peers": peers[:6],   # cap at 6 peers to keep LLM context lean
            "peer_count": len(peers[:6]),
            "industry_mapped": bool(raw_peers),
        }

        log.info("tool.fetch_peer_group.done", symbol=symbol,
                 industry=industry, peer_count=len(peers[:6]))
        return result

    except Exception as exc:
        log.error("tool.fetch_peer_group.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}


@tool("fetch_peer_metrics", args_schema=PeerMetricsInput)
async def fetch_peer_metrics(symbols: list[str]) -> dict:
    """
    Fetch a standardised set of valuation and financial metrics for a list
    of tickers in parallel.

    Returns a dict keyed by ticker with: valuation multiples, profitability
    ratios, growth, analyst target, and market cap. Use this to build the
    comparison table in the benchmark report.
    """
    log.info("tool.fetch_peer_metrics", symbols=symbols)

    async def _fetch_one(symbol: str) -> tuple[str, dict]:
        try:
            info = await asyncio.to_thread(_get_info, symbol)
            return symbol, {
                # Identity
                "name":     _safe(info, "longName", symbol),
                "sector":   _safe(info, "sector"),
                "industry": _safe(info, "industry"),
                # Size
                "marketCap":         _safe(info, "marketCap"),
                "enterpriseValue":   _safe(info, "enterpriseValue"),
                # Valuation multiples
                "trailingPE":        _safe(info, "trailingPE"),
                "forwardPE":         _safe(info, "forwardPE"),
                "priceToBook":       _safe(info, "priceToBook"),
                "priceToSales":      _safe(info, "priceToSalesTrailing12Months"),
                "pegRatio":          _safe(info, "pegRatio"),
                "evToEbitda":        _safe(info, "enterpriseToEbitda"),
                "evToRevenue":       _safe(info, "enterpriseToRevenue"),
                # Profitability
                "grossMargins":      _safe(info, "grossMargins"),
                "operatingMargins":  _safe(info, "operatingMargins"),
                "profitMargins":     _safe(info, "profitMargins"),
                "returnOnEquity":    _safe(info, "returnOnEquity"),
                "returnOnAssets":    _safe(info, "returnOnAssets"),
                # Leverage / liquidity
                "debtToEquity":      _safe(info, "debtToEquity"),
                "currentRatio":      _safe(info, "currentRatio"),
                # Growth
                "revenueGrowth":     _safe(info, "revenueGrowth"),
                "earningsGrowth":    _safe(info, "earningsGrowth"),
                # Analyst consensus
                "analystConsensus":  _safe(info, "recommendationKey", "").replace("_", " ").title(),
                "targetMeanPrice":   _safe(info, "targetMeanPrice"),
                "numberOfAnalysts":  _safe(info, "numberOfAnalystOpinions"),
                # Price
                "currentPrice":      _safe(info, "currentPrice") or _safe(info, "regularMarketPrice"),
                "fiftyTwoWeekHigh":  _safe(info, "fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow":   _safe(info, "fiftyTwoWeekLow"),
                "dividendYield":     _safe(info, "dividendYield"),
            }
        except Exception as exc:
            log.warning("tool.fetch_peer_metrics.single_error",
                        symbol=symbol, error=str(exc))
            return symbol, {"error": str(exc)}

    # Fetch all tickers concurrently
    tasks   = [_fetch_one(s) for s in symbols]
    results = await asyncio.gather(*tasks)

    output = dict(results)
    log.info("tool.fetch_peer_metrics.done",
             fetched=len([v for v in output.values() if "error" not in v]),
             failed=len([v for v in output.values() if "error" in v]))
    return output