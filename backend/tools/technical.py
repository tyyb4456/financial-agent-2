"""
tools/technical.py
------------------
Technical analysis tools using yfinance OHLCV price history.
All indicators are computed from raw price/volume data — no TA-Lib dependency.

Three tools exposed (async):
  - fetch_price_history     → raw OHLCV dataframe → serialised dict
  - fetch_indicators        → trend, momentum, volatility, volume indicators
  - fetch_support_resistance→ key S/R levels + chart pattern signals
"""

from __future__ import annotations

import asyncio
import math
from typing import Optional

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

class PriceHistoryInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. 'AAPL'.")
    period: str = Field(
        default="6mo",
        description=(
            "Lookback period for price history. "
            "Valid values: '1mo','3mo','6mo','1y','2y'. "
            "Default '6mo' is suitable for most technical analysis."
        ),
    )


class IndicatorsInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. 'AAPL'.")
    period: str = Field(
        default="6mo",
        description="Lookback period. '6mo' or '1y' recommended for indicator accuracy.",
    )


class SupportResistanceInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. 'AAPL'.")
    period: str = Field(
        default="6mo",
        description="Lookback period for pivot and level detection.",
    )


# ── Pure-Python indicator math ────────────────────────────────────────────────
# All functions take a pd.Series or np.ndarray and return float / pd.Series.
# No external TA library required.

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, window=20, num_std=2) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = _sma(close, window)
    std   = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period=14, d_period=3) -> tuple[pd.Series, pd.Series]:
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, float("nan"))
    d = k.rolling(d_period).mean()
    return k, d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_tpv       = (typical_price * volume).cumsum()
    cum_vol       = volume.cumsum()
    return cum_tpv / cum_vol.replace(0, float("nan"))


def _safe_float(val) -> Optional[float]:
    """Convert numpy scalar / float to Python float, None if NaN/inf."""
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ── Shared data fetcher ───────────────────────────────────────────────────────

@_retry
def _download_ohlcv(symbol: str, period: str) -> pd.DataFrame:
    """Download OHLCV history and validate it."""
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty or len(df) < 30:
        raise ValueError(f"Insufficient price data for '{symbol}' (period={period}).")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna(subset=["Close"])
    return df


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("fetch_price_history", args_schema=PriceHistoryInput)
async def fetch_price_history(symbol: str, period: str = "6mo") -> dict:
    """
    Fetch OHLCV price history for a stock ticker.

    Returns the last 30 trading days of open, high, low, close, volume data
    plus basic price statistics (current price, period return, avg volume).
    Call this when you need raw price data or a price summary.
    """
    log.info("tool.fetch_price_history", symbol=symbol, period=period)
    try:
        df = await asyncio.to_thread(_download_ohlcv, symbol, period)

        close   = df["Close"]
        volume  = df["Volume"]
        last_30 = df.tail(30)

        # Period return
        period_return = _safe_float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)

        # Rolling 30-day return
        ret_30d = _safe_float((close.iloc[-1] - close.iloc[-30]) / close.iloc[-30] * 100) \
                  if len(close) >= 30 else None

        result = {
            "symbol":         symbol.upper(),
            "period":         period,
            "data_points":    len(df),
            "current_price":  _safe_float(close.iloc[-1]),
            "period_open":    _safe_float(close.iloc[0]),
            "period_high":    _safe_float(df["High"].max()),
            "period_low":     _safe_float(df["Low"].min()),
            "period_return_pct": period_return,
            "return_30d_pct":    ret_30d,
            "avg_volume":     _safe_float(volume.mean()),
            "last_volume":    _safe_float(volume.iloc[-1]),
            # Last 30 sessions serialised for LLM context
            "recent_closes": [
                {"date": str(d.date()), "close": _safe_float(c), "volume": int(v)}
                for d, c, v in zip(last_30.index, last_30["Close"], last_30["Volume"])
            ],
        }

        log.info("tool.fetch_price_history.done", symbol=symbol, rows=len(df))
        return result

    except Exception as exc:
        log.error("tool.fetch_price_history.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}


@tool("fetch_indicators", args_schema=IndicatorsInput)
async def fetch_indicators(symbol: str, period: str = "6mo") -> dict:
    """
    Compute a full suite of technical indicators for a stock.

    Covers:
      - Trend:      SMA(20/50/200), EMA(9/21), price vs MAs, golden/death cross
      - Momentum:   RSI(14), MACD(12/26/9), Stochastic(%K/%D)
      - Volatility: Bollinger Bands(20,2), ATR(14), BB %B, BB Width
      - Volume:     OBV, VWAP, volume vs 20-day avg
      - Signals:    plain-English signal for each indicator group

    Call this to get a complete technical picture before writing the report.
    """
    log.info("tool.fetch_indicators", symbol=symbol, period=period)
    try:
        df = await asyncio.to_thread(_download_ohlcv, symbol, period)

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # ── Trend ──────────────────────────────────────────────────────────
        sma20  = _sma(close, 20)
        sma50  = _sma(close, 50)
        sma200 = _sma(close, 200)
        ema9   = _ema(close, 9)
        ema21  = _ema(close, 21)

        price_now  = close.iloc[-1]
        sma20_now  = sma20.iloc[-1]
        sma50_now  = sma50.iloc[-1]
        sma200_now = sma200.iloc[-1]
        ema9_now   = ema9.iloc[-1]
        ema21_now  = ema21.iloc[-1]

        # Golden / death cross — sma50 vs sma200 crossover in last 5 sessions
        cross_signal = "none"
        if len(sma50.dropna()) > 5 and len(sma200.dropna()) > 5:
            s50  = sma50.dropna().iloc[-5:]
            s200 = sma200.dropna().iloc[-5:]
            aligned = pd.concat([s50, s200], axis=1).dropna()
            if len(aligned) >= 2:
                prev_diff = aligned.iloc[-2, 0] - aligned.iloc[-2, 1]
                curr_diff = aligned.iloc[-1, 0] - aligned.iloc[-1, 1]
                if prev_diff < 0 and curr_diff > 0:
                    cross_signal = "golden_cross"
                elif prev_diff > 0 and curr_diff < 0:
                    cross_signal = "death_cross"

        above_sma20  = bool(price_now > sma20_now)  if not math.isnan(sma20_now)  else None
        above_sma50  = bool(price_now > sma50_now)  if not math.isnan(sma50_now)  else None
        above_sma200 = bool(price_now > sma200_now) if not math.isnan(sma200_now) else None

        mas_aligned_bullish = (
            above_sma20 and above_sma50 and above_sma200
            and _safe_float(sma20_now) > _safe_float(sma50_now) > _safe_float(sma200_now)
        ) if all(v is not None for v in [above_sma20, above_sma50, above_sma200]) else None

        trend_signal = (
            "strongly bullish" if mas_aligned_bullish
            else "bullish"     if (above_sma50 and above_sma200)
            else "bearish"     if (not above_sma50 and not above_sma200)
            else "mixed"
        )

        # ── Momentum ───────────────────────────────────────────────────────
        rsi_series                   = _rsi(close)
        macd_line, sig_line, hist    = _macd(close)
        stoch_k, stoch_d             = _stochastic(high, low, close)

        rsi_now    = _safe_float(rsi_series.iloc[-1])
        macd_now   = _safe_float(macd_line.iloc[-1])
        macd_sig   = _safe_float(sig_line.iloc[-1])
        macd_hist  = _safe_float(hist.iloc[-1])
        stoch_k_now = _safe_float(stoch_k.iloc[-1])
        stoch_d_now = _safe_float(stoch_d.iloc[-1])

        # MACD crossover in last 3 sessions
        macd_cross = "none"
        hist_tail  = hist.dropna().iloc[-3:]
        if len(hist_tail) >= 2:
            if hist_tail.iloc[-2] < 0 and hist_tail.iloc[-1] > 0:
                macd_cross = "bullish_crossover"
            elif hist_tail.iloc[-2] > 0 and hist_tail.iloc[-1] < 0:
                macd_cross = "bearish_crossover"

        rsi_signal = (
            "overbought" if rsi_now and rsi_now > 70
            else "oversold"   if rsi_now and rsi_now < 30
            else "neutral"
        )
        momentum_signal = (
            "bullish" if (macd_now and macd_sig and macd_now > macd_sig and rsi_signal != "overbought")
            else "bearish" if (macd_now and macd_sig and macd_now < macd_sig)
            else "neutral"
        )

        # ── Volatility ─────────────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = _bollinger(close)
        atr_series                 = _atr(high, low, close)

        bb_upper_now = _safe_float(bb_upper.iloc[-1])
        bb_mid_now   = _safe_float(bb_mid.iloc[-1])
        bb_lower_now = _safe_float(bb_lower.iloc[-1])
        atr_now      = _safe_float(atr_series.iloc[-1])

        # %B: where price sits within the bands (0 = lower band, 1 = upper)
        bb_pct_b = (
            _safe_float((price_now - bb_lower_now) / (bb_upper_now - bb_lower_now))
            if bb_upper_now and bb_lower_now and bb_upper_now != bb_lower_now
            else None
        )
        bb_width = (
            _safe_float((bb_upper_now - bb_lower_now) / bb_mid_now * 100)
            if bb_mid_now and bb_mid_now != 0
            else None
        )
        bb_signal = (
            "squeeze"     if bb_width and bb_width < 5
            else "breakout_upper" if bb_pct_b and bb_pct_b > 1
            else "breakout_lower" if bb_pct_b and bb_pct_b < 0
            else "normal"
        )

        # ATR as % of price (normalised volatility)
        atr_pct = _safe_float(atr_now / price_now * 100) if atr_now and price_now else None

        # ── Volume ─────────────────────────────────────────────────────────
        obv_series  = _obv(close, volume)
        vwap_series = _vwap(high, low, close, volume)

        obv_now  = _safe_float(obv_series.iloc[-1])
        vwap_now = _safe_float(vwap_series.iloc[-1])

        vol_avg_20  = _safe_float(volume.rolling(20).mean().iloc[-1])
        vol_ratio   = _safe_float(volume.iloc[-1] / vol_avg_20) if vol_avg_20 else None
        above_vwap  = bool(price_now > vwap_now) if vwap_now else None

        # OBV trend: compare last 10 sessions slope
        obv_trend = "flat"
        obv_tail  = obv_series.dropna().iloc[-10:]
        if len(obv_tail) >= 10:
            slope = float(np.polyfit(range(len(obv_tail)), obv_tail.values, 1)[0])
            obv_trend = "rising" if slope > 0 else "falling"

        volume_signal = (
            "high_volume_move"   if vol_ratio and vol_ratio > 1.5
            else "low_volume"    if vol_ratio and vol_ratio < 0.7
            else "normal"
        )

        # ── Overall composite signal ────────────────────────────────────────
        bullish_count = sum([
            trend_signal    in ("bullish", "strongly bullish"),
            momentum_signal == "bullish",
            rsi_signal      == "neutral",   # not overbought = ok to enter
            obv_trend       == "rising",
            above_vwap      is True,
        ])
        bearish_count = sum([
            trend_signal    == "bearish",
            momentum_signal == "bearish",
            rsi_signal      == "overbought",
            obv_trend       == "falling",
            above_vwap      is False,
        ])
        composite = (
            "STRONG BUY"  if bullish_count >= 4
            else "BUY"    if bullish_count == 3
            else "SELL"   if bearish_count >= 4
            else "SHORT"  if bearish_count == 3
            else "NEUTRAL"
        )

        result = {
            "symbol":  symbol.upper(),
            "period":  period,
            "current_price": _safe_float(price_now),

            # ── Trend ──
            "trend": {
                "sma_20":          _safe_float(sma20_now),
                "sma_50":          _safe_float(sma50_now),
                "sma_200":         _safe_float(sma200_now),
                "ema_9":           _safe_float(ema9_now),
                "ema_21":          _safe_float(ema21_now),
                "above_sma20":     above_sma20,
                "above_sma50":     above_sma50,
                "above_sma200":    above_sma200,
                "mas_aligned_bullish": mas_aligned_bullish,
                "cross_signal":    cross_signal,   # golden_cross / death_cross / none
                "signal":          trend_signal,
            },

            # ── Momentum ──
            "momentum": {
                "rsi_14":          rsi_now,
                "rsi_signal":      rsi_signal,        # overbought / oversold / neutral
                "macd_line":       macd_now,
                "macd_signal":     macd_sig,
                "macd_histogram":  macd_hist,
                "macd_cross":      macd_cross,         # bullish_crossover / bearish_crossover / none
                "stoch_k":         stoch_k_now,
                "stoch_d":         stoch_d_now,
                "signal":          momentum_signal,
            },

            # ── Volatility ──
            "volatility": {
                "bb_upper":        bb_upper_now,
                "bb_middle":       bb_mid_now,
                "bb_lower":        bb_lower_now,
                "bb_pct_b":        bb_pct_b,          # 0–1 range; >1 above upper, <0 below lower
                "bb_width_pct":    bb_width,
                "atr_14":          atr_now,
                "atr_pct_of_price": atr_pct,
                "signal":          bb_signal,          # squeeze / breakout_upper / breakout_lower / normal
            },

            # ── Volume ──
            "volume": {
                "obv":             obv_now,
                "obv_trend":       obv_trend,          # rising / falling / flat
                "vwap":            vwap_now,
                "above_vwap":      above_vwap,
                "volume_ratio_20d": vol_ratio,         # today vs 20-day avg
                "signal":          volume_signal,
            },

            # ── Composite ──
            "composite_signal":  composite,            # STRONG BUY / BUY / NEUTRAL / SELL / SHORT
            "bullish_factors":   bullish_count,
            "bearish_factors":   bearish_count,
        }

        log.info("tool.fetch_indicators.done", symbol=symbol, composite=composite)
        return result

    except Exception as exc:
        log.error("tool.fetch_indicators.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}


@tool("fetch_support_resistance", args_schema=SupportResistanceInput)
async def fetch_support_resistance(symbol: str, period: str = "6mo") -> dict:
    """
    Identify key support and resistance levels plus chart pattern signals.

    Detects:
      - Pivot-based S/R levels (classic floor-trader pivots)
      - Swing highs and lows over the lookback period
      - Proximity of current price to nearest S/R level
      - Candlestick pattern signals (doji, hammer, engulfing) for last session
      - Trend channel: is price making higher highs + higher lows (uptrend)?

    Use this alongside fetch_indicators for entry/exit zone analysis.
    """
    log.info("tool.fetch_support_resistance", symbol=symbol, period=period)
    try:
        df = await asyncio.to_thread(_download_ohlcv, symbol, period)

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        _open  = df["Open"]

        price_now = float(close.iloc[-1])

        # ── Classic floor-trader pivot points (based on last full session) ──
        h = float(high.iloc[-2])
        l = float(low.iloc[-2])
        c = float(close.iloc[-2])

        pivot = (h + l + c) / 3
        r1    = 2 * pivot - l
        r2    = pivot + (h - l)
        r3    = h + 2 * (pivot - l)
        s1    = 2 * pivot - h
        s2    = pivot - (h - l)
        s3    = l - 2 * (h - pivot)

        # ── Swing highs / lows (local extrema with ±5 bar window) ─────────
        window = 5
        swing_highs, swing_lows = [], []

        for i in range(window, len(df) - window):
            if high.iloc[i] == high.iloc[i - window:i + window + 1].max():
                swing_highs.append({
                    "date":  str(df.index[i].date()),
                    "price": _safe_float(high.iloc[i]),
                })
            if low.iloc[i] == low.iloc[i - window:i + window + 1].min():
                swing_lows.append({
                    "date":  str(df.index[i].date()),
                    "price": _safe_float(low.iloc[i]),
                })

        # Keep the 5 most recent swing points
        swing_highs = swing_highs[-5:]
        swing_lows  = swing_lows[-5:]

        # ── Nearest S/R to current price ──────────────────────────────────
        all_levels = [
            ("pivot", pivot), ("R1", r1), ("R2", r2), ("S1", s1), ("S2", s2),
            *[("swing_high", sh["price"]) for sh in swing_highs if sh["price"]],
            *[("swing_low",  sl["price"]) for sl in swing_lows  if sl["price"]],
        ]
        resistance_levels = [(n, p) for n, p in all_levels if p and p > price_now]
        support_levels    = [(n, p) for n, p in all_levels if p and p < price_now]

        nearest_resistance = min(resistance_levels, key=lambda x: x[1] - price_now) if resistance_levels else None
        nearest_support    = max(support_levels,    key=lambda x: price_now - x[1]) if support_levels    else None

        dist_to_resistance = _safe_float(
            (nearest_resistance[1] - price_now) / price_now * 100
        ) if nearest_resistance else None
        dist_to_support = _safe_float(
            (price_now - nearest_support[1]) / price_now * 100
        ) if nearest_support else None

        # ── Candlestick pattern (last 3 sessions) ─────────────────────────
        patterns = []
        if len(df) >= 3:
            o2, h2, l2, c2 = float(_open.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
            o1, h1, l1, c1 = float(_open.iloc[-2]), float(high.iloc[-2]), float(low.iloc[-2]), float(close.iloc[-2])
            body   = abs(c2 - o2)
            candle = h2 - l2

            # Doji: body < 10% of total candle range
            if candle > 0 and body / candle < 0.1:
                patterns.append("doji — indecision, potential reversal")

            # Hammer / hanging man: small body near top, long lower wick
            lower_wick = min(o2, c2) - l2
            upper_wick = h2 - max(o2, c2)
            if candle > 0 and lower_wick > 2 * body and upper_wick < body:
                patterns.append("hammer — bullish reversal signal" if c2 < c1 else "hanging man — bearish reversal signal")

            # Bullish engulfing
            if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
                patterns.append("bullish engulfing — strong buy signal")

            # Bearish engulfing
            if c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
                patterns.append("bearish engulfing — strong sell signal")

            # Shooting star: small body near bottom, long upper wick
            if candle > 0 and upper_wick > 2 * body and lower_wick < body and c2 < o2:
                patterns.append("shooting star — bearish reversal signal")

        # ── Trend structure (higher highs + higher lows = uptrend) ────────
        trend_structure = "undefined"
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1]["price"] > swing_highs[-2]["price"]
            hl = swing_lows[-1]["price"]  > swing_lows[-2]["price"]
            lh = swing_highs[-1]["price"] < swing_highs[-2]["price"]
            ll = swing_lows[-1]["price"]  < swing_lows[-2]["price"]
            trend_structure = (
                "uptrend"   if hh and hl
                else "downtrend" if lh and ll
                else "ranging"
            )

        result = {
            "symbol":       symbol.upper(),
            "current_price": _safe_float(price_now),

            # ── Pivot levels ──
            "pivot_levels": {
                "pivot": _safe_float(pivot),
                "R1":    _safe_float(r1),
                "R2":    _safe_float(r2),
                "R3":    _safe_float(r3),
                "S1":    _safe_float(s1),
                "S2":    _safe_float(s2),
                "S3":    _safe_float(s3),
            },

            # ── Swing points ──
            "swing_highs": swing_highs,
            "swing_lows":  swing_lows,

            # ── Nearest levels ──
            "nearest_resistance": {
                "level": nearest_resistance[0] if nearest_resistance else None,
                "price": _safe_float(nearest_resistance[1]) if nearest_resistance else None,
                "distance_pct": dist_to_resistance,
            },
            "nearest_support": {
                "level": nearest_support[0] if nearest_support else None,
                "price": _safe_float(nearest_support[1]) if nearest_support else None,
                "distance_pct": dist_to_support,
            },

            # ── Patterns & structure ──
            "candlestick_patterns": patterns if patterns else ["no notable pattern"],
            "trend_structure":      trend_structure,   # uptrend / downtrend / ranging / undefined
        }

        log.info("tool.fetch_support_resistance.done",
                 symbol=symbol, trend=trend_structure, patterns=len(patterns))
        return result

    except Exception as exc:
        log.error("tool.fetch_support_resistance.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}