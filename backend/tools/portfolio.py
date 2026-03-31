"""
tools/portfolio.py
------------------
Portfolio optimization tools using Modern Portfolio Theory (MPT).

Three async tools exposed:
  - optimize_portfolio       → efficient frontier optimization (max Sharpe, min volatility, target return)
  - analyze_current_portfolio→ risk metrics for existing holdings
  - generate_rebalancing_plan→ buy/sell orders to reach target allocation
"""

from __future__ import annotations

import asyncio
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

class OptimizePortfolioInput(BaseModel):
    symbols: list[str] = Field(
        description="List of stock tickers to optimize, e.g. ['AAPL', 'MSFT', 'GOOGL']."
    )
    optimization_method: Literal["max_sharpe", "min_volatility", "efficient_return"] = Field(
        default="max_sharpe",
        description=(
            "Optimization objective: 'max_sharpe' for best risk-adjusted return, "
            "'min_volatility' for lowest risk, 'efficient_return' for target return."
        ),
    )
    target_return: float | None = Field(
        default=None,
        description="Target annual return (e.g. 0.15 for 15%) — required if method is 'efficient_return'.",
    )
    lookback_period: str = Field(
        default="2y",
        description="Historical data period for estimating returns/risk: '1y', '2y', '3y', '5y'.",
    )


class AnalyzePortfolioInput(BaseModel):
    holdings: dict[str, float] = Field(
        description=(
            "Current portfolio holdings as {symbol: weight}, e.g. "
            "{'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.3}. Weights must sum to 1.0."
        )
    )
    lookback_period: str = Field(
        default="2y",
        description="Historical data period for risk calculation.",
    )


class RebalancingPlanInput(BaseModel):
    current_holdings: dict[str, float] = Field(
        description="Current holdings: {symbol: number_of_shares}, e.g. {'AAPL': 10, 'MSFT': 5}."
    )
    target_weights: dict[str, float] = Field(
        description=(
            "Target allocation from optimizer: {symbol: weight}, e.g. "
            "{'AAPL': 0.25, 'MSFT': 0.35, 'GOOGL': 0.40}."
        )
    )
    total_portfolio_value: float = Field(
        description="Total portfolio value in USD, e.g. 100000."
    )


# ── Shared data fetcher ───────────────────────────────────────────────────────

@_retry
def _download_prices(symbols: list[str], period: str) -> pd.DataFrame:
    """Download adjusted close prices for a list of symbols."""
    data = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]
    if data is None or data.empty:
        raise ValueError(f"No price data for symbols: {symbols}")
    # If single symbol, data is Series → convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0])
    data = data.dropna()  # Remove days with missing data
    if len(data) < 30:
        raise ValueError(f"Insufficient data ({len(data)} days) for portfolio optimization.")
    return data


def _safe_float(val, decimals=4) -> float | None:
    """Convert to float, None if NaN/inf."""
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, decimals)
    except (TypeError, ValueError):
        return None


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("optimize_portfolio", args_schema=OptimizePortfolioInput)
async def optimize_portfolio(
    symbols: list[str],
    optimization_method: Literal["max_sharpe", "min_volatility", "efficient_return"] = "max_sharpe",
    target_return: float | None = None,
    lookback_period: str = "2y",
) -> dict:
    """
    Optimize a portfolio using Modern Portfolio Theory (Markowitz efficient frontier).

    Returns optimal weights, expected performance metrics (return, volatility, Sharpe ratio),
    and the efficient frontier data for visualization.

    Use this to find the best allocation for a set of stocks based on historical data.
    """
    log.info("tool.optimize_portfolio", symbols=symbols, method=optimization_method)

    try:
        # Download price data
        prices = await asyncio.to_thread(_download_prices, symbols, lookback_period)

        # Calculate daily returns
        returns = prices.pct_change().dropna()

        # ── Calculate expected returns and covariance ──
        # Mean historical return (annualized)
        mu = returns.mean() * 252  # 252 trading days/year
        # Sample covariance (annualized)
        cov_matrix = returns.cov() * 252

        # ── Optimization using scipy (no external PyPortfolioOpt dependency) ──
        from scipy.optimize import minimize

        n_assets = len(symbols)
        init_guess = np.ones(n_assets) / n_assets

        def portfolio_return(weights):
            return np.dot(weights, mu)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def neg_sharpe(weights, risk_free_rate=0.02):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - risk_free_rate) / vol

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        # Bounds: no short selling (0 <= weight <= 1)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # ── Optimize based on method ──
        if optimization_method == "max_sharpe":
            result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
        elif optimization_method == "min_volatility":
            result = minimize(portfolio_volatility, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
        elif optimization_method == "efficient_return":
            if target_return is None:
                raise ValueError("target_return must be provided for efficient_return optimization.")
            # Add constraint: portfolio return = target_return
            target_constraint = {"type": "eq", "fun": lambda w: portfolio_return(w) - target_return}
            all_constraints = [constraints, target_constraint]
            result = minimize(portfolio_volatility, init_guess, method="SLSQP", bounds=bounds, constraints=all_constraints)
        else:
            raise ValueError(f"Invalid optimization_method: {optimization_method}")

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        optimal_weights = result.x
        optimal_return = portfolio_return(optimal_weights)
        optimal_volatility = portfolio_volatility(optimal_weights)
        optimal_sharpe = (optimal_return - 0.02) / optimal_volatility

        # ── Generate efficient frontier (sample 50 portfolios) ──
        target_returns = np.linspace(mu.min(), mu.max(), 50)
        frontier_volatilities = []

        for target_ret in target_returns:
            try:
                target_constraint = {"type": "eq", "fun": lambda w: portfolio_return(w) - target_ret}
                res = minimize(
                    portfolio_volatility, init_guess, method="SLSQP",
                    bounds=bounds, constraints=[constraints, target_constraint],
                    options={"maxiter": 200}
                )
                if res.success:
                    frontier_volatilities.append(portfolio_volatility(res.x))
                else:
                    frontier_volatilities.append(None)
            except Exception:
                frontier_volatilities.append(None)

        # Clean frontier data
        frontier = [
            {"return": _safe_float(r), "volatility": _safe_float(v)}
            for r, v in zip(target_returns, frontier_volatilities)
            if v is not None
        ]

        # ── Build response ──
        weights_dict = {
            sym: _safe_float(w, 4)
            for sym, w in zip(symbols, optimal_weights)
            if w > 0.01  # Filter out tiny allocations
        }

        output = {
            "optimization_method": optimization_method,
            "lookback_period": lookback_period,
            "optimal_weights": weights_dict,
            "expected_annual_return": _safe_float(optimal_return),
            "expected_volatility": _safe_float(optimal_volatility),
            "sharpe_ratio": _safe_float(optimal_sharpe),
            "efficient_frontier": frontier[:20],  # First 20 points for brevity
            "data_points": len(returns),
        }

        log.info("tool.optimize_portfolio.done", symbols=symbols, sharpe=_safe_float(optimal_sharpe))
        return output

    except Exception as exc:
        log.error("tool.optimize_portfolio.error", symbols=symbols, error=str(exc))
        return {"error": str(exc)}


@tool("analyze_current_portfolio", args_schema=AnalyzePortfolioInput)
async def analyze_current_portfolio(
    holdings: dict[str, float],
    lookback_period: str = "2y",
) -> dict:
    """
    Analyze risk and return metrics for a current portfolio allocation.

    Takes portfolio weights, calculates expected return, volatility, Sharpe ratio,
    and correlation matrix. Use this to evaluate how well your current portfolio
    is balanced before rebalancing.
    """
    log.info("tool.analyze_current_portfolio", holdings=list(holdings.keys()))

    try:
        # Validate weights sum to 1
        total_weight = sum(holdings.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight:.4f}")

        symbols = list(holdings.keys())
        weights = np.array([holdings[s] for s in symbols])

        # Download price data
        prices = await asyncio.to_thread(_download_prices, symbols, lookback_period)
        returns = prices.pct_change().dropna()

        # Calculate metrics
        mu = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        portfolio_return = np.dot(weights, mu)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility

        # Correlation matrix
        corr_matrix = returns.corr()

        # Individual asset metrics
        asset_metrics = []
        for sym in symbols:
            asset_metrics.append({
                "symbol": sym,
                "weight": _safe_float(holdings[sym]),
                "expected_return": _safe_float(mu[sym]),
                "volatility": _safe_float(np.sqrt(cov_matrix.loc[sym, sym])),
                "contribution_to_return": _safe_float(holdings[sym] * mu[sym]),
            })

        # Diversification ratio (1 = perfectly diversified, lower = concentrated)
        weighted_volatilities = sum(holdings[s] * np.sqrt(cov_matrix.loc[s, s]) for s in symbols)
        diversification_ratio = weighted_volatilities / portfolio_volatility

        output = {
            "portfolio_return": _safe_float(portfolio_return),
            "portfolio_volatility": _safe_float(portfolio_volatility),
            "sharpe_ratio": _safe_float(sharpe_ratio),
            "diversification_ratio": _safe_float(diversification_ratio),
            "asset_metrics": asset_metrics,
            "correlation_matrix": {
                sym1: {sym2: _safe_float(corr_matrix.loc[sym1, sym2]) for sym2 in symbols}
                for sym1 in symbols
            },
            "lookback_period": lookback_period,
            "data_points": len(returns),
        }

        log.info("tool.analyze_current_portfolio.done", sharpe=_safe_float(sharpe_ratio))
        return output

    except Exception as exc:
        log.error("tool.analyze_current_portfolio.error", error=str(exc))
        return {"error": str(exc)}


@tool("generate_rebalancing_plan", args_schema=RebalancingPlanInput)
async def generate_rebalancing_plan(
    current_holdings: dict[str, float],
    target_weights: dict[str, float],
    total_portfolio_value: float,
) -> dict:
    """
    Generate actionable buy/sell orders to rebalance from current holdings to target allocation.

    Returns exact number of shares to buy/sell for each position, transaction costs estimate,
    and net cash flow required. This is the final step after optimization — it produces
    executable trades.
    """
    log.info("tool.generate_rebalancing_plan", holdings=list(current_holdings.keys()))

    try:
        # Validate target weights
        if not (0.99 <= sum(target_weights.values()) <= 1.01):
            raise ValueError(f"Target weights must sum to 1.0, got {sum(target_weights.values()):.4f}")

        # Get current prices
        all_symbols = list(set(current_holdings.keys()) | set(target_weights.keys()))

        @_retry
        def _get_prices():
            prices = {}
            for sym in all_symbols:
                ticker = yf.Ticker(sym)
                info = ticker.info
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                if not price:
                    raise ValueError(f"Could not fetch current price for {sym}")
                prices[sym] = float(price)
            return prices

        current_prices = await asyncio.to_thread(_get_prices)

        # Calculate current value per position
        current_values = {
            sym: shares * current_prices[sym]
            for sym, shares in current_holdings.items()
        }
        actual_portfolio_value = sum(current_values.values())

        # Target values
        target_values = {
            sym: weight * total_portfolio_value
            for sym, weight in target_weights.items()
        }

        # Generate trades
        trades = []
        total_buys = 0
        total_sells = 0

        for sym in all_symbols:
            current_val = current_values.get(sym, 0)
            target_val = target_values.get(sym, 0)
            diff_val = target_val - current_val

            current_shares = current_holdings.get(sym, 0)
            target_shares = int(target_val / current_prices[sym])
            diff_shares = target_shares - int(current_shares)

            if abs(diff_shares) > 0:
                action = "BUY" if diff_shares > 0 else "SELL"
                trades.append({
                    "symbol": sym,
                    "action": action,
                    "shares": abs(diff_shares),
                    "price": _safe_float(current_prices[sym], 2),
                    "value": _safe_float(abs(diff_shares * current_prices[sym]), 2),
                    "current_weight": _safe_float(current_val / actual_portfolio_value if actual_portfolio_value else 0),
                    "target_weight": _safe_float(target_weights.get(sym, 0)),
                })

                if action == "BUY":
                    total_buys += abs(diff_shares * current_prices[sym])
                else:
                    total_sells += abs(diff_shares * current_prices[sym])

        # Estimate transaction costs (0.1% per trade)
        transaction_costs = (total_buys + total_sells) * 0.001

        # Cash required (buys - sells + costs)
        net_cash_flow = total_buys - total_sells + transaction_costs

        output = {
            "trades": sorted(trades, key=lambda x: x["value"], reverse=True),
            "summary": {
                "total_trades": len(trades),
                "total_buy_value": _safe_float(total_buys, 2),
                "total_sell_value": _safe_float(total_sells, 2),
                "estimated_transaction_costs": _safe_float(transaction_costs, 2),
                "net_cash_required": _safe_float(net_cash_flow, 2),
                "current_portfolio_value": _safe_float(actual_portfolio_value, 2),
                "target_portfolio_value": _safe_float(total_portfolio_value, 2),
            },
        }

        log.info("tool.generate_rebalancing_plan.done", trades=len(trades))
        return output

    except Exception as exc:
        log.error("tool.generate_rebalancing_plan.error", error=str(exc))
        return {"error": str(exc)}