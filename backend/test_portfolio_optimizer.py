"""
test_portfolio_optimizer.py
----------------------------
Example usage of the Portfolio Optimizer Agent.
Demonstrates both NEW portfolio optimization and REBALANCING workflows.
"""

import asyncio
from agents.portfolio_optimizer import portfolio_optimizer_agent


async def test_new_portfolio():
    """Test Case 1: Optimize a NEW portfolio from scratch."""
    print("\n" + "="*80)
    print("TEST 1: NEW PORTFOLIO OPTIMIZATION")
    print("="*80 + "\n")

    request = "Optimize portfolio for: AAPL, MSFT, GOOGL, NVDA, AMZN. Method: max_sharpe"

    result = await portfolio_optimizer_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    response = result["messages"][-1].content
    if isinstance(response, list):
        response = "\n".join(b["text"] for b in response if b.get("type") == "text")

    print(response)
    print("\n" + "="*80 + "\n")


async def test_rebalance_portfolio():
    """Test Case 2: REBALANCE an existing portfolio."""
    print("\n" + "="*80)
    print("TEST 2: PORTFOLIO REBALANCING")
    print("="*80 + "\n")

    request = """
    Rebalance my portfolio.
    Current holdings: AAPL:20, MSFT:15, GOOGL:10
    Target tickers: AAPL, MSFT, GOOGL, NVDA
    Portfolio value: $50,000
    Method: max_sharpe
    """

    result = await portfolio_optimizer_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    response = result["messages"][-1].content
    if isinstance(response, list):
        response = "\n".join(b["text"] for b in response if b.get("type") == "text")

    print(response)
    print("\n" + "="*80 + "\n")


async def test_min_volatility():
    """Test Case 3: Minimum volatility portfolio (conservative investor)."""
    print("\n" + "="*80)
    print("TEST 3: MINIMUM VOLATILITY PORTFOLIO")
    print("="*80 + "\n")

    request = "Optimize portfolio for: JNJ, PG, KO, WMT, VZ. Method: min_volatility"

    result = await portfolio_optimizer_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    response = result["messages"][-1].content
    if isinstance(response, list):
        response = "\n".join(b["text"] for b in response if b.get("type") == "text")

    print(response)
    print("\n" + "="*80 + "\n")


async def main():
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         PORTFOLIO OPTIMIZER AGENT - TEST SUITE                    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # Run all tests
    await test_new_portfolio()
    await test_rebalance_portfolio()
    await test_min_volatility()

    print("\n✅ All tests completed!\n")


if __name__ == "__main__":
    asyncio.run(main())