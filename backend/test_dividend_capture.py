"""
test_dividend_capture.py
-------------------------
Test suite for Dividend Capture Agent.
Demonstrates calendar lookup, profit analysis, and opportunity screening.
"""

import asyncio
from agents.dividend_capture import dividend_capture_agent


async def test_calendar():
    """Test Case 1: Upcoming dividend calendar."""
    print("\n" + "="*80)
    print("TEST 1: DIVIDEND CALENDAR")
    print("="*80 + "\n")

    request = "Show me upcoming ex-dividend dates for: KO, T, VZ, PG, JNJ"

    result = await dividend_capture_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    response = result["messages"][-1].content
    if isinstance(response, list):
        response = "\n".join(b["text"] for b in response if b.get("type") == "text")

    print(response)
    print("\n" + "="*80 + "\n")


async def test_profit_analysis():
    """Test Case 2: Profit analysis for a specific trade."""
    print("\n" + "="*80)
    print("TEST 2: DIVIDEND CAPTURE PROFIT ANALYSIS")
    print("="*80 + "\n")

    request = """
    Analyze dividend capture trade:
    - Symbol: KO (Coca-Cola)
    - Entry price: $71.23
    - Shares: 100
    - Dividend: $0.485 per share
    - Tax rate: 22%
    
    Is this trade profitable?
    """

    result = await dividend_capture_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    response = result["messages"][-1].content
    if isinstance(response, list):
        response = "\n".join(b["text"] for b in response if b.get("type") == "text")

    print(response)
    print("\n" + "="*80 + "\n")


async def test_screening():
    """Test Case 3: Screen for best dividend capture opportunities."""
    print("\n" + "="*80)
    print("TEST 3: SCREEN FOR BEST DIVIDEND OPPORTUNITIES")
    print("="*80 + "\n")

    request = """
    Screen for best dividend capture opportunities in S&P 500.
    Min yield: 2.0%
    Max volatility: 2.5%
    Next 14 days
    """

    result = await dividend_capture_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    response = result["messages"][-1].content
    if isinstance(response, list):
        response = "\n".join(b["text"] for b in response if b.get("type") == "text")

    print(response)
    print("\n" + "="*80 + "\n")


async def test_high_yield_vs_low_yield():
    """Test Case 4: Compare high-yield vs low-yield dividend capture."""
    print("\n" + "="*80)
    print("TEST 4: HIGH-YIELD VS LOW-YIELD COMPARISON")
    print("="*80 + "\n")

    # High-yield telecom (T = AT&T, 5%+ yield)
    request_high = """
    Analyze dividend capture trade:
    - Symbol: T (AT&T)
    - Entry price: $18.45
    - Shares: 500
    - Dividend: $0.28 per share (5.2% yield)
    """

    # Low-yield blue chip (AAPL = Apple, <1% yield)
    request_low = """
    Analyze dividend capture trade:
    - Symbol: AAPL (Apple)
    - Entry price: $225.40
    - Shares: 50
    - Dividend: $0.24 per share (0.4% yield)
    """

    print("HIGH-YIELD TRADE (AT&T):")
    result1 = await dividend_capture_agent.ainvoke({
        "messages": [{"role": "user", "content": request_high}]
    })
    resp1 = result1["messages"][-1].content
    if isinstance(resp1, list):
        resp1 = "\n".join(b["text"] for b in resp1 if b.get("type") == "text")
    print(resp1)

    print("\n" + "-"*80 + "\n")

    print("LOW-YIELD TRADE (AAPL):")
    result2 = await dividend_capture_agent.ainvoke({
        "messages": [{"role": "user", "content": request_low}]
    })
    resp2 = result2["messages"][-1].content
    if isinstance(resp2, list):
        resp2 = "\n".join(b["text"] for b in resp2 if b.get("type") == "text")
    print(resp2)

    print("\n" + "="*80 + "\n")


async def main():
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         DIVIDEND CAPTURE AGENT - TEST SUITE                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # Run all tests
    # await test_calendar()
    # await test_profit_analysis()
    await test_screening()
    # await test_high_yield_vs_low_yield()

    print("\n✅ All tests completed!\n")


if __name__ == "__main__":
    asyncio.run(main())