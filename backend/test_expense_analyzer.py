"""
test_expense_analyzer.py
-------------------------
Test suite for the Expense Categorization & Budget Agent.
Mirrors the pattern from test_portfolio_optimizer.py.

Run:
    python backend/test_expense_analyzer.py
"""

import asyncio
from agents.expense_analysis import expense_analyzer_agent


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if b.get("type") == "text")
    return content


# ── Shared sample data ────────────────────────────────────────────────────────

SAMPLE_TRANSACTIONS = [
    # Income
    {"date": "2024-03-01", "description": "Direct Deposit - Payroll", "amount": -4500.00, "account": "Checking"},
    {"date": "2024-03-15", "description": "Freelance Payment - Client ABC", "amount": -800.00, "account": "Checking"},

    # Housing
    {"date": "2024-03-01", "description": "Rent - Apartment 4B", "amount": 1400.00, "account": "Checking"},
    {"date": "2024-03-05", "description": "Comcast Internet Bill", "amount": 79.99, "account": "Checking"},
    {"date": "2024-03-10", "description": "Electricity Bill - Con Edison", "amount": 112.50, "account": "Checking"},

    # Food & Dining
    {"date": "2024-03-02", "description": "Whole Foods Market", "amount": 143.22, "account": "Credit Card"},
    {"date": "2024-03-04", "description": "Starbucks Coffee", "amount": 8.75, "account": "Credit Card"},
    {"date": "2024-03-06", "description": "DoorDash - Thai Palace", "amount": 34.50, "account": "Credit Card"},
    {"date": "2024-03-08", "description": "Trader Joe's", "amount": 87.30, "account": "Credit Card"},
    {"date": "2024-03-09", "description": "Starbucks Coffee", "amount": 7.25, "account": "Credit Card"},
    {"date": "2024-03-11", "description": "McDonald's", "amount": 12.40, "account": "Credit Card"},
    {"date": "2024-03-13", "description": "Chipotle Mexican Grill", "amount": 18.90, "account": "Credit Card"},
    {"date": "2024-03-14", "description": "DoorDash - Pizza Palace", "amount": 28.75, "account": "Credit Card"},
    {"date": "2024-03-16", "description": "Whole Foods Market", "amount": 165.80, "account": "Credit Card"},
    {"date": "2024-03-18", "description": "Uber Eats - Sushi Place", "amount": 52.30, "account": "Credit Card"},
    {"date": "2024-03-20", "description": "Restaurant Eleven Madison Park", "amount": 310.00, "account": "Credit Card"},
    {"date": "2024-03-22", "description": "Starbucks Coffee", "amount": 6.90, "account": "Credit Card"},
    {"date": "2024-03-25", "description": "Kroger Supermarket", "amount": 95.40, "account": "Credit Card"},
    {"date": "2024-03-28", "description": "DoorDash - Burger Joint", "amount": 22.60, "account": "Credit Card"},

    # Transport
    {"date": "2024-03-03", "description": "Uber Trip", "amount": 18.50, "account": "Credit Card"},
    {"date": "2024-03-07", "description": "Shell Gas Station", "amount": 65.00, "account": "Credit Card"},
    {"date": "2024-03-12", "description": "Lyft Ride", "amount": 24.30, "account": "Credit Card"},
    {"date": "2024-03-19", "description": "Parking - Downtown Garage", "amount": 35.00, "account": "Credit Card"},
    {"date": "2024-03-24", "description": "Uber Trip", "amount": 22.10, "account": "Credit Card"},
    {"date": "2024-03-27", "description": "Shell Gas Station", "amount": 58.00, "account": "Credit Card"},

    # Subscriptions & Entertainment
    {"date": "2024-03-01", "description": "Netflix Monthly", "amount": 15.49, "account": "Credit Card"},
    {"date": "2024-03-01", "description": "Spotify Premium", "amount": 10.99, "account": "Credit Card"},
    {"date": "2024-03-01", "description": "Disney+ Subscription", "amount": 13.99, "account": "Credit Card"},
    {"date": "2024-03-01", "description": "ChatGPT Plus - OpenAI", "amount": 20.00, "account": "Credit Card"},
    {"date": "2024-03-01", "description": "Adobe Creative Cloud", "amount": 54.99, "account": "Credit Card"},
    {"date": "2024-03-15", "description": "AMC Theatre - Movie Night", "amount": 42.00, "account": "Credit Card"},
    {"date": "2024-03-22", "description": "Steam - Game Purchase", "amount": 59.99, "account": "Credit Card"},

    # Shopping
    {"date": "2024-03-06", "description": "Amazon.com Order", "amount": 89.99, "account": "Credit Card"},
    {"date": "2024-03-10", "description": "Amazon.com Order", "amount": 234.50, "account": "Credit Card"},
    {"date": "2024-03-17", "description": "Nike Store Online", "amount": 145.00, "account": "Credit Card"},
    {"date": "2024-03-23", "description": "IKEA Home Furnishings", "amount": 387.00, "account": "Credit Card"},

    # Health
    {"date": "2024-03-05", "description": "Planet Fitness Membership", "amount": 24.99, "account": "Credit Card"},
    {"date": "2024-03-14", "description": "CVS Pharmacy", "amount": 47.30, "account": "Credit Card"},
    {"date": "2024-03-20", "description": "Doctor Visit Copay", "amount": 40.00, "account": "Checking"},

    # Finance
    {"date": "2024-03-15", "description": "Student Loan Payment", "amount": 280.00, "account": "Checking"},
    {"date": "2024-03-28", "description": "Credit Card Payment", "amount": 800.00, "account": "Checking"},
]


async def test_standard_analysis():
    """Test Case 1: Standard monthly expense analysis."""
    print("\n" + "="*80)
    print("TEST 1: STANDARD MONTHLY EXPENSE ANALYSIS")
    print("="*80 + "\n")

    request = f"""
Analyze my March 2024 spending and tell me where I'm overspending.
Transactions: {SAMPLE_TRANSACTIONS}
Month: 2024-03
"""

    result = await expense_analyzer_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    print(_extract_text(result["messages"][-1].content))
    print("\n" + "="*80 + "\n")


async def test_with_custom_budget():
    """Test Case 2: Analysis with user-defined budget limits."""
    print("\n" + "="*80)
    print("TEST 2: CUSTOM BUDGET LIMITS")
    print("="*80 + "\n")

    custom_budget = {
        "Food & Dining":  450.0,
        "Shopping":       200.0,
        "Entertainment":  100.0,
        "Transport":      250.0,
        "Subscriptions":   60.0,
    }

    request = f"""
Analyze my spending with these budget limits: {custom_budget}
Transactions: {SAMPLE_TRANSACTIONS}
Month: 2024-03
"""

    result = await expense_analyzer_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    print(_extract_text(result["messages"][-1].content))
    print("\n" + "="*80 + "\n")


async def test_minimal_transactions():
    """Test Case 3: Minimal data — tests graceful handling."""
    print("\n" + "="*80)
    print("TEST 3: MINIMAL TRANSACTION SET")
    print("="*80 + "\n")

    minimal_txns = [
        {"date": "2024-03-01", "description": "Salary Deposit", "amount": -3000.0},
        {"date": "2024-03-05", "description": "Rent Payment", "amount": 1200.0},
        {"date": "2024-03-10", "description": "Walmart Grocery", "amount": 180.0},
        {"date": "2024-03-15", "description": "Netflix", "amount": 15.49},
        {"date": "2024-03-20", "description": "Uber Trip", "amount": 22.00},
        {"date": "2024-03-25", "description": "Amazon Order", "amount": 67.80},
    ]

    request = f"""
Quick budget check for my March transactions.
Transactions: {minimal_txns}
"""

    result = await expense_analyzer_agent.ainvoke({
        "messages": [{"role": "user", "content": request}]
    })

    print(_extract_text(result["messages"][-1].content))
    print("\n" + "="*80 + "\n")


async def main():
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         EXPENSE ANALYZER AGENT - TEST SUITE                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    await test_standard_analysis()
    await test_with_custom_budget()
    await test_minimal_transactions()

    print("\n✅ All tests completed!\n")


if __name__ == "__main__":
    asyncio.run(main())