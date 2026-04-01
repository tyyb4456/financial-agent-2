"""
tools/expense.py
----------------
Expense categorization and budget analysis tools.

No Plaid/Yodlee needed — works with raw transaction data (from CSV,
manual input, or any bank export). Uses rule-based keyword matching
with LLM-assisted fallback for unknown merchants.

Three tools exposed (async):
  - parse_transactions       → normalize raw transaction dicts into clean records
  - categorize_expenses      → assign categories + subcategories to each transaction
  - generate_budget_report   → spending summary, budget vs actuals, anomalies, cuts
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, date
from typing import Literal
from collections import defaultdict

import structlog
from pydantic import BaseModel, Field
from langchain.tools import tool

log = structlog.get_logger(__name__)


# ── Category taxonomy ─────────────────────────────────────────────────────────
# Each entry: keyword (lowercase, partial match ok) → (category, subcategory)
# Ordered from most specific to most generic — first match wins.

_KEYWORD_MAP: list[tuple[str, str, str]] = [
    # Housing
    ("rent",          "Housing",        "Rent"),
    ("mortgage",      "Housing",        "Mortgage"),
    ("hoa",           "Housing",        "HOA"),
    ("electricity",   "Housing",        "Utilities"),
    ("gas bill",      "Housing",        "Utilities"),
    ("water bill",    "Housing",        "Utilities"),
    ("internet",      "Housing",        "Internet"),
    ("comcast",       "Housing",        "Internet"),
    ("at&t",          "Housing",        "Internet/Phone"),
    ("verizon",       "Housing",        "Phone"),
    ("t-mobile",      "Housing",        "Phone"),

    # Food & Dining
    ("mcdonald",      "Food & Dining",  "Fast Food"),
    ("burger king",   "Food & Dining",  "Fast Food"),
    ("kfc",           "Food & Dining",  "Fast Food"),
    ("subway",        "Food & Dining",  "Fast Food"),
    ("domino",        "Food & Dining",  "Fast Food"),
    ("pizza",         "Food & Dining",  "Fast Food"),
    ("starbucks",     "Food & Dining",  "Coffee"),
    ("dunkin",        "Food & Dining",  "Coffee"),
    ("costa",         "Food & Dining",  "Coffee"),
    ("doordash",      "Food & Dining",  "Food Delivery"),
    ("uber eats",     "Food & Dining",  "Food Delivery"),
    ("grubhub",       "Food & Dining",  "Food Delivery"),
    ("instacart",     "Food & Dining",  "Groceries"),
    ("whole foods",   "Food & Dining",  "Groceries"),
    ("trader joe",    "Food & Dining",  "Groceries"),
    ("walmart",       "Food & Dining",  "Groceries"),
    ("costco",        "Food & Dining",  "Groceries"),
    ("kroger",        "Food & Dining",  "Groceries"),
    ("aldi",          "Food & Dining",  "Groceries"),
    ("safeway",       "Food & Dining",  "Groceries"),
    ("supermarket",   "Food & Dining",  "Groceries"),
    ("grocery",       "Food & Dining",  "Groceries"),
    ("restaurant",    "Food & Dining",  "Dining Out"),
    ("cafe",          "Food & Dining",  "Dining Out"),
    ("diner",         "Food & Dining",  "Dining Out"),
    ("sushi",         "Food & Dining",  "Dining Out"),
    ("chipotle",      "Food & Dining",  "Dining Out"),

    # Transport
    ("uber",          "Transport",      "Rideshare"),
    ("lyft",          "Transport",      "Rideshare"),
    ("careem",        "Transport",      "Rideshare"),
    ("grab",          "Transport",      "Rideshare"),
    ("gas station",   "Transport",      "Fuel"),
    ("shell",         "Transport",      "Fuel"),
    ("chevron",       "Transport",      "Fuel"),
    ("bp",            "Transport",      "Fuel"),
    ("exxon",         "Transport",      "Fuel"),
    ("parking",       "Transport",      "Parking"),
    ("toll",          "Transport",      "Tolls"),
    ("metro",         "Transport",      "Public Transit"),
    ("subway fare",   "Transport",      "Public Transit"),
    ("bus pass",      "Transport",      "Public Transit"),
    ("train",         "Transport",      "Public Transit"),
    ("airlines",      "Transport",      "Flights"),
    ("emirates",      "Transport",      "Flights"),
    ("delta",         "Transport",      "Flights"),
    ("united air",    "Transport",      "Flights"),
    ("car wash",      "Transport",      "Car Maintenance"),
    ("jiffy lube",    "Transport",      "Car Maintenance"),
    ("auto repair",   "Transport",      "Car Maintenance"),

    # Health
    ("pharmacy",      "Health",         "Pharmacy"),
    ("cvs",           "Health",         "Pharmacy"),
    ("walgreens",     "Health",         "Pharmacy"),
    ("rite aid",      "Health",         "Pharmacy"),
    ("doctor",        "Health",         "Medical"),
    ("clinic",        "Health",         "Medical"),
    ("hospital",      "Health",         "Medical"),
    ("dentist",       "Health",         "Dental"),
    ("gym",           "Health",         "Fitness"),
    ("planet fitness","Health",         "Fitness"),
    ("24 hour fitnes","Health",         "Fitness"),
    ("yoga",          "Health",         "Fitness"),
    ("health insuran","Health",         "Insurance"),

    # Entertainment
    ("netflix",       "Entertainment",  "Streaming"),
    ("spotify",       "Entertainment",  "Streaming"),
    ("hulu",          "Entertainment",  "Streaming"),
    ("disney+",       "Entertainment",  "Streaming"),
    ("hbo",           "Entertainment",  "Streaming"),
    ("apple tv",      "Entertainment",  "Streaming"),
    ("prime video",   "Entertainment",  "Streaming"),
    ("youtube premium","Entertainment", "Streaming"),
    ("cinema",        "Entertainment",  "Movies"),
    ("amc theatre",   "Entertainment",  "Movies"),
    ("movie",         "Entertainment",  "Movies"),
    ("concert",       "Entertainment",  "Events"),
    ("ticketmaster",  "Entertainment",  "Events"),
    ("steam",         "Entertainment",  "Gaming"),
    ("playstation",   "Entertainment",  "Gaming"),
    ("xbox",          "Entertainment",  "Gaming"),
    ("nintendo",      "Entertainment",  "Gaming"),

    # Shopping
    ("amazon",        "Shopping",       "Online Retail"),
    ("ebay",          "Shopping",       "Online Retail"),
    ("aliexpress",    "Shopping",       "Online Retail"),
    ("shopify",       "Shopping",       "Online Retail"),
    ("apple store",   "Shopping",       "Electronics"),
    ("best buy",      "Shopping",       "Electronics"),
    ("ikea",          "Shopping",       "Home & Furniture"),
    ("home depot",    "Shopping",       "Home & Furniture"),
    ("lowes",         "Shopping",       "Home & Furniture"),
    ("nike",          "Shopping",       "Clothing"),
    ("zara",          "Shopping",       "Clothing"),
    ("h&m",           "Shopping",       "Clothing"),

    # Finance
    ("loan payment",  "Finance",        "Loan Repayment"),
    ("credit card",   "Finance",        "Credit Card Payment"),
    ("insurance",     "Finance",        "Insurance"),
    ("investment",    "Finance",        "Investments"),
    ("brokerage",     "Finance",        "Investments"),
    ("bank fee",      "Finance",        "Bank Fees"),
    ("atm fee",       "Finance",        "Bank Fees"),
    ("interest charg","Finance",        "Interest"),

    # Education
    ("tuition",       "Education",      "Tuition"),
    ("udemy",         "Education",      "Online Courses"),
    ("coursera",      "Education",      "Online Courses"),
    ("book",          "Education",      "Books"),
    ("kindle",        "Education",      "Books"),
    ("school",        "Education",      "Tuition"),

    # Income (credits)
    ("payroll",       "Income",         "Salary"),
    ("salary",        "Income",         "Salary"),
    ("direct deposit","Income",         "Salary"),
    ("freelance",     "Income",         "Freelance"),
    ("transfer in",   "Income",         "Transfer"),
    ("refund",        "Income",         "Refund"),
    ("cashback",      "Income",         "Cashback"),

    # Travel
    ("hotel",         "Travel",         "Accommodation"),
    ("airbnb",        "Travel",         "Accommodation"),
    ("booking.com",   "Travel",         "Accommodation"),
    ("expedia",       "Travel",         "Travel Agency"),

    # Subscriptions
    ("subscription",  "Subscriptions",  "Software"),
    ("saas",          "Subscriptions",  "Software"),
    ("adobe",         "Subscriptions",  "Software"),
    ("microsoft 365", "Subscriptions",  "Software"),
    ("dropbox",       "Subscriptions",  "Software"),
    ("notion",        "Subscriptions",  "Software"),
    ("chatgpt",       "Subscriptions",  "AI Tools"),
    ("claude",        "Subscriptions",  "AI Tools"),
    ("openai",        "Subscriptions",  "AI Tools"),
]

# Default budgets (monthly, USD) — user can override via budget_limits param
_DEFAULT_BUDGETS: dict[str, float] = {
    "Housing":       1500.0,
    "Food & Dining":  600.0,
    "Transport":      300.0,
    "Health":         200.0,
    "Entertainment":  150.0,
    "Shopping":       300.0,
    "Finance":        400.0,
    "Education":      100.0,
    "Travel":         200.0,
    "Subscriptions":   80.0,
    "Other":          100.0,
}


# ── Arg schemas ───────────────────────────────────────────────────────────────

class ParseTransactionsInput(BaseModel):
    transactions: list[dict] = Field(
        description=(
            "List of raw transaction dicts. Each must have at minimum: "
            "'date' (YYYY-MM-DD or MM/DD/YYYY), 'description' (merchant name/note), "
            "'amount' (positive = debit/expense, negative = credit/income). "
            "Optional: 'id', 'account'."
        )
    )


class CategorizeExpensesInput(BaseModel):
    transactions: list[dict] = Field(
        description=(
            "Parsed transactions from parse_transactions. Each has: "
            "id, date, description, amount, account."
        )
    )


class GenerateBudgetReportInput(BaseModel):
    categorized: list[dict] = Field(
        description="Categorized transactions from categorize_expenses."
    )
    budget_limits: dict[str, float] | None = Field(
        default=None,
        description=(
            "Monthly budget limits per category in USD, e.g. "
            "{'Food & Dining': 500, 'Transport': 200}. "
            "Categories not specified get default limits."
        ),
    )
    month: str | None = Field(
        default=None,
        description=(
            "Month to analyze in YYYY-MM format, e.g. '2024-03'. "
            "If None, uses the most common month in the transactions."
        ),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_date(raw: str) -> str:
    """Convert various date formats to YYYY-MM-DD."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Last resort: return as-is
    return raw.strip()


def _categorize_single(description: str, amount: float) -> tuple[str, str]:
    """Return (category, subcategory) for a single transaction."""
    desc_lower = description.lower()

    # Credits (income/refunds)
    if amount < 0:
        for keyword, cat, subcat in _KEYWORD_MAP:
            if cat == "Income" and keyword in desc_lower:
                return cat, subcat
        return "Income", "Other Income"

    # Debits — keyword match
    for keyword, cat, subcat in _KEYWORD_MAP:
        if keyword in desc_lower:
            return cat, subcat

    return "Other", "Uncategorized"


def _safe_float(val) -> float:
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return 0.0


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("parse_transactions", args_schema=ParseTransactionsInput)
async def parse_transactions(transactions: list[dict]) -> list[dict]:
    """
    Normalize raw bank transaction records into a clean, consistent format.

    Accepts transactions from any source (CSV export, manual entry, bank API).
    Standardizes date format, amount sign convention, and adds a unique ID.

    Returns a list of clean transaction dicts ready for categorize_expenses.
    Call this first before any other expense tool.
    """
    log.info("tool.parse_transactions", count=len(transactions))

    def _parse_one(i: int, tx: dict) -> dict:
        raw_date   = str(tx.get("date", ""))
        desc       = str(tx.get("description", tx.get("memo", tx.get("name", "Unknown")))).strip()
        amount     = _safe_float(tx.get("amount", tx.get("debit", tx.get("credit", 0))))
        account    = str(tx.get("account", tx.get("account_name", "Primary"))).strip()
        tx_id      = str(tx.get("id", tx.get("transaction_id", f"tx_{i:04d}")))

        # Some exports use separate debit/credit columns
        if "debit" in tx and "credit" in tx:
            debit  = _safe_float(tx.get("debit") or 0)
            credit = _safe_float(tx.get("credit") or 0)
            amount = debit - credit  # positive = expense, negative = income

        return {
            "id":          tx_id,
            "date":        _normalize_date(raw_date),
            "description": desc,
            "amount":      amount,      # positive = expense / debit
            "account":     account,
        }

    def _run_sync():
        return [_parse_one(i, tx) for i, tx in enumerate(transactions)]

    try:
        result = await asyncio.to_thread(_run_sync)
        log.info("tool.parse_transactions.done", parsed=len(result))
        return result
    except Exception as exc:
        log.error("tool.parse_transactions.error", error=str(exc))
        return [{"error": str(exc)}]


@tool("categorize_expenses", args_schema=CategorizeExpensesInput)
async def categorize_expenses(transactions: list[dict]) -> list[dict]:
    """
    Categorize each transaction into a spending category and subcategory.

    Uses a merchant keyword database covering 80+ common merchants and
    payment types across Housing, Food & Dining, Transport, Health,
    Entertainment, Shopping, Finance, Education, Travel, Subscriptions.

    Returns transactions with added 'category', 'subcategory', and
    'is_income' fields. Call parse_transactions first, then this.
    """
    log.info("tool.categorize_expenses", count=len(transactions))

    def _run_sync():
        result = []
        for tx in transactions:
            if "error" in tx:
                result.append(tx)
                continue
            desc    = tx.get("description", "")
            amount  = _safe_float(tx.get("amount", 0))
            cat, subcat = _categorize_single(desc, amount)
            result.append({
                **tx,
                "category":    cat,
                "subcategory": subcat,
                "is_income":   amount < 0,
            })
        return result

    try:
        result = await asyncio.to_thread(_run_sync)
        cats = defaultdict(int)
        for tx in result:
            cats[tx.get("category", "?")] += 1
        log.info("tool.categorize_expenses.done",
                 total=len(result),
                 categories=dict(cats))
        return result
    except Exception as exc:
        log.error("tool.categorize_expenses.error", error=str(exc))
        return [{"error": str(exc)}]


@tool("generate_budget_report", args_schema=GenerateBudgetReportInput)
async def generate_budget_report(
    categorized: list[dict],
    budget_limits: dict[str, float] | None = None,
    month: str | None = None,
) -> dict:
    """
    Generate a comprehensive monthly budget report from categorized transactions.

    Produces:
      - Total income vs total spending
      - Per-category spending vs budget (over/under, % used)
      - Top 10 largest individual transactions
      - Unusual spends (transactions > 2x category average)
      - Subscription inventory (recurring monthly charges)
      - Actionable cut suggestions for over-budget categories
      - Net savings rate

    Call categorize_expenses first, then pass results here.
    """
    log.info("tool.generate_budget_report", txn_count=len(categorized))

    def _run_sync():
        budgets = {**_DEFAULT_BUDGETS, **(budget_limits or {})}

        # ── Filter to target month ──────────────────────────────────────────
        all_months = []
        for tx in categorized:
            d = tx.get("date", "")
            if len(d) >= 7:
                all_months.append(d[:7])

        target_month = month
        if not target_month and all_months:
            # Use the most common month in the dataset
            target_month = max(set(all_months), key=all_months.count)

        if target_month:
            monthly_txns = [tx for tx in categorized if tx.get("date", "").startswith(target_month)]
        else:
            monthly_txns = categorized

        if not monthly_txns:
            return {"error": f"No transactions found for month '{target_month}'."}

        # ── Totals ──────────────────────────────────────────────────────────
        income_txns  = [tx for tx in monthly_txns if tx.get("is_income")]
        expense_txns = [tx for tx in monthly_txns if not tx.get("is_income")]

        total_income   = round(sum(abs(tx["amount"]) for tx in income_txns), 2)
        total_expenses = round(sum(tx["amount"] for tx in expense_txns), 2)
        net_savings    = round(total_income - total_expenses, 2)
        savings_rate   = round(net_savings / total_income * 100, 1) if total_income > 0 else None

        # ── Per-category breakdown ──────────────────────────────────────────
        cat_totals: dict[str, float] = defaultdict(float)
        cat_counts: dict[str, int]   = defaultdict(int)
        cat_txns:   dict[str, list]  = defaultdict(list)

        for tx in expense_txns:
            cat = tx.get("category", "Other")
            cat_totals[cat] = round(cat_totals[cat] + tx["amount"], 2)
            cat_counts[cat] += 1
            cat_txns[cat].append(tx)

        category_summary = []
        over_budget_cats = []

        for cat, spent in sorted(cat_totals.items(), key=lambda x: -x[1]):
            budget     = budgets.get(cat, budgets.get("Other", 100.0))
            remaining  = round(budget - spent, 2)
            pct_used   = round(spent / budget * 100, 1) if budget > 0 else None
            over_budget = spent > budget

            if over_budget:
                over_budget_cats.append({
                    "category":  cat,
                    "spent":     spent,
                    "budget":    budget,
                    "overage":   round(spent - budget, 2),
                    "pct_used":  pct_used,
                })

            category_summary.append({
                "category":       cat,
                "spent":          spent,
                "budget":         budget,
                "remaining":      remaining,
                "pct_used":       pct_used,
                "transaction_count": cat_counts[cat],
                "status":         "OVER BUDGET" if over_budget else ("WARNING" if pct_used and pct_used > 80 else "OK"),
            })

        # ── Top 10 largest expenses ─────────────────────────────────────────
        top_expenses = sorted(expense_txns, key=lambda x: -x["amount"])[:10]
        top_expenses_out = [
            {
                "date":        tx["date"],
                "description": tx["description"],
                "amount":      tx["amount"],
                "category":    tx.get("category", "Other"),
                "subcategory": tx.get("subcategory", ""),
            }
            for tx in top_expenses
        ]

        # ── Unusual spends (> 2x category average) ─────────────────────────
        anomalies = []
        for cat, txns in cat_txns.items():
            if len(txns) < 2:
                continue
            avg = sum(t["amount"] for t in txns) / len(txns)
            for tx in txns:
                if tx["amount"] > avg * 2 and tx["amount"] > 50:
                    anomalies.append({
                        "date":        tx["date"],
                        "description": tx["description"],
                        "amount":      tx["amount"],
                        "category":    cat,
                        "avg_for_cat": round(avg, 2),
                        "multiple":    round(tx["amount"] / avg, 1),
                    })
        anomalies.sort(key=lambda x: -x["amount"])

        # ── Subscription inventory ──────────────────────────────────────────
        subscription_cats = {"Subscriptions", "Entertainment"}
        subscriptions = []
        seen_subs: set[str] = set()

        for tx in expense_txns:
            if tx.get("category") in subscription_cats or tx.get("subcategory") == "Streaming":
                key = tx["description"].lower()[:20]
                if key not in seen_subs:
                    seen_subs.add(key)
                    subscriptions.append({
                        "service": tx["description"],
                        "amount":  tx["amount"],
                        "category": tx.get("subcategory", tx.get("category")),
                    })

        total_subscriptions = round(sum(s["amount"] for s in subscriptions), 2)

        # ── Cut suggestions ─────────────────────────────────────────────────
        cut_suggestions = []

        for item in over_budget_cats:
            cat     = item["category"]
            overage = item["overage"]
            spent   = item["spent"]

            if cat == "Food & Dining":
                cut_suggestions.append({
                    "category":   cat,
                    "overage":    overage,
                    "suggestion": f"You overspent by ${overage:.0f}. Cook 3 more meals at home per week (~$60–90 savings) and cut food delivery to max 2x/week.",
                    "potential_saving": round(overage * 0.6, 2),
                })
            elif cat == "Entertainment":
                cut_suggestions.append({
                    "category":   cat,
                    "overage":    overage,
                    "suggestion": f"Audit your {len(subscriptions)} active subscriptions (${total_subscriptions:.0f}/mo total). Cancel any unused for 3+ months.",
                    "potential_saving": round(total_subscriptions * 0.3, 2),
                })
            elif cat == "Shopping":
                cut_suggestions.append({
                    "category":   cat,
                    "overage":    overage,
                    "suggestion": f"${spent:.0f} spent on shopping. Introduce a 48-hour rule before purchases over $50 to reduce impulse buys.",
                    "potential_saving": round(overage * 0.5, 2),
                })
            elif cat == "Transport":
                cut_suggestions.append({
                    "category":   cat,
                    "overage":    overage,
                    "suggestion": f"${spent:.0f} on transport. Replace 2 Uber rides/week with public transit or carpooling to save ~${overage * 0.4:.0f}/mo.",
                    "potential_saving": round(overage * 0.4, 2),
                })
            elif cat == "Subscriptions":
                cut_suggestions.append({
                    "category":   cat,
                    "overage":    overage,
                    "suggestion": f"${total_subscriptions:.0f}/mo in subscriptions. Review and cancel 1–2 services you use less than once/week.",
                    "potential_saving": round(total_subscriptions * 0.35, 2),
                })
            else:
                cut_suggestions.append({
                    "category":   cat,
                    "overage":    overage,
                    "suggestion": f"${cat} exceeded budget by ${overage:.0f}. Set a firm weekly sub-limit of ${item['budget'] / 4:.0f}/week.",
                    "potential_saving": round(overage * 0.5, 2),
                })

        total_potential_saving = round(sum(c["potential_saving"] for c in cut_suggestions), 2)

        return {
            "month":           target_month or "All periods",
            "transaction_count": len(monthly_txns),

            # ── Top-line ──
            "total_income":    total_income,
            "total_expenses":  total_expenses,
            "net_savings":     net_savings,
            "savings_rate_pct": savings_rate,

            # ── Category breakdown ──
            "category_breakdown":  category_summary,
            "over_budget_count":   len(over_budget_cats),
            "over_budget_categories": over_budget_cats,

            # ── Details ──
            "top_expenses":     top_expenses_out,
            "anomalies":        anomalies[:5],   # top 5 unusual spends
            "subscriptions":    subscriptions,
            "total_subscriptions_cost": total_subscriptions,

            # ── Actions ──
            "cut_suggestions":           cut_suggestions,
            "total_potential_saving":    total_potential_saving,
        }

    try:
        result = await asyncio.to_thread(_run_sync)
        log.info("tool.generate_budget_report.done",
                 month=result.get("month"),
                 total_expenses=result.get("total_expenses"),
                 over_budget=result.get("over_budget_count"))
        return result
    except Exception as exc:
        log.error("tool.generate_budget_report.error", error=str(exc))
        return {"error": str(exc)}