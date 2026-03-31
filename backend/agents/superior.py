# superior agent

from .financial_reporter import financial_reporter_agent
from .investment_advisor import investment_advisor_agent
from .query_answerer import query_agent

from dotenv import load_dotenv
load_dotenv()

import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain.tools import tool
from langchain.agents import create_agent
from backend.config.llm import get_llm

_llm = get_llm()

_retry = retry(
    retry=retry_if_exception_type((httpx.RemoteProtocolError, httpx.ConnectError, Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)

def _extract_text(content) -> str:
    """Handle both string and Gemini list-of-blocks content format."""
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if b.get("type") == "text")
    return content

@tool("financial", description="Generate a full financial snapshot report for a stock. Use for broad financial status questions.")
async def call_financial_agent(query: str) -> str:
    @_retry
    async def _run():
        result = await financial_reporter_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()

@tool("investment", description="Generate a full investment report with BUY/HOLD/SELL recommendation and price target.")
async def call_investment_agent(query: str) -> str:
    @_retry
    async def _run():
        result = await investment_advisor_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()

@tool("query", description="Answer any free-form question about a stock. Use for specific questions like 'Is TSLA overvalued?', 'What is Apple profit margin?', 'What do analysts think about NVDA?'")
async def call_query_agent(query: str) -> str:
    @_retry
    async def _run():
        result = await query_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()

main_agent = create_agent(
    model=_llm,
    tools=[call_financial_agent, call_investment_agent, call_query_agent],
    system_prompt="""\
You are a financial assistant that routes user questions to the right specialist tool.

## Routing Rules
- Broad financial status questions (revenue, income, market cap overview) → use `financial` tool
- Investment decision questions (should I buy/sell, price target, recommendation) → use `investment` tool  
- Specific free-form questions about a stock → use `query` tool

Always pass the full original question including the stock ticker to the tool.
"""
)

async def main():
    print("Financial Assistant ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("Ask me anything about a stock: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = await main_agent.ainvoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        content = result["messages"][-1].content
        print("\n" + _extract_text(content) + "\n")

if __name__ == "__main__":
    asyncio.run(main())