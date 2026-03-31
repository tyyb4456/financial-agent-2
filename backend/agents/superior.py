# superior agent

from .financial_reporting import financial_reporter_agent
from .investment_advise import investment_advisor_agent
from .query_answer import query_agent
from .news_reporting import news_reporter_agent
from .competitor_benchmarking import competitor_benchmarking_agent
from .reddit_posts_analysis import reddit_posts_analyst_agent
from .technical_analysis import technical_analysis_agent

from dotenv import load_dotenv
load_dotenv()

import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain.tools import tool
from langchain.agents import create_agent
from backend.config.llm import get_llm
from pydantic import BaseModel, Field

class RedditInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. TSLA")
    subreddit: str = Field(description="Subreddit name, e.g. wallstreetbets")

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


# ------------- Define tool wrappers for async calls to each agent -------------



# ---------------------
# fincial reporter tool
#---------------------

@tool("financial", description="Generate a full financial snapshot report for a stock. Use for broad financial status questions.")
async def call_financial_agent(symbol: str) -> str:
    @_retry
    async def _run():
        result = await financial_reporter_agent.ainvoke({
            "messages": [{"role": "user", "content": symbol}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()





# ----------------------
# investment advisor tool
#----------------------

@tool("investment", description="Generate a full investment report with BUY/HOLD/SELL recommendation and price target.")
async def call_investment_agent(symbol: str) -> str:
    @_retry
    async def _run():
        result = await investment_advisor_agent.ainvoke({
            "messages": [{"role": "user", "content": symbol}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()




# ----------------------
# query agent tool
#----------------------

@tool("query", description="Answer any free-form question about a stock. Use for specific questions like 'Is TSLA overvalued?', 'What is Apple profit margin?', 'What do analysts think about NVDA?'")
async def call_query_agent(query: str) -> str:
    @_retry
    async def _run():
        result = await query_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()




# ----------------------
# news reporter tool
#----------------------


@tool("search_news", description="Search recent news articles about a stock or topic. Returns a list of relevant news stories with title, snippet, URL, and date.")
async def call_news_agent(query: str) -> str:
    @_retry
    async def _run():
        result = await news_reporter_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()



# ----------------------
# competitor benchmarking tool
# ----------------------



@tool("benchmark", description="Compare a stock's financial and market metrics against its top 3 competitors. Use for questions like 'How does MSFT compare to its competitors?'")
async def call_benchmark_agent(query: str) -> str:
    @_retry
    async def _run():
        result = await competitor_benchmarking_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()



# ----------------------
# reddit posts analyst tool
# ----------------------


@tool("reddit_post_analyst", args_schema=RedditInput,
      description="Analyse Reddit sentiment for a stock using a specific subreddit.")
async def call_reddit_agent(symbol: str, subreddit: str) -> str:
    @_retry
    async def _run():
        result = await reddit_posts_analyst_agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"{symbol} {subreddit}"
            }]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()



# ------------------
# technical analyst tool
# ------------------

@tool("technical_analysis", description="Provide a technical analysis summary based on recent price and volume trends. Use for questions like 'What is the technical outlook for AMZN?' or 'Is AAPL showing bullish or bearish technical signals?'")
async def call_technical_agent(query: str) -> str:
    @_retry
    async def _run():
        result = await technical_analysis_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        return _extract_text(result["messages"][-1].content)
    return await _run()




# ------------- Main superior agent that routes to the right specialist tool -------------

MAIN_PROMPT = """\
You are a financial research assistant that decomposes user queries and routes them
to specialist tools. You synthesise the results into a single coherent response.

## Your Specialist Tools

| Tool                 | Use When                                                              |
|----------------------|-----------------------------------------------------------------------|
| `financial`          | User wants an overview: revenue, margins, market cap, balance sheet   |
| `investment`         | User wants a decision: buy/hold/sell, price target, upside/downside   |
| `query`              | User asks a specific factual question about a stock                   |
| `search_news`        | User asks about recent events, catalysts, or headlines                |
| `benchmark`          | User asks how a stock compares to peers or competitors                |
| `reddit_post_analyst`| User asks about retail sentiment, Reddit chatter, or crowd opinion    |
| `technical_analysis` | User asks about chart signals, price action, or a trade setup         |

---

## Routing Rules

### Step 1 — Identify intent(s)
A query can have multiple intents. Extract each one.
  "Is NVDA a good buy right now and what does Reddit think?"
  → Intent 1: investment decision → `investment`
  → Intent 2: retail sentiment → `reddit_post_analyst`
  Call both tools. Synthesise results.

### Step 2 — Select tool(s)
Match each intent to exactly one tool using the table above.
  - If the query is broad (e.g. "tell me everything about TSLA") → call `financial`, `investment`, and `query` in parallel.
  - If the query mentions "chart", "technical", "RSI", "support", "resistance", "trade setup" → always include `technical_analysis`.
  - If the query mentions "news", "earnings", "announced", "just happened" → always include `search_news`.
  - If the query mentions "Reddit", "wallstreetbets", "retail", "sentiment" → always include `reddit_post_analyst`.
  - If the query mentions "compare", "vs", "better than", "peers", "competitors" → always include `benchmark`.

### Step 3 — 
--pass the only stock symbol to tools like financial, investment, benchmark
-- Pass the full query to tools like query, news, technical since they may need the context to determine what specific information to pull.
-- passs the stock symbol and subreddit to reddit_post_analyst since it needs both to fetch the right posts and analyse them.
Always pass the user's original query including the ticker symbol to every tool you call.
Do not paraphrase or strip context from the query before sending it.

### Step 4 — Pass Inputs to Tools

- For `financial`, `investment`, and `benchmark`:
  → Pass only the stock symbol (e.g. "AAPL", "TSLA").

- For `query`, `search_news`, and `technical_analysis`:
  → Pass the full user query, as these tools require context to extract specific insights.

- For `reddit_post_analyst`:
  → Pass both:
     • the stock symbol
     • the subreddit name (e.g. "wallstreetbets")

- Always preserve the original user intent.
  Do not paraphrase or remove important context when passing inputs to tools.

## Hard Rules
- Never answer from your own knowledge alone if a tool is available for that intent.
- Never call the same tool twice for the same query.
- Never say "I can't access real-time data" — you have tools for that.
- If a ticker is ambiguous (e.g. "Apple" without AAPL), resolve it to the most likely symbol before calling tools.
- If no ticker is provided and one is required, ask the user for it before calling any tool.
"""


main_agent = create_agent(
    model=_llm,
    tools=[
        call_financial_agent,
        call_investment_agent,
        call_query_agent,
        call_news_agent,
        call_benchmark_agent,
        call_reddit_agent,
        call_technical_agent,
    ],
    system_prompt=MAIN_PROMPT
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