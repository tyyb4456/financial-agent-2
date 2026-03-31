# superior agent

from .financial_reporter import financial_reporter_agent
from .investment_advisor import financial_reporter_agent

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain.tools import tool
from langchain.agents import create_agent
from backend.config.llm import _llm

# Wrap it as a tool
@tool("financial", description="Analyze financial data and return insights")
def call_financial_agent(query: str):
    result = financial_reporter_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["result"]

@tool("investment", description="Analyze financial and valuation data to produce an investment report")
def call_investment_agent(query: str):
    result = financial_reporter_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["result"]

# Main agent with subagent as a tool
main_agent = create_agent(model=_llm, tools=[call_financial_agent, call_investment_agent])

result = main_agent.invoke({"input": "What is the current financial status of Apple Inc. and should I invest in it?"})
print(result)