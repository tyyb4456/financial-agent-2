"""
LLM factory
-----------
One place to create LLM instances.
Supports Google Gemini (default) and Groq as a fallback.

Usage:
    from config.llm import get_llm
    llm = get_llm()                          # default: gemini-1.5-flash
    llm = get_llm("groq/llama-3.3-70b-versatile")
"""

from functools import lru_cache
from langchain.chat_models import init_chat_model
from config.settings import settings


# ── Default model identifiers ─────────────────────────────────────────────────
# Format: (model_name, provider)  — kept separate for init_chat_model compat
DEFAULT_MODEL    = "gemini-2.5-flash"
DEFAULT_PROVIDER = "google_genai"


@lru_cache(maxsize=8)
def get_llm(
    model:    str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    temperature: float = 0.3,
):
    """
    Return a cached ChatModel instance.

    Args:
        model:       Model name, e.g. "gemini-1.5-flash" or "llama-3.3-70b-versatile"
        provider:    LangChain provider string, e.g. "google_genai" or "groq"
        temperature: Sampling temperature (0.0 – 1.0).

    Returns:
        A LangChain BaseChatModel instance, ready for .invoke() / .stream().
    """
    return init_chat_model(
        model,
        model_provider=provider,
        temperature=temperature,
    )