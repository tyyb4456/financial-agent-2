from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
# Load .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """
    All environment variables are read from .env automatically.
    Pydantic-settings v2 syntax — no class Config needed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM providers ────────────────────────────────────────────────────────
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # ── External APIs ─────────────────────────────────────────────────────────
    serper_api_key: str = Field(default="", alias="SERPER_API_KEY")
    reddit_client_id: str = Field(default="", alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", alias="REDDIT_CLIENT_SECRET")

    # ── LangSmith (optional — set to enable tracing) ─────────────────────────
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="crewai-migration", alias="LANGSMITH_PROJECT")

    # ── App ───────────────────────────────────────────────────────────────────
    app_port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


# Module-level singleton — import this everywhere
settings = Settings()