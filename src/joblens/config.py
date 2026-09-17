"""
Configuration.

Everything environment-specific lies here so the same code runs locally,
in CI and on a server. Nothing headcoded in the modules themselves.

"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql://joblens:joblens@localhost:5432/joblens"

    user_agent: str = "joblens/0.1"

    request_timeout: float = 20.0

    adzuna_app_id: str | None = None
    adzuna_app_key: str | None = None
    adzuna_country: str = "gb"

    # Phase 3. The local embedder needs nothing; the API one is only used to
    # reproduce the cost/quality comparison.
    embedder: str = "local"
    chunk_strategy: str = "whole"
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_api_key: str | None = None

    # Phase 4. `llm_backend` picks which of these two is live, so the same
    # code path serves whichever one is configured.
    llm_backend: str = "ollama"
    llm_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    llm_max_tokens: int = 1024
    llm_timeout: float = 120.0

    # Per-caller request budget for the LLM endpoints, counted in a fixed
    # window. Small on purpose: this is a personal project with a public URL.
    rate_limit_per_minute: int = 10

    @property
    def adzuna_enabled(self) -> bool:
        return bool(self.adzuna_app_id and self.adzuna_app_key)

    @property
    def llm_enabled(self) -> bool:
        if self.llm_backend == "anthropic":
            return bool(self.anthropic_api_key)
        return self.llm_backend == "ollama"


@lru_cache
def get_settings() -> Settings:
    return Settings()
