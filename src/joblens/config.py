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

    @property
    def adzuna_enabled(self) -> bool:
        return bool(self.adzuna_app_id and self.adzuna_app_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
