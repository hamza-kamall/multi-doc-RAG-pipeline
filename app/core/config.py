from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multi Document RAG API"
    app_version: str = "0.1.0"
    environment: str = "local"
    log_level: str = "INFO"
    openai_api_key: str | None = None
    chroma_dir: str = "chroma_db"
    chroma_collection: str = "rag_docs"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
