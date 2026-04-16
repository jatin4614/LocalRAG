"""Application settings loaded from environment."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")

    database_url:   str = Field(..., alias="DATABASE_URL")
    redis_url:      str = Field(..., alias="REDIS_URL")
    qdrant_url:     str = Field(..., alias="QDRANT_URL")
    session_secret: str = Field(..., alias="SESSION_SECRET", min_length=32)
    tei_url:        str = Field("http://tei:80", alias="TEI_URL")
    vector_size:    int = Field(1024,            alias="RAG_VECTOR_SIZE")


@lru_cache
def _settings_cached() -> Settings:
    return Settings()  # type: ignore[call-arg]  # fields populated from env


def get_settings() -> Settings:
    return _settings_cached()


def clear_settings_cache() -> None:
    _settings_cached.cache_clear()
