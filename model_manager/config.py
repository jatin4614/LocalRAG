"""Model manager settings."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelManagerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")

    vision_url:  str = Field(..., alias="VISION_URL")
    whisper_url: str = Field(..., alias="WHISPER_URL")
    idle_secs:   int = Field(300, alias="MODEL_UNLOAD_IDLE_SECS")
    poll_secs:   int = Field(15,  alias="MODEL_MANAGER_POLL_SECS")
    host:        str = Field("0.0.0.0", alias="MODEL_MANAGER_HOST")
    port:        int = Field(8080,      alias="MODEL_MANAGER_PORT")
