from __future__ import annotations

import os
import tomllib
from typing import Any

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class RookeenSettings(BaseModel):
    """Settings controlling Rookeen CLI behavior.

    Precedence when loading effective values:
    1) CLI flags (handled in CLI layer)
    2) Environment variables (prefix ROOKEEN_)
    3) Config file (TOML)
    4) Defaults defined here
    """

    models_auto_download: bool = Field(default=True)
    languages_preload: list[str] = Field(default_factory=list)
    format: str = Field(default="json")
    output_dir: str = Field(default="results")
    concurrency: int = Field(default=2)
    timeout_seconds: int = Field(default=30)
    rate_limit_rps: float = Field(default=0.5)
    log_level: str = Field(default="INFO")
    default_language: str = Field(default="")
    # Embeddings configuration (optional analyzers)
    embeddings_backend: str = Field(default="miniLM")
    embeddings_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    openai_api_key: str = Field(default="")

    model_config = ConfigDict(extra="ignore")


def _parse_bool(value: str) -> bool:
    truthy = {"1", "true", "yes", "on", "y", "t"}
    falsy = {"0", "false", "no", "off", "n", "f"}
    v = value.strip().lower()
    if v in truthy:
        return True
    if v in falsy:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_list_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_toml_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    # Support either flat keys or under a [rookeen] table
    if isinstance(data, dict) and "rookeen" in data and isinstance(data["rookeen"], dict):
        return dict(data["rookeen"])  # copy
    return dict(data)


def _apply_env_overrides(base: dict[str, Any], env_prefix: str = "ROOKEEN_") -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    # Map of field -> parser
    field_parsers: dict[str, Any] = {
        "models_auto_download": _parse_bool,
        "languages_preload": _parse_list_csv,
        "format": str,
        "output_dir": str,
        "concurrency": int,
        "timeout_seconds": int,
        "rate_limit_rps": float,
        "log_level": str,
        "default_language": str,
        # Embeddings-related overrides
        "embeddings_backend": str,
        "embeddings_model": str,
        "openai_api_key": str,
    }
    for field, parser in field_parsers.items():
        env_key = f"{env_prefix}{field}".upper()
        raw = os.getenv(env_key)
        if raw is None:
            continue
        try:
            merged[field] = parser(raw)
        except Exception:
            # Ignore invalid env values; validation will catch later if needed
            continue
    return merged


def load_settings(
    config_path: str | None = None, env_prefix: str = "ROOKEEN_"
) -> RookeenSettings:
    """Load settings from optional TOML file, then apply environment overrides.

    Environment variables use the given prefix and snake_case field names, e.g.:
    - ROOKEEN_FORMAT=md
    - ROOKEEN_LANGUAGES_PRELOAD=en,de
    - ROOKEEN_MODELS_AUTO_DOWNLOAD=false
    """
    file_cfg = _load_toml_config(config_path)
    merged = _apply_env_overrides(file_cfg, env_prefix=env_prefix)
    return RookeenSettings(**merged)
