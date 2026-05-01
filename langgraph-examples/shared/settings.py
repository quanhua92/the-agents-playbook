"""Settings and LLM factory helpers for LangGraph examples.

Adapted from the_agents_playbook/settings.py. Cannot import from root SDK
since langgraph-examples is an independent workspace member.

Looks for .env in the current directory first, then walks up to find it
in the project root.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Path:
    """Walk up from this file to find .env."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        current = current.parent
    return Path(".env")  # fallback, will rely on env vars


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_find_env_file())

    # OpenAI / OpenRouter
    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    openai_model: str = "openai/gpt-oss-20b"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_model: str = "claude-sonnet-4-6"

    # Embeddings via OpenRouter
    embedding_api_key: str = ""
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"

    # General
    mock_only: bool = False

    @model_validator(mode="after")
    def at_least_one_api_key(self) -> Settings:
        if self.mock_only:
            return self
        keys = [self.openai_api_key, self.anthropic_api_key]
        if not any(keys):
            raise ValueError(
                "At least one API key must be set: OPENAI_API_KEY, ANTHROPIC_API_KEY "
                "(or set MOCK_ONLY=true)"
            )
        return self


def validate_config(cfg: Settings | None = None) -> list[str]:
    """Cross-check settings that Pydantic can't validate alone.

    Returns a list of warning strings. Empty list means no issues.
    """
    cfg = cfg or settings
    warnings: list[str] = []

    if cfg.openai_api_key and not cfg.openai_api_key.startswith("sk-"):
        warnings.append(
            f"OPENAI_API_KEY looks malformed (expected 'sk-...' prefix): "
            f"{cfg.openai_api_key[:8]}..."
        )

    if cfg.anthropic_api_key and not cfg.anthropic_api_key.startswith("sk-ant-"):
        warnings.append(
            f"ANTHROPIC_API_KEY looks malformed (expected 'sk-ant-...' prefix): "
            f"{cfg.anthropic_api_key[:8]}..."
        )

    if cfg.openai_api_key and "anthropic" in cfg.openai_base_url.lower():
        warnings.append(
            "OPENAI_API_KEY is set but OPENAI_BASE_URL contains 'anthropic'. "
            "Check that the key matches the provider."
        )

    if cfg.anthropic_api_key and "openai" in cfg.anthropic_base_url.lower():
        warnings.append(
            "ANTHROPIC_API_KEY is set but ANTHROPIC_BASE_URL contains 'openai'. "
            "Check that the key matches the provider."
        )

    if not cfg.mock_only and not cfg.embedding_api_key:
        warnings.append(
            "No EMBEDDING_API_KEY set. Vector memory will not work until "
            "an embedding provider is configured."
        )

    if (
        cfg.openai_model.startswith("anthropic/")
        and "anthropic" not in cfg.openai_base_url.lower()
        and "openrouter" not in cfg.openai_base_url.lower()
    ):
        warnings.append(
            f"OPENAI_MODEL starts with 'anthropic/' but OPENAI_BASE_URL "
            f"({cfg.openai_base_url}) may not route to Anthropic. "
            f"Consider using the AnthropicProvider directly."
        )

    return warnings


settings = Settings()


def get_openai_llm(**overrides):
    """Return a ChatOpenAI instance configured from Settings."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=SecretStr(settings.openai_api_key) if settings.openai_api_key else None,
        base_url=settings.openai_base_url,
        **overrides,
    )


def get_anthropic_llm(**overrides):
    """Return a ChatAnthropic instance configured from Settings."""
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for get_anthropic_llm()")
    return ChatAnthropic(
        model_name=settings.anthropic_model,
        api_key=SecretStr(settings.anthropic_api_key),
        base_url=settings.anthropic_base_url,
        **overrides,
    )
