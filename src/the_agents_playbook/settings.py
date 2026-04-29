from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

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
    def at_least_one_api_key(self) -> "Settings":
        if self.mock_only:
            return self
        keys = [self.openai_api_key, self.anthropic_api_key]
        if not any(keys):
            raise ValueError(
                "At least one API key must be set: OPENAI_API_KEY, ANTHROPIC_API_KEY "
                "(or set MOCK_ONLY=true)"
            )
        return self


settings = Settings()
