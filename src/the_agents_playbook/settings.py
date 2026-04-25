from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    openai_model: str = "openai/gpt-oss-20b"

    @model_validator(mode="after")
    def at_least_one_api_key(self) -> "Settings":
        keys = [self.openai_api_key]
        if not any(keys):
            raise ValueError(
                "At least one API key must be set: OPENAI_API_KEY"
            )
        return self


settings = Settings()
