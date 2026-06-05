from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_TIMEOUT: float = 120.0          # seconds per request
    OLLAMA_MAX_RETRIES: int = 3
    OLLAMA_MODE: str = "local"             # "local" | "cloud"

    # Required when OLLAMA_MODE=cloud
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    DATABASE_URL: str = "postgres://daedalus:daedalus@localhost:5432/daedalus"
    PYTHON_AI_PORT: int = 3020
    GO_API_URL: str = "http://localhost:3010"


settings = Settings()
