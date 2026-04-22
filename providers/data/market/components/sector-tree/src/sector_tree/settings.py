from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/market_data"

    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, extra="ignore")


settings = Settings()
