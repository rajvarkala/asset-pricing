from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    """Settings for financial data processor."""

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/market_data"
    serialized_data_path: str = "./data/serialized"

    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, extra="ignore")


settings = Settings()
