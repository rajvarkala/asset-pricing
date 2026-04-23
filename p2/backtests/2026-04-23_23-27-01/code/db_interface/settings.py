from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / "providers" / "data" / "market" / ".env"


class Settings(BaseSettings):
    database_url: str = "sqlite+pysqlite:///:memory:"

    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, extra="ignore")


settings = Settings()
