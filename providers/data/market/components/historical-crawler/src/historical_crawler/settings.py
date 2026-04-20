from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    database_url: str = "sqlite+pysqlite:///:memory:"
    kite_api_key: str = ""
    kite_access_token: str = ""
    historical_sync_utc_hour: int = 1

    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, extra="ignore")


settings = Settings()
