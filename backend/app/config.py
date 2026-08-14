from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
OCR_CACHE_DIR = DATA_DIR / "ocr"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    mistral_api_key: str
    database_url: str = "postgresql://rapport:rapport@localhost:5433/rapport"

    agent_model: str = "mistral-medium-latest"
    judge_model: str = "mistral-small-latest"
    embed_model: str = "mistral-embed"
    ocr_model: str = "mistral-ocr-latest"


settings = Settings()
