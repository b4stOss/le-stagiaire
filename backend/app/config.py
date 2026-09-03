from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
OCR_CACHE_DIR = DATA_DIR / "ocr"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    mistral_api_key: str
    database_url: str = "postgresql://rapport:rapport@localhost:5433/rapport"

    agent_model: str = "mistral-medium-latest"
    judge_model: str = "mistral-small-latest"
    embed_model: str = "mistral-embed"
    ocr_model: str = "mistral-ocr-latest"

    # Public demo guardrails: the app answers on a personal API key, every question costs ~1 cent.
    demo_daily_budget: int = 150
    demo_hourly_per_ip: int = 12
    demo_max_concurrent: int = 3

    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()  # pyright: ignore[reportCallIssue]  # fields come from the environment
