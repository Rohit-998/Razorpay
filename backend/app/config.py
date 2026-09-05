"""PayRevive — Application configuration (env-based)."""

from pathlib import Path

from pydantic_settings import BaseSettings
from functools import lru_cache

# `env_file` is resolved against the process's working directory, which made loading the
# credentials depend on where you happened to be standing. `uvicorn app.main:app` has to be
# run from `backend/`, `python -m app.eval` too, but the repository root is where a `.env`
# naturally lands — and from `backend/` that file is invisible. The symptom is not a missing
# key, it is `SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env` thrown by a
# service whose `.env` exists and is correct one directory up.
#
# So both are named, absolutely, and pydantic reads them in order with the later winning:
# repository root first, then `backend/.env` as the override the README tells you to create.
_BACKEND = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND.parent


class Settings(BaseSettings):
    """All settings loaded from environment variables."""

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Supabase
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_service_role_key: str = ""
    database_url: str = ""  # Direct PostgreSQL URL for migrations

    # Redis Cloud
    redis_url: str = "redis://localhost:6379/0"

    # Razorpay
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    # Gemini
    gemini_api_key: str = ""

    # Recovery Policy
    max_retries_per_payment: int = 3
    max_contacts_per_day: int = 2
    quiet_hours_start: int = 22
    quiet_hours_end: int = 8
    max_recovery_window_hours: int = 72
    min_retry_interval_minutes: int = 15
    upi_transaction_ceiling_paise: int = 10_000_000
    require_action_above_paise: int = 1_000_000

    # LLM
    llm_confidence_threshold: float = 0.7
    llm_amount_threshold_paise: int = 1_000_000

    model_config = {
        "env_file": (str(_REPO_ROOT / ".env"), str(_BACKEND / ".env")),
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
