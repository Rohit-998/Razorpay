"""PayRevive — Application configuration (env-based)."""

from pydantic_settings import BaseSettings
from functools import lru_cache


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
    max_link_amount_paise: int = 5_000_000
    require_action_above_paise: int = 1_000_000

    # LLM
    llm_confidence_threshold: float = 0.7
    llm_amount_threshold_paise: int = 1_000_000

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
