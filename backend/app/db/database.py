"""Supabase client — database access layer."""

from supabase import create_client, Client
from app.config import get_settings
import structlog

logger = structlog.get_logger()

_client: Client | None = None


def get_supabase() -> Client:
    """Get or create the Supabase client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env"
            )
        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        logger.info("supabase.connected", url=settings.supabase_url)
    return _client


async def check_db_health() -> bool:
    """Check if Supabase is reachable."""
    try:
        client = get_supabase()
        result = client.table("recovery_settings").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error("supabase.health_check_failed", error=str(e))
        return False
