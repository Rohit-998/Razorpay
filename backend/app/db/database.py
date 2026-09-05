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


def select_all(
    table: str, columns: str, *, page_size: int = 1000, **filters: str
) -> list[dict]:
    """Every row of `table`, paged, because PostgREST caps an unbounded select.

    A `select()` with no `range()` returns at most `db-max-rows` — 1000 on Supabase's
    default configuration — and returns it without error, without a truncation flag, and
    without any indication that the answer is partial. For a list endpoint that is a
    missing page. For a *total*, it is a wrong number that looks right: an aggregate over
    the first thousand rows carries the label of an aggregate over all of them.

    This was not hypothetical. `audit_events` holds 1158 rows, and a census of event types
    run against it summed to exactly 1000 — the cap, mistaken for the population, which is
    the kind of arithmetic this project exists to stop shipping.

    `filters` are applied as equality filters, server side, so the cap is spent on rows
    that matter instead of on rows that are discarded locally.
    """
    db = get_supabase()
    rows: list[dict] = []
    offset = 0
    while True:
        query = db.table(table).select(columns)
        for column, value in filters.items():
            query = query.eq(column, value)
        page = query.range(offset, offset + page_size - 1).execute().data or []
        rows.extend(page)
        # A short page is the last page. An exactly-full final page costs one extra
        # round trip that comes back empty, which is the cheap half of the trade.
        if len(page) < page_size:
            return rows
        offset += page_size


async def check_db_health() -> bool:
    """Check if Supabase is reachable."""
    try:
        client = get_supabase()
        result = client.table("recovery_settings").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error("supabase.health_check_failed", error=str(e))
        return False
