"""ARQ Worker — background task processing."""
from arq.connections import RedisSettings
from app.config import get_settings
import structlog

logger = structlog.get_logger()

settings = get_settings()


async def process_failed_payment(ctx, payment_id: str):
    """Process a failed payment through the recovery pipeline."""
    logger.info("worker.processing", payment_id=payment_id)
    # Full pipeline will be implemented here
    # 1. Extract features
    # 2. Classify root cause
    # 3. Select strategy
    # 4. Execute recovery
    # 5. Log audit events


async def execute_delayed_retry(ctx, payment_id: str, session_id: str):
    """Execute a delayed retry after bank recovery."""
    logger.info("worker.delayed_retry", payment_id=payment_id, session_id=session_id)


async def poll_bank_downtimes(ctx):
    """Poll Razorpay Downtime API for bank health updates."""
    logger.info("worker.polling_downtimes")


async def startup(ctx):
    logger.info("worker.started")


async def shutdown(ctx):
    logger.info("worker.stopped")


# Parse Redis URL for ARQ settings
def parse_redis_url(url: str) -> RedisSettings:
    """Parse redis:// URL into ARQ RedisSettings."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        database=int(parsed.path.lstrip("/") or 0),
    )


class WorkerSettings:
    functions = [process_failed_payment, execute_delayed_retry, poll_bank_downtimes]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = parse_redis_url(settings.redis_url)
    max_jobs = 10
    job_timeout = 300  # 5 minutes
