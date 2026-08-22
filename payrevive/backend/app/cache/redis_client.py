"""Redis client — Feature Store, Circuit Breaker, Idempotency, Job Broker."""

import redis.asyncio as redis
from app.config import get_settings

settings = get_settings()

redis_pool = redis.ConnectionPool.from_url(
    settings.redis_url,
    max_connections=20,
    decode_responses=True,
)

redis_client = redis.Redis(connection_pool=redis_pool)


async def get_redis() -> redis.Redis:
    """FastAPI dependency — returns redis client."""
    return redis_client


async def check_redis_health() -> bool:
    """Check if Redis is reachable."""
    try:
        return await redis_client.ping()
    except Exception:
        return False
