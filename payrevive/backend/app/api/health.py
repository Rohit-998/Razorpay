"""Health check endpoint."""

from fastapi import APIRouter
from app.cache.redis_client import check_redis_health
from app.db.database import check_db_health

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check API, database, and Redis connectivity."""
    redis_ok = await check_redis_health()
    db_ok = await check_db_health()

    return {
        "status": "healthy" if (redis_ok and db_ok) else "degraded",
        "services": {
            "api": True,
            "database": db_ok,
            "redis": redis_ok,
        },
    }
