"""The one place that puts a job on the queue.

Both the worker and the executor need to defer work: the worker when compliance says "not
now" and names a wake-up time, the executor when the chosen strategy is a delayed retry. The
worker had a private helper for it and the executor had the call commented out, which is why
`RETRY_SCHEDULED` appeared in the audit trail for retries that were never scheduled.

They cannot share the worker's helper directly — `app.worker` imports `app.execution.executor`,
so an import back the other way is a cycle. Hence this module, which imports nothing from
either.

Enqueueing is treated as fallible rather than assumed. Redis Cloud is a network hop, and a
deferral that silently failed to enqueue would leave a session `OPEN` forever, waiting for a
wake-up that no longer exists — the worst of the three outcomes, because it looks like
patience.
"""

from __future__ import annotations

from urllib.parse import urlparse

import structlog
from arq import create_pool
from arq.connections import RedisSettings

from app.config import get_settings

logger = structlog.get_logger()


def parse_redis_url(url: str) -> RedisSettings:
    """Parse a `redis://` URL into ARQ's settings object."""
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        database=int(parsed.path.lstrip("/") or 0),
    )


async def enqueue(job: str, *args, defer_minutes: int = 0) -> bool:
    """Queue `job`, optionally deferred. Returns whether it was actually queued.

    The return value is the point. Every caller has a different correct response to a queue
    that is unreachable — the worker writes the session off as an exception rather than
    leaving it asleep, the executor reports the action as not taken so the strategy is not
    credited — and neither can decide that if this raises or returns nothing.
    """
    try:
        pool = await create_pool(parse_redis_url(get_settings().redis_url))
        await pool.enqueue_job(job, *args, _defer_by=defer_minutes * 60)
        logger.info("queue.enqueued", job=job, defer_minutes=defer_minutes)
        return True
    except Exception as exc:  # noqa: BLE001 — any failure to queue is the same failure
        logger.error("queue.enqueue_failed", job=job, error=str(exc))
        return False
