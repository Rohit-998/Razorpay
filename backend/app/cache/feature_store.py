"""Feature Store — Redis-backed pre-computed features for <10ms classifier inference."""

import time
from datetime import datetime
from typing import Optional
from app.cache.redis_client import redis_client
from app.models.schemas import FailedPayment, FeatureVector, BankHealthStatus
import structlog

logger = structlog.get_logger()


class FeatureStore:
    """
    Pre-computes and caches features in Redis Hashes.
    
    Key patterns:
      bank_health:{bank_code}   → success_rate_1h, failure_count_1h, is_downtime, ...
      customer:{customer_id}    → success_rate_30d, preferred_method, ...
      method_health:{method}    → success_rate_1h, success_rate_24h
    """

    # ── Bank Health ──────────────────────────────────────────

    async def update_bank_event(self, bank: str, success: bool):
        """Track payment outcome for bank health calculation."""
        if not bank:
            return
        
        timestamp = time.time()
        event_type = "success" if success else "failure"
        key = f"bank_events:{bank}:{event_type}"

        # Add to sorted set (score = timestamp)
        await redis_client.zadd(key, {f"{timestamp}:{event_type}": timestamp})
        # Remove events older than 24 hours
        await redis_client.zremrangebyscore(key, 0, timestamp - 86400)
        # Recalculate health
        await self._recalculate_bank_health(bank)

    async def _recalculate_bank_health(self, bank: str):
        """Recalculate rolling success rates for a bank."""
        now = time.time()
        one_hour_ago = now - 3600

        success_key = f"bank_events:{bank}:success"
        failure_key = f"bank_events:{bank}:failure"

        # 1-hour window
        successes_1h = await redis_client.zcount(success_key, one_hour_ago, now)
        failures_1h = await redis_client.zcount(failure_key, one_hour_ago, now)
        total_1h = successes_1h + failures_1h

        rate_1h = successes_1h / total_1h if total_1h > 0 else 0.95

        await redis_client.hset(f"bank_health:{bank}", mapping={
            "success_rate_1h": str(rate_1h),
            "failure_count_1h": str(failures_1h),
            "total_1h": str(total_1h),
            "last_updated": str(now),
        })

    async def set_bank_downtime(self, bank: str, severity: str, is_down: bool):
        """Update bank downtime status from Razorpay Downtime API."""
        await redis_client.hset(f"bank_health:{bank}", mapping={
            "is_downtime": str(is_down).lower(),
            "downtime_severity": severity if is_down else "none",
        })

    async def get_bank_health(self, bank: str) -> BankHealthStatus:
        """Get current bank health from Redis."""
        if not bank:
            return BankHealthStatus(bank_code="unknown", is_healthy=True)

        data = await redis_client.hgetall(f"bank_health:{bank}")
        if not data:
            return BankHealthStatus(bank_code=bank, is_healthy=True)

        success_rate = float(data.get("success_rate_1h", "0.95"))
        failure_count = int(data.get("failure_count_1h", "0"))
        is_downtime = data.get("is_downtime", "false") == "true"

        return BankHealthStatus(
            bank_code=bank,
            is_healthy=success_rate > 0.5 and not is_downtime,
            success_rate_1h=success_rate,
            failure_count_1h=failure_count,
            is_in_downtime=is_downtime,
            downtime_severity=data.get("downtime_severity"),
            recommendation="WAIT" if is_downtime or success_rate < 0.5 else "RETRY_NOW",
        )

    # ── Customer History ─────────────────────────────────────

    async def update_customer_event(self, customer_id: str, success: bool, method: str):
        """Update customer payment history."""
        if not customer_id:
            return

        key = f"customer:{customer_id}"
        data = await redis_client.hgetall(key)

        total = int(data.get("total_payments", "0")) + 1
        successes = int(data.get("total_successes", "0")) + (1 if success else 0)
        failures_7d = int(data.get("failure_count_7d", "0")) + (0 if success else 1)

        await redis_client.hset(key, mapping={
            "total_payments": str(total),
            "total_successes": str(successes),
            "success_rate_30d": str(successes / total if total > 0 else 0.9),
            "failure_count_7d": str(failures_7d),
            "preferred_method": method,
            "last_updated": str(time.time()),
        })
        await redis_client.expire(key, 86400 * 30)  # 30 day TTL

    async def get_customer_features(self, customer_id: str) -> dict:
        """Get customer history features from Redis."""
        if not customer_id:
            return {
                "customer_success_rate_30d": 0.9,
                "customer_failure_count_7d": 0,
                "customer_recovery_response": 0.5,
            }

        data = await redis_client.hgetall(f"customer:{customer_id}")
        if not data:
            return {
                "customer_success_rate_30d": 0.9,
                "customer_failure_count_7d": 0,
                "customer_recovery_response": 0.5,
            }

        return {
            "customer_success_rate_30d": float(data.get("success_rate_30d", "0.9")),
            "customer_failure_count_7d": int(data.get("failure_count_7d", "0")),
            "customer_recovery_response": float(data.get("recovery_response", "0.5")),
        }

    # ── Method Health ────────────────────────────────────────

    async def update_method_event(self, method: str, success: bool):
        """Track payment method health."""
        now = time.time()
        event_type = "success" if success else "failure"
        key = f"method_events:{method}:{event_type}"
        await redis_client.zadd(key, {f"{now}:{event_type}": now})
        await redis_client.zremrangebyscore(key, 0, now - 86400)

        # Recalculate
        one_hour_ago = now - 3600
        s = await redis_client.zcount(f"method_events:{method}:success", one_hour_ago, now)
        f = await redis_client.zcount(f"method_events:{method}:failure", one_hour_ago, now)
        total = s + f
        rate = s / total if total > 0 else 0.95

        await redis_client.hset(f"method_health:{method}", mapping={
            "success_rate_1h": str(rate),
        })

    async def get_method_health(self, method: str) -> float:
        """Get method success rate."""
        data = await redis_client.hgetall(f"method_health:{method}")
        return float(data.get("success_rate_1h", "0.95"))


class FeatureExtractor:
    """
    Extracts 17 features from a failed payment.
    Bank health + customer features come from Redis Feature Store (<5ms).
    """

    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store

    async def extract(self, payment: FailedPayment) -> FeatureVector:
        """Extract all 17 features for ML classification."""

        # 1. Payment Context Features (from the event itself)
        payment_features = {
            "error_source": payment.error_source.value,
            "error_step": payment.error_step,
            "error_reason": payment.error_reason,
            "payment_method": payment.method.value,
            "amount_bucket": payment.amount_bucket,
        }

        # 2. Temporal Features (from timestamp)
        temporal_features = self._extract_temporal(payment.created_at)

        # 3. Bank Health Features (from Redis — fast!)
        bank_health = await self.store.get_bank_health(payment.bank or "unknown")
        bank_features = {
            "bank_success_rate_1h": bank_health.success_rate_1h,
            "bank_failure_count_1h": bank_health.failure_count_1h,
            "bank_is_in_downtime": bank_health.is_in_downtime,
            "method_success_rate_1h": await self.store.get_method_health(payment.method.value),
        }

        # 4. Customer History Features (from Redis — fast!)
        customer_id = payment.customer_contact or payment.customer_email
        customer_features = await self.store.get_customer_features(customer_id)

        return FeatureVector(
            **payment_features,
            **temporal_features,
            **bank_features,
            **customer_features,
        )

    def _extract_temporal(self, dt: datetime) -> dict:
        """Extract temporal features from payment timestamp."""
        return {
            "hour_of_day": dt.hour,
            "day_of_week": dt.weekday(),
            "is_month_end": dt.day >= 25,
            "is_salary_window": dt.day <= 5,
            "is_maintenance_window": 0 <= dt.hour < 6,
        }


# Singleton instances
feature_store = FeatureStore()
feature_extractor = FeatureExtractor(feature_store)
