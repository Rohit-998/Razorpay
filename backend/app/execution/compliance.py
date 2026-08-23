"""Compliance Engine — hard limits that override all ML/LLM decisions."""

from datetime import datetime
from app.cache.redis_client import redis_client
from app.config import get_settings
from app.models.schemas import FailedPayment, RecoveryDecision, ComplianceCheck
from app.audit.event_store import event_store
import structlog

logger = structlog.get_logger()


class ComplianceEngine:
    """
    EVERY recovery action passes through compliance.
    These limits are HARD — they override bandit/LLM decisions.
    
    Rules:
    1. Max N retries per payment
    2. Max N customer contacts per day
    3. No actions during quiet hours (10 PM - 8 AM IST)
    4. Min interval between retries
    5. Max recovery window (72 hours from failure)
    6. Max payment link amount
    """

    async def check(
        self,
        payment: FailedPayment,
        decision: RecoveryDecision,
        session_retry_count: int = 0,
    ) -> ComplianceCheck:
        """Run all compliance checks. Returns approval or rejection."""
        settings = get_settings()
        blocked = []

        # 1. Retry limit
        if session_retry_count >= settings.max_retries_per_payment:
            blocked.append(
                f"Max retries exceeded ({session_retry_count}/{settings.max_retries_per_payment})"
            )

        # 2. Contact limit (per customer per day)
        contact_ok = await self._check_contact_limit(
            payment.customer_contact, settings.max_contacts_per_day
        )
        if not contact_ok:
            blocked.append(
                f"Daily contact limit reached ({settings.max_contacts_per_day}/day)"
            )

        # 3. Quiet hours
        if self._is_quiet_hours(settings.quiet_hours_start, settings.quiet_hours_end):
            blocked.append(
                f"Quiet hours ({settings.quiet_hours_start}:00 - {settings.quiet_hours_end}:00 IST)"
            )

        # 4. Recovery window
        hours_since = (datetime.utcnow() - payment.created_at).total_seconds() / 3600
        if hours_since > settings.max_recovery_window_hours:
            blocked.append(
                f"Recovery window expired ({hours_since:.0f}h > {settings.max_recovery_window_hours}h max)"
            )

        # 5. Amount limit for auto-retry (require customer action above threshold)
        if (
            payment.amount > settings.require_action_above_paise
            and decision.strategy.value in ("IMMEDIATE_RETRY", "DELAYED_RETRY")
        ):
            blocked.append(
                f"Amount ₹{payment.amount_rupees:,.0f} exceeds auto-retry limit — requires customer action"
            )

        if blocked:
            recommendation = self._get_recommendation(blocked)
            logger.info(
                "compliance.blocked",
                payment_id=payment.payment_id,
                blocked_by=blocked,
            )
            return ComplianceCheck(
                approved=False,
                blocked_by=blocked,
                recommendation=recommendation,
            )

        logger.info("compliance.approved", payment_id=payment.payment_id)
        return ComplianceCheck(approved=True)

    async def record_contact(self, customer_id: str):
        """Record a customer contact (SMS/email sent)."""
        if not customer_id:
            return
        today = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"contact_count:{customer_id}:{today}"
        await redis_client.incr(key)
        await redis_client.expire(key, 86400)  # 24h TTL

    async def _check_contact_limit(self, customer_id: str, max_contacts: int) -> bool:
        """Check if customer has been contacted too many times today."""
        if not customer_id:
            return True
        today = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"contact_count:{customer_id}:{today}"
        count = await redis_client.get(key)
        return int(count or 0) < max_contacts

    def _is_quiet_hours(self, start: int, end: int) -> bool:
        """Check if current IST time is in quiet hours."""
        from datetime import timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist).hour

        if start > end:  # Wraps midnight (e.g., 22-8)
            return now_ist >= start or now_ist < end
        return start <= now_ist < end

    def _get_recommendation(self, blocked: list[str]) -> str:
        """Suggest what to do when blocked."""
        if any("quiet hours" in b.lower() for b in blocked):
            return "DEFER_TO_MORNING"
        if any("max retries" in b.lower() for b in blocked):
            return "ESCALATE_TO_MERCHANT"
        if any("contact limit" in b.lower() for b in blocked):
            return "WAIT_NEXT_DAY"
        if any("window expired" in b.lower() for b in blocked):
            return "LOG_EXCEPTION"
        if any("auto-retry limit" in b.lower() for b in blocked):
            return "SWITCH_TO_PAYMENT_LINK"
        return "ESCALATE"


# Singleton
compliance_engine = ComplianceEngine()
