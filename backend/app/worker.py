"""ARQ Worker — background task processing for payment recovery."""

from arq.connections import RedisSettings
from arq import create_pool
from app.config import get_settings
from app.db.database import get_supabase
from app.models.schemas import (
    ErrorSource,
    FailedPayment,
    PaymentMethod,
    RecoveryStrategy,
)
from app.cache.feature_store import feature_extractor
from app.ml.classifier import classifier
from app.strategy.bandit import bandit
from app.strategy.llm_reasoner import llm_reasoner
from app.execution.compliance import IST, compliance_engine
from app.execution.executor import executor
from app.audit.event_store import event_store
from datetime import datetime, timedelta
import structlog
import traceback

logger = structlog.get_logger()
settings = get_settings()


async def process_failed_payment(ctx, payment_id: str):
    """
    Full recovery pipeline for a single payment.
    Triggered via webhook when a payment fails.
    """
    logger.info("worker.processing_start", payment_id=payment_id)
    db = get_supabase()

    # 1. Fetch payment data
    p_result = db.table("payments").select("*").eq("payment_id", payment_id).single().execute()
    if not p_result.data:
        logger.error("worker.payment_not_found", payment_id=payment_id)
        return

    p = p_result.data
    payment = FailedPayment(
        payment_id=p["payment_id"],
        order_id=p.get("order_id"),
        amount=p["amount"],
        currency=p.get("currency", "INR"),
        method=PaymentMethod(p["method"]),
        bank=p.get("bank"),
        wallet=p.get("wallet"),
        vpa=p.get("vpa"),
        error_code=p.get("error_code", "UNKNOWN"),
        error_source=ErrorSource(p.get("error_source", "gateway")),
        error_step=p.get("error_step", "unknown"),
        error_reason=p.get("error_reason", "unknown"),
        error_description=p.get("error_description", ""),
        customer_contact=p.get("customer_contact"),
        customer_email=p.get("customer_email"),
        is_recurring=p.get("is_recurring", False),
        created_at=datetime.fromisoformat(p["created_at"]) if isinstance(p["created_at"], str) else p["created_at"],
    )

    # 2. Check if there's an open session, create one if not
    s_result = db.table("recovery_sessions").select("*").eq("payment_id", payment_id).execute()
    
    if s_result.data:
        session = s_result.data[0]
        if session["status"] != "OPEN":
            logger.info("worker.session_closed", payment_id=payment_id, status=session["status"])
            return
        session_id = session["id"]
        retry_count = session.get("retry_count", 0)
    else:
        new_session = db.table("recovery_sessions").insert({
            "payment_id": payment_id,
            "status": "OPEN",
        }).execute()
        session_id = new_session.data[0]["id"]
        retry_count = 0
        event_store.log(session_id, payment_id, "INGESTED")

    try:
        # 3. Extract Features
        features = await feature_extractor.extract(payment)

        # 4. Classify Root Cause (if not already classified)
        classification = None
        if classifier.is_loaded():
            classification = classifier.predict(features)
            event_store.log_classification(
                session_id, payment_id, classification.root_cause.value,
                classification.confidence,
                [e.model_dump() for e in classification.shap_explanations],
                classification.explanation_summary
            )
            
            db.table("recovery_sessions").update({
                "root_cause": classification.root_cause.value,
                "root_cause_confidence": classification.confidence,
                "shap_explanation": [e.model_dump() for e in classification.shap_explanations[:5]],
            }).eq("id", session_id).execute()
        else:
            logger.warning("worker.classifier_not_loaded", payment_id=payment_id)
            return  # Can't proceed without classification

        # 5. Select Strategy
        # LLM override for high-value / low-confidence
        decision = None
        if (
            payment.amount > settings.llm_amount_threshold_paise or
            classification.confidence < settings.llm_confidence_threshold
        ):
            bank_health = await feature_extractor.store.get_bank_health(payment.bank or "unknown")
            decision = await llm_reasoner.reason(payment, classification, bank_health)
        
        if not decision:
            decision = await bandit.select_strategy(classification.root_cause.value, features)

        event_store.log_strategy(
            session_id, payment_id, decision.strategy.value, decision.decided_by,
            decision.reasoning, decision.confidence
        )

        db.table("recovery_sessions").update({
            "strategy": decision.strategy.value,
            "decided_by": decision.decided_by,
        }).eq("id", session_id).execute()

        # 6. Compliance Check
        compliance = await compliance_engine.check(payment, decision, retry_count)
        event_store.log_compliance_check(session_id, payment_id, compliance.approved, compliance.blocked_by)

        if not compliance.approved:
            await _honour_the_remedy(
                db, session_id, payment, decision, compliance, retry_count
            )
            return

        # 7. Execute Strategy
        executed = await executor.execute(payment, session_id, decision)
        
        if executed:
            # Increment retry count
            db.table("recovery_sessions").update({
                "retry_count": retry_count + 1
            }).eq("id", session_id).execute()
            
            if decision.strategy.value == "ESCALATE":
                db.table("recovery_sessions").update({
                    "status": "ESCALATED",
                    "closed_at": datetime.utcnow().isoformat()
                }).eq("id", session_id).execute()
        else:
            logger.error("worker.execution_failed", payment_id=payment_id, strategy=decision.strategy.value)
            db.table("recovery_sessions").update({
                "status": "FAILED",
                "closed_at": datetime.utcnow().isoformat()
            }).eq("id", session_id).execute()

    except Exception as e:
        logger.error("worker.pipeline_error", payment_id=payment_id, error=str(e), trace=traceback.format_exc())
        event_store.log_exception(session_id, payment_id, str(e), "PIPELINE_ERROR")


def _minutes_until(recommendation: str) -> int | None:
    """How long to sleep before trying this payment again, or `None` if not a wait.

    Quiet hours end at 08:00 IST and the daily contact budget resets at IST midnight,
    so both are answered by looking at the wall clock rather than by a fixed backoff.
    A fixed backoff would either wake up inside quiet hours again or waste most of the
    recovery window sleeping through the morning it was allowed to act in.
    """
    settings_ = get_settings()
    now = datetime.now(IST)
    if recommendation == "WAIT_FOR_INTERVAL":
        return settings_.min_retry_interval_minutes
    if recommendation == "DEFER_TO_MORNING":
        target = now.replace(hour=settings_.quiet_hours_end, minute=5, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return max(1, int((target - now).total_seconds() // 60))
    if recommendation == "WAIT_NEXT_DAY":
        target = (now + timedelta(days=1)).replace(
            hour=settings_.quiet_hours_end, minute=5, second=0, microsecond=0
        )
        return max(1, int((target - now).total_seconds() // 60))
    return None


def _switched_rail(payment: FailedPayment, decision) -> PaymentMethod | None:
    """A rail that can carry this amount, when the one we picked cannot.

    The ceiling that blocked us is the rail's, not the customer's, so the remedy is a
    different rail rather than a smaller ambition or a phone call. Cards and netbanking
    have no ceiling we can see, which is exactly why they are the fallback: we know UPI
    will refuse a lakh and we do not know that an issuer will.
    """
    from app.execution.compliance import method_ceiling_paise

    current = decision.preferred_method or payment.method
    for candidate in (PaymentMethod.CARD, PaymentMethod.NETBANKING, PaymentMethod.UPI):
        if candidate is current:
            continue
        ceiling = method_ceiling_paise(candidate)
        if ceiling is None or payment.amount <= ceiling:
            return candidate
    return None


async def _requeue(payment_id: str, minutes: int) -> bool:
    """Put the payment back on the queue to be reconsidered later.

    Re-runs the whole pipeline rather than replaying the blocked action, because by the
    time it wakes up the bank's health, the contact ledger and the elapsed-time bucket
    have all moved — and the right action then is not necessarily the one that was
    refused now.
    """
    try:
        arq_redis = await create_pool(parse_redis_url(settings.redis_url))
        await arq_redis.enqueue_job(
            "process_failed_payment", payment_id, _defer_by=minutes * 60
        )
        return True
    except Exception as exc:
        logger.error("worker.requeue_failed", payment_id=payment_id, error=str(exc))
        return False


async def _honour_the_remedy(
    db, session_id: str, payment: FailedPayment, decision, compliance, retry_count: int = 0
) -> None:
    """Do what the compliance engine said to do instead.

    A block is almost never "never". It is "not now" or "not that way", and the engine
    names which — `DEFER_TO_MORNING`, `WAIT_FOR_INTERVAL`, `SWITCH_METHOD`,
    `SWITCH_TO_PAYMENT_LINK`, one per rule. This used to read the recommendation, check
    it against `ESCALATE_TO_MERCHANT` (which the engine never returns) and `LOG_EXCEPTION`,
    and close the session as FAILED for everything else. Five of the seven remedies were
    discarded, and the payments they applied to were written off at the moment the system
    had been told exactly how to recover them.

    The same defect, measured in the simulator, cost 35 of the largest failures in a
    400-payment batch. A rule whose remedy is ignored is indistinguishable from a rule
    that just deletes money.

    Every substituted action is re-checked before it executes, and `retry_count` is
    threaded through so that re-check sees the same history the first one did. None of
    the three substitutions is a retry today, so the retry limits do not currently bite
    — but a remedy that skipped the gate on its way to fixing a gate violation is the
    exact shape of the bug this function exists to remove, and the substitute strategies
    are a table in `compliance.STRATEGY_ACTIONS` that someone will eventually edit.
    """
    payment_id = payment.payment_id
    remedy = compliance.recommendation
    reason = ", ".join(compliance.blocked_by)

    minutes = _minutes_until(remedy)
    if minutes is not None:
        queued = await _requeue(payment_id, minutes)
        wake_at = datetime.utcnow() + timedelta(minutes=minutes)
        # The wake-up time goes in the audit trail rather than on the session row.
        # `recovery_sessions` has no column for it, and this is a thing that happened at
        # a moment — which is what the append-only trail is for — rather than state to
        # be queried. The dashboard reads the latest event for the countdown.
        event_store.log(
            session_id, payment_id, "COMPLIANCE_REMEDY",
            {
                "recommendation": remedy,
                "blocked_by": compliance.blocked_by,
                "defer_minutes": minutes,
                "wake_at": wake_at.isoformat(),
                "queued": queued,
            },
        )
        db.table("recovery_sessions").update({
            # Deliberately still OPEN. The payment has not failed; it is asleep.
            "status": "OPEN" if queued else "FAILED",
        }).eq("id", session_id).execute()
        logger.info(
            "worker.deferred", payment_id=payment_id, remedy=remedy, minutes=minutes,
            queued=queued,
        )
        return

    event_store.log(
        session_id, payment_id, "COMPLIANCE_REMEDY",
        {"recommendation": remedy, "blocked_by": compliance.blocked_by},
    )

    if remedy == "SWITCH_METHOD":
        rail = _switched_rail(payment, decision)
        if rail is not None:
            decision.preferred_method = rail
            decision.strategy = RecoveryStrategy.LINK_ALT_METHOD
            decision.reasoning = (
                f"{reason} — offering {rail.value} instead, which can carry the amount"
            )
            recheck = await compliance_engine.check(payment, decision, retry_count)
            if recheck.approved and await executor.execute(payment, session_id, decision):
                db.table("recovery_sessions").update({
                    "strategy": decision.strategy.value,
                }).eq("id", session_id).execute()
                logger.info("worker.switched_rail", payment_id=payment_id, rail=rail.value)
                return

    if remedy == "SWITCH_TO_PAYMENT_LINK":
        decision.strategy = RecoveryStrategy.LINK_SAME_METHOD
        decision.reasoning = f"{reason} — asking the customer instead of retrying"
        recheck = await compliance_engine.check(payment, decision, retry_count)
        if recheck.approved and await executor.execute(payment, session_id, decision):
            db.table("recovery_sessions").update({
                "strategy": decision.strategy.value,
            }).eq("id", session_id).execute()
            logger.info("worker.switched_to_link", payment_id=payment_id)
            return

    if remedy in ("ESCALATE_TO_AGENT", "ESCALATE"):
        # `ESCALATE` is `_recommendation`'s default — the answer when a block matched none
        # of its patterns. No rule reaches it today, and that is exactly why it is handled
        # here: the next rule someone adds will arrive with a message the ladder does not
        # recognise, and the failure mode of leaving it out is not an error but a silent
        # write-off. A block we cannot name is still a payment a human could work.
        decision.strategy = RecoveryStrategy.ESCALATE
        decision.reasoning = f"{reason} — no automated route left"
        recheck = await compliance_engine.check(payment, decision, retry_count)
        if recheck.approved and await executor.execute(payment, session_id, decision):
            db.table("recovery_sessions").update({
                "status": "ESCALATED",
                "strategy": decision.strategy.value,
                "closed_at": datetime.utcnow().isoformat(),
            }).eq("id", session_id).execute()
            return

    if remedy == "LOG_EXCEPTION":
        # The only remedy that really does mean "never". The 72-hour recovery window has
        # closed, so there is no later moment to wake up in and no rail that changes the
        # answer. Named explicitly rather than left to fall off the end of this function,
        # because "the window expired" and "we did not recognise this instruction" are
        # different facts and a reviewer reading the exception list needs to see which.
        _write_off(db, session_id, payment_id, f"recovery window closed: {reason}")
        return

    # Either the remedy had no route left, the recheck refused the substitute, or the
    # substitute failed to execute. A real outcome rather than an error — recorded as an
    # exception so it appears in the list a human works, instead of dissolving into a
    # recovery rate where an abandoned payment reads the same as an unrecoverable one.
    _write_off(
        db, session_id, payment_id,
        f"compliance block with no reachable remedy: {reason}",
    )
    logger.info("worker.no_remedy", payment_id=payment_id, remedy=remedy, reason=reason)


def _write_off(db, session_id: str, payment_id: str, reason: str) -> None:
    """Close a session that has genuinely run out of legal moves."""
    event_store.log_exception(session_id, payment_id, reason, "COMPLIANCE")
    db.table("recovery_sessions").update({
        "status": "FAILED",
        "closed_at": datetime.utcnow().isoformat(),
    }).eq("id", session_id).execute()


async def execute_delayed_retry(ctx, payment_id: str, session_id: str):
    """Execute a delayed retry after bank recovery (invoked by ARQ)."""
    logger.info("worker.executing_delayed_retry", payment_id=payment_id, session_id=session_id)
    # 1. Check if payment is still pending
    db = get_supabase()
    s_result = db.table("recovery_sessions").select("status").eq("id", session_id).single().execute()
    
    if s_result.data and s_result.data["status"] != "OPEN":
        logger.info("worker.delayed_retry_aborted", reason="Session no longer OPEN", session_id=session_id)
        return
        
    event_store.log_recovery_attempt(session_id, payment_id, "DELAYED_RETRY_EXECUTED")
    # Simulation: In a real app, charge the tokenized card.


async def poll_bank_downtimes(ctx):
    """Poll Razorpay Downtime API for bank health updates (Cron)."""
    logger.info("worker.polling_downtimes")
    # Simulation: Would hit Razorpay API and update feature_store.set_bank_downtime


async def startup(ctx):
    logger.info("worker.started")
    # Load ML model on startup if not already loaded
    if not classifier.is_loaded():
        classifier.load()


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
    
    # Simple cron jobs (ARQ supports this)
    # cron_jobs = [cron(poll_bank_downtimes, minute=set(range(0, 60, 5)))]
