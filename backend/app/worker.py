"""ARQ Worker — background task processing for payment recovery."""

from arq.connections import RedisSettings
from app.config import get_settings
from app.db.database import get_supabase
from app.models.schemas import FailedPayment, PaymentMethod, ErrorSource
from app.cache.feature_store import feature_extractor
from app.ml.classifier import classifier
from app.strategy.bandit import bandit
from app.strategy.llm_reasoner import llm_reasoner
from app.execution.compliance import compliance_engine
from app.execution.executor import executor
from app.audit.event_store import event_store
from datetime import datetime
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
            # Handle compliance block
            db.table("recovery_sessions").update({
                "status": "ESCALATED" if compliance.recommendation == "ESCALATE_TO_MERCHANT" else "FAILED",
                "closed_at": datetime.utcnow().isoformat()
            }).eq("id", session_id).execute()
            
            if compliance.recommendation == "LOG_EXCEPTION":
                event_store.log_exception(session_id, payment_id, " compliance block: " + ", ".join(compliance.blocked_by), "COMPLIANCE")
            elif compliance.recommendation == "ESCALATE_TO_MERCHANT":
                event_store.log_escalation(session_id, payment_id, " compliance block: " + ", ".join(compliance.blocked_by))
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
