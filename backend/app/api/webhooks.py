"""Razorpay Webhook handler — ingests payment events.

Two directions, and the second one is what makes the learning loop real.

`payment.failed` opens a recovery session and queues the pipeline. `payment.captured`
and `payment_link.paid` close it with an outcome the system *observed* rather than
assumed — which is the only honest source of a bandit reward. Before this handler
existed the batch endpoint filled the gap by drawing `random.random()` against a table
of recovery rates keyed on the classifier's own prediction, and fed that to
`bandit.update()`. A learning loop whose reward is a coin flip correlated with its own
output is not learning; it converges on whatever the classifier is confident about and
reports the result as measured performance.
"""

import hmac
import hashlib
from fastapi import APIRouter, Request, HTTPException
from app.config import get_settings
from app.cache.redis_client import redis_client
from app.db.database import get_supabase
from app.execution import attribution
from app.models.schemas import FailedPayment, PaymentMethod, ErrorSource
from datetime import datetime
from arq import create_pool
from app.worker import parse_redis_url
import structlog

logger = structlog.get_logger()
router = APIRouter()

RECOVERY_EVENTS = ("payment.captured", "payment_link.paid")
"""The two ways a failure we were working on can turn into money."""


@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    """
    Receive Razorpay webhook events.
    
    1. Verify HMAC-SHA256 signature
    2. Check idempotency (Redis SET NX)
    3. Normalize the event
    4. Store in Supabase
    5. Queue for processing
    """
    settings = get_settings()
    body = await request.body()

    # 1. Verify signature
    signature = request.headers.get("X-Razorpay-Signature", "")
    if settings.razorpay_webhook_secret:
        expected = hmac.new(
            settings.razorpay_webhook_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            logger.warning("webhook.invalid_signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    event = payload.get("event", "")
    entity = payload.get("payload", {}).get("payment", {}).get("entity", {})

    logger.info("webhook.received", event=event, payment_id=entity.get("id"))

    if event in RECOVERY_EVENTS:
        return await _record_recovery(event, payload)

    # Only handle payment.failed events
    if event != "payment.failed":
        return {"status": "ignored", "event": event}

    payment_id = entity.get("id")
    if not payment_id:
        return {"status": "ignored", "reason": "no payment_id"}

    # 2. Idempotency check
    is_new = await redis_client.set(
        f"idempotency:{payment_id}", "1", nx=True, ex=259200  # 72h TTL
    )
    if not is_new:
        logger.info("webhook.duplicate", payment_id=payment_id)
        return {"status": "duplicate", "payment_id": payment_id}

    # 3. Normalize
    error_data = entity.get("error", {})
    method_raw = entity.get("method", "upi")
    if method_raw not in [m.value for m in PaymentMethod]:
        method_raw = "upi"

    source_raw = error_data.get("source", "gateway")
    if source_raw not in [s.value for s in ErrorSource]:
        source_raw = "gateway"

    payment_record = {
        "payment_id": payment_id,
        "order_id": entity.get("order_id"),
        "amount": entity.get("amount", 0),
        "currency": entity.get("currency", "INR"),
        "method": method_raw,
        "bank": entity.get("bank"),
        "wallet": entity.get("wallet"),
        "vpa": entity.get("vpa"),
        "error_code": error_data.get("code", "UNKNOWN"),
        "error_source": source_raw,
        "error_step": error_data.get("step", "unknown"),
        "error_reason": error_data.get("reason", "unknown"),
        "error_description": error_data.get("description", ""),
        "customer_contact": entity.get("contact"),
        "customer_email": entity.get("email"),
        "is_recurring": entity.get("recurring", False),
        "raw_webhook": payload,
        "created_at": datetime.utcnow().isoformat(),
    }

    # 4. Store in Supabase
    try:
        db = get_supabase()
        db.table("payments").insert(payment_record).execute()
        logger.info("webhook.stored", payment_id=payment_id)
    except Exception as e:
        logger.error("webhook.store_failed", payment_id=payment_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to store payment")

    # 5. Queue for processing via ARQ worker
    try:
        arq_redis = await create_pool(parse_redis_url(settings.redis_url))
        await arq_redis.enqueue_job("process_failed_payment", payment_id)
        logger.info("webhook.enqueued", payment_id=payment_id)
    except Exception as e:
        logger.error("webhook.enqueue_failed", payment_id=payment_id, error=str(e))
        # Don't fail the request if queueing fails, we have it in DB

    return {"status": "accepted", "payment_id": payment_id}


def _recovered_reference(event: str, payload: dict) -> tuple[str | None, bool]:
    """Which failed payment this success belongs to, and whether it came via our link.

    `payment_link.paid` carries the `reference_id` we set when creating the link, which
    is the id of the payment that failed — so the join is recorded, not guessed, and the
    attribution is `SYSTEM_RECOVERED` without inference.

    `payment.captured` is the customer paying however they liked. Razorpay gives it a
    fresh payment id, so the only thing tying it to the failure is the order, and that
    is the honest amount of certainty available: we know they paid, we do not know we
    caused it.
    """
    if event == "payment_link.paid":
        link = payload.get("payload", {}).get("payment_link", {}).get("entity", {})
        return link.get("reference_id"), True
    entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    return entity.get("order_id"), False


async def _record_recovery(event: str, payload: dict) -> dict:
    """Close an open session with an observed outcome, then let the bandit learn from it.

    Ordering matters here. The bandit is updated last and only from a verdict that
    survived attribution, so an arm's posterior can never move on a recovery the system
    could not prove it caused.
    """
    reference, via_our_link = _recovered_reference(event, payload)
    if not reference:
        return {"status": "ignored", "reason": "no reference to a failed payment"}

    db = get_supabase()
    # A link's reference_id is the failed payment id; a bare capture only knows the
    # order. Either way we are looking for a session we opened and have not closed.
    column = "payment_id" if via_our_link else "order_id"
    if column == "order_id":
        matches = db.table("payments").select("payment_id").eq("order_id", reference).execute()
        candidates = [row["payment_id"] for row in (matches.data or [])]
    else:
        candidates = [reference]
    if not candidates:
        return {"status": "ignored", "reason": "no failed payment matches"}

    sessions = (
        db.table("recovery_sessions")
        .select("*")
        .in_("payment_id", candidates)
        .neq("status", "RECOVERED")
        .execute()
    )
    if not sessions.data:
        return {"status": "ignored", "reason": "no open session"}
    session = sessions.data[0]
    payment_id = session["payment_id"]

    # Idempotency on the *outcome*, not just the event: two webhooks for the same
    # recovery must not double-count the money or update the bandit twice.
    is_new = await redis_client.set(
        f"recovered:{payment_id}", "1", nx=True, ex=259200
    )
    if not is_new:
        return {"status": "duplicate", "payment_id": payment_id}

    paid_at = datetime.utcnow()
    last_contact = await _last_contact_at(payment_id)
    verdict, why = attribution.attribute(
        paid_at=paid_at, via_our_link=via_our_link, last_contact_at=last_contact
    )

    amount = 0
    try:
        row = db.table("payments").select("amount").eq("payment_id", payment_id).single().execute()
        amount = (row.data or {}).get("amount", 0)
    except Exception:
        pass

    db.table("recovery_sessions").update({
        "status": "RECOVERED",
        # Only money the system can prove it caused is booked against it. The rest is
        # recorded as recovered — it really did arrive — with the credit withheld.
        "amount_recovered": amount if verdict == attribution.SYSTEM_RECOVERED else 0,
        "attribution": verdict,
        "closed_at": paid_at.isoformat(),
    }).eq("id", session["id"]).execute()

    db.table("audit_events").insert({
        "recovery_session_id": session["id"],
        "payment_id": payment_id,
        "event_type": "RECOVERY_OBSERVED",
        "event_data": {
            "event": event,
            "attribution": verdict,
            "reasoning": why,
            "amount": amount,
            "claimed": verdict == attribution.SYSTEM_RECOVERED,
            "hours_since_last_contact": (
                None if last_contact is None
                else (paid_at - last_contact).total_seconds() / 3600.0
            ),
        },
    }).execute()

    learned = await _teach_bandit(session, verdict)
    logger.info(
        "webhook.recovery_recorded",
        payment_id=payment_id,
        attribution=verdict,
        bandit_updated=learned,
    )
    return {
        "status": "recorded",
        "payment_id": payment_id,
        "attribution": verdict,
        "reasoning": why,
        "bandit_updated": learned,
    }


async def _last_contact_at(payment_id: str) -> datetime | None:
    """When we last messaged or phoned this customer, from the audit trail.

    Read from `audit_events` rather than a counter, because the attribution window is a
    question about a specific instant and a counter only knows totals. The events it
    looks for are the ones `executor.py` writes when a contact actually goes out, so a
    decision that was blocked before execution correctly leaves no trace here.
    """
    db = get_supabase()
    try:
        result = (
            db.table("audit_events")
            .select("created_at")
            .eq("payment_id", payment_id)
            .in_("event_type", ["PAYMENT_LINK_SENT", "ESCALATION"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("webhook.contact_lookup_failed", payment_id=payment_id, error=str(exc))
        return None
    if not result.data:
        return None
    raw = result.data[0]["created_at"]
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None)
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=None)


async def _teach_bandit(session: dict, verdict: str) -> bool:
    """Update the arm that made this decision, if the outcome says anything about it.

    Returns whether the posterior actually moved. An ambiguous recovery deliberately
    teaches nothing: `attribution.reward` returns `None`, and a Beta updated with a
    zero would have been told the arm failed rather than told nothing.

    The features are re-extracted rather than stored, which is a real approximation and
    worth naming: bank health has moved since the decision, so a contextual arm is
    updated against slightly different context than it chose under. Storing the feature
    vector on the session at decision time is the fix, and it is a schema change.
    """
    value = attribution.reward(verdict)
    strategy = session.get("strategy")
    root_cause = session.get("root_cause")
    if value is None or not strategy or not root_cause:
        return False
    try:
        from app.cache.feature_store import feature_extractor
        from app.strategy.bandit import bandit

        payment = await _rebuild_payment(session["payment_id"])
        if payment is None:
            return False
        features = await feature_extractor.extract(payment)
        await bandit.update(
            root_cause=root_cause,
            features=features,
            strategy=strategy,
            reward=value,
        )
        return True
    except Exception as exc:
        logger.error(
            "webhook.bandit_update_failed",
            payment_id=session.get("payment_id"),
            error=str(exc),
        )
        return False


async def _rebuild_payment(payment_id: str) -> FailedPayment | None:
    """Rehydrate the failure record so features can be recomputed for the update."""
    db = get_supabase()
    try:
        result = db.table("payments").select("*").eq("payment_id", payment_id).single().execute()
    except Exception:
        return None
    p = result.data
    if not p:
        return None
    created = p["created_at"]
    return FailedPayment(
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
        created_at=datetime.fromisoformat(created) if isinstance(created, str) else created,
    )
