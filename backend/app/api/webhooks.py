"""Razorpay Webhook handler — ingests payment.failed events."""

import hmac
import hashlib
from fastapi import APIRouter, Request, HTTPException
from app.config import get_settings
from app.cache.redis_client import redis_client
from app.db.database import get_supabase
from app.models.schemas import FailedPayment, PaymentMethod, ErrorSource
from datetime import datetime
from arq import create_pool
from app.worker import parse_redis_url
import structlog

logger = structlog.get_logger()
router = APIRouter()


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
