"""Batch operations — generate synthetic data, run pipeline, get report."""

from fastapi import APIRouter
from app.db.database import get_supabase
from app.synthetic.generator import generator
from app.cache.feature_store import feature_store
import structlog

logger = structlog.get_logger()
router = APIRouter()


@router.post("/batch/generate")
async def generate_synthetic_data(count: int = 150, duration_days: int = 7):
    """Generate synthetic failed payments and store in Supabase."""
    db = get_supabase()

    # Generate customers
    customers = generator.generate_customers(n=50)
    for c in customers:
        try:
            db.table("customers").upsert(c, on_conflict="customer_id").execute()
        except Exception:
            pass  # Skip duplicates

    # Generate failures
    payments, labels = generator.generate_failures(
        n=count, customers=customers, duration_days=duration_days
    )

    # Store in Supabase
    stored = 0
    for payment, label in zip(payments, labels):
        record = {
            "payment_id": payment.payment_id,
            "order_id": payment.order_id,
            "amount": payment.amount,
            "currency": payment.currency,
            "method": payment.method.value,
            "bank": payment.bank,
            "wallet": payment.wallet,
            "vpa": payment.vpa,
            "error_code": payment.error_code,
            "error_source": payment.error_source.value,
            "error_step": payment.error_step,
            "error_reason": payment.error_reason,
            "error_description": payment.error_description,
            "customer_contact": payment.customer_contact,
            "customer_email": payment.customer_email,
            "is_recurring": payment.is_recurring,
            "created_at": payment.created_at.isoformat(),
        }
        try:
            db.table("payments").insert(record).execute()
            stored += 1

            # Update feature store with synthetic bank health data
            await feature_store.update_bank_event(payment.bank, success=False)
            await feature_store.update_method_event(payment.method.value, success=False)
        except Exception as e:
            logger.warning("batch.insert_failed", payment_id=payment.payment_id, error=str(e))

    # Store labels mapping (for training)
    for payment, label in zip(payments, labels):
        try:
            db.table("recovery_sessions").insert({
                "payment_id": payment.payment_id,
                "status": "OPEN",
                "root_cause": label,
            }).execute()
        except Exception:
            pass

    logger.info("batch.generated", total=count, stored=stored)

    return {
        "status": "success",
        "generated": count,
        "stored": stored,
        "customers": len(customers),
        "distribution": {rc: labels.count(rc) for rc in set(labels)},
    }


@router.post("/batch/run")
async def run_batch_pipeline():
    """Run the full recovery pipeline on all OPEN sessions."""
    from app.ml.classifier import classifier
    from app.cache.feature_store import feature_extractor
    from app.strategy.bandit import bandit
    from app.models.schemas import FailedPayment, PaymentMethod, ErrorSource
    from datetime import datetime

    if not classifier.is_loaded():
        return {"status": "error", "message": "Model not trained. Run POST /api/v1/model/train first."}

    db = get_supabase()

    # Get all open sessions
    sessions = db.table("recovery_sessions").select("*").eq("status", "OPEN").execute()
    if not sessions.data:
        return {"status": "no_data", "message": "No open recovery sessions found."}

    results = {"total": len(sessions.data), "processed": 0, "recovered": 0, "failed": 0, "escalated": 0}

    for session in sessions.data:
        payment_data = db.table("payments").select("*").eq("payment_id", session["payment_id"]).single().execute()
        if not payment_data.data:
            continue

        p = payment_data.data
        try:
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

            # 1. Extract features
            features = await feature_extractor.extract(payment)

            # 2. Classify root cause
            classification = classifier.predict(features)

            # 3. Select strategy
            decision = await bandit.select_strategy(
                root_cause=classification.root_cause.value,
                features=features,
            )

            # 4. Simulate recovery outcome based on root cause recoverability
            import random
            recovery_rates = {
                "BANK_DOWNTIME": 0.84, "NETWORK_TRANSIENT": 0.86,
                "AUTH_TIMEOUT": 0.30, "INSUFFICIENT_FUNDS": 0.10,
                "WRONG_CREDENTIALS": 0.15, "PERMANENT_DECLINE": 0.0,
                "MERCHANT_ERROR": 0.0,
            }
            rate = recovery_rates.get(classification.root_cause.value, 0.0)
            recovered = random.random() < rate

            # 5. Update session in Supabase
            new_status = "RECOVERED" if recovered else ("ESCALATED" if decision.strategy.value == "ESCALATE" else "FAILED")
            shap_data = [e.model_dump() for e in classification.shap_explanations[:5]]

            db.table("recovery_sessions").update({
                "status": new_status,
                "root_cause": classification.root_cause.value,
                "root_cause_confidence": classification.confidence,
                "strategy": decision.strategy.value,
                "decided_by": decision.decided_by,
                "amount_recovered": payment.amount if recovered else 0,
                "shap_explanation": shap_data,
                "attribution": "SYSTEM_RECOVERED" if recovered else None,
                "closed_at": datetime.utcnow().isoformat(),
            }).eq("id", session["id"]).execute()

            # 6. Log audit event
            db.table("audit_events").insert({
                "recovery_session_id": session["id"],
                "payment_id": payment.payment_id,
                "event_type": "PIPELINE_COMPLETE",
                "event_data": {
                    "root_cause": classification.root_cause.value,
                    "confidence": classification.confidence,
                    "strategy": decision.strategy.value,
                    "decided_by": decision.decided_by,
                    "recovered": recovered,
                    "reasoning": decision.reasoning,
                    "shap_summary": classification.explanation_summary,
                },
            }).execute()

            # 7. Update bandit with outcome
            await bandit.update(
                root_cause=classification.root_cause.value,
                features=features,
                strategy=decision.strategy.value,
                reward=1.0 if recovered else 0.0,
            )

            results["processed"] += 1
            if recovered:
                results["recovered"] += 1
            elif new_status == "ESCALATED":
                results["escalated"] += 1
            else:
                results["failed"] += 1

        except Exception as e:
            import traceback
            logger.error("batch.payment_error", payment_id=session["payment_id"], error=str(e), tb=traceback.format_exc())
            results["failed"] += 1

    return {"status": "complete", "results": results}
