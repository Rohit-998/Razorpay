"""Pipeline data API — surfaces Supabase tables for the pipeline view."""
from fastapi import APIRouter
from app.db.database import get_supabase

router = APIRouter()


@router.get("/pipeline/data")
async def get_pipeline_data(limit: int = 50, offset: int = 0):
    """Return payments joined with their recovery sessions for the pipeline table."""
    db = get_supabase()

    # Payments with pagination
    try:
        payments_res = (
            db.table("payments")
            .select("payment_id, amount, currency, method, bank, error_code, error_reason, created_at")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        payments = payments_res.data or []
    except Exception:
        payments = []

    # Recovery sessions
    try:
        sessions_res = (
            db.table("recovery_sessions")
            .select("payment_id, status, root_cause, root_cause_confidence, strategy, decided_by, amount_recovered, shap_explanation, llm_reasoning")
            .execute()
        )
        sessions_by_id = {s["payment_id"]: s for s in (sessions_res.data or [])}
    except Exception:
        sessions_by_id = {}

    # Audit events (latest per payment)
    try:
        audit_res = (
            db.table("audit_events")
            .select("payment_id, event_type, event_data, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        audit_by_id: dict = {}
        for evt in (audit_res.data or []):
            pid = evt["payment_id"]
            if pid not in audit_by_id:
                audit_by_id[pid] = evt
    except Exception:
        audit_by_id = {}

    # Merge into pipeline rows
    rows = []
    for p in payments:
        pid = p["payment_id"]
        session = sessions_by_id.get(pid, {})
        audit = audit_by_id.get(pid, {})
        rows.append({
            "payment_id": pid,
            "amount": p.get("amount", 0),
            "currency": p.get("currency", "INR"),
            "method": p.get("method"),
            "bank": p.get("bank"),
            "error_code": p.get("error_code"),
            "error_reason": p.get("error_reason"),
            "created_at": p.get("created_at"),
            "root_cause": session.get("root_cause"),
            "confidence": session.get("root_cause_confidence"),
            "strategy": session.get("strategy"),
            "decided_by": session.get("decided_by"),
            "recovery_status": session.get("status"),
            "amount_recovered": session.get("amount_recovered", 0),
            "shap_explanation": session.get("shap_explanation"),
            "llm_reasoning": session.get("llm_reasoning"),
            "audit_event": audit.get("event_type"),
        })

    # Summary counts
    total_payments = len(payments)
    total_sessions = len(sessions_by_id)
    recovered = sum(1 for s in sessions_by_id.values() if s.get("status") == "RECOVERED")
    failed = sum(1 for s in sessions_by_id.values() if s.get("status") == "FAILED")

    return {
        "rows": rows,
        "summary": {
            "total_payments": total_payments,
            "total_sessions": total_sessions,
            "recovered": recovered,
            "failed": failed,
            "audit_events": len(audit_by_id),
        },
    }
