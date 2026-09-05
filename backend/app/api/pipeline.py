"""Pipeline data API — surfaces Supabase tables for the pipeline view."""
from fastapi import APIRouter
from app.db.database import get_supabase, select_all

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

    # Recovery sessions. Paged, because this table has passed a thousand rows and an
    # unbounded `select()` stops there silently — the join would still fill in for the page
    # of payments on screen, but `total_sessions` and `recovered` below are counted over
    # everything this returns, and a total over the first thousand rows carries the label of
    # a total over all of them.
    try:
        sessions_by_id = {
            s["payment_id"]: s
            for s in select_all(
                "recovery_sessions",
                "payment_id, status, root_cause, root_cause_confidence, strategy, "
                "decided_by, amount_recovered, shap_explanation, llm_reasoning",
            )
        }
    except Exception:
        sessions_by_id = {}

    # Latest audit event per payment, asked for only the payments on this page. The previous
    # version pulled the whole table and kept the first row it saw per id, which PostgREST
    # caps at `db-max-rows` regardless of the range asked for — so the oldest payments in a
    # growing table lost their audit column with no sign that anything was missing. Filtering
    # server side by the ids actually on screen spends the cap on rows that get rendered.
    audit_by_id: dict = {}
    if payments:
        try:
            audit_res = (
                db.table("audit_events")
                .select("payment_id, event_type, event_data, created_at")
                .in_("payment_id", [p["payment_id"] for p in payments])
                .order("created_at", desc=True)
                .execute()
            )
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

    # Summary counts. `total_payments` is the size of this page; the session counts are over
    # the whole table, which is the only reading that makes them comparable to the dashboard.
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
            # Named for what it counts. It was `audit_events`, which read as a table total
            # and never was one.
            "audited_on_page": len(audit_by_id),
        },
    }
