"""Dashboard stats API."""
from fastapi import APIRouter
from app.db.database import get_supabase

router = APIRouter()


@router.get("/dashboard/stats")
async def get_stats():
    """Get recovery dashboard summary stats."""
    db = get_supabase()
    sessions = db.table("recovery_sessions").select("status, amount_recovered, root_cause").execute()

    total = len(sessions.data) if sessions.data else 0
    recovered = [s for s in (sessions.data or []) if s["status"] == "RECOVERED"]
    failed = [s for s in (sessions.data or []) if s["status"] == "FAILED"]
    escalated = [s for s in (sessions.data or []) if s["status"] == "ESCALATED"]

    total_recovered_amount = sum(s.get("amount_recovered", 0) or 0 for s in recovered)

    payments = db.table("payments").select("amount").execute()
    total_failed_amount = sum(p["amount"] for p in (payments.data or []))

    return {
        "total_failed": total,
        "total_recovered": len(recovered),
        "total_failed_permanent": len(failed),
        "total_escalated": len(escalated),
        "total_failed_amount": total_failed_amount,
        "total_recovered_amount": total_recovered_amount,
        "recovery_rate": round(len(recovered) / total * 100, 1) if total > 0 else 0,
    }
