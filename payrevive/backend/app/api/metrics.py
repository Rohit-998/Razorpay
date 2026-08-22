"""Metrics API — classifier performance, bandit learning, batch report."""
from fastapi import APIRouter
from app.db.database import get_supabase
from app.strategy.bandit import bandit

router = APIRouter()


@router.get("/metrics/batch")
async def get_batch_report():
    """Get full batch run metrics."""
    db = get_supabase()
    sessions = db.table("recovery_sessions").select("*").neq("status", "OPEN").execute()
    data = sessions.data or []

    recovered = [s for s in data if s["status"] == "RECOVERED"]
    total_amount = sum(s.get("amount_recovered", 0) or 0 for s in data)
    recovered_amount = sum(s.get("amount_recovered", 0) or 0 for s in recovered)

    # Per-class breakdown
    by_class = {}
    for s in data:
        rc = s.get("root_cause", "UNKNOWN")
        if rc not in by_class:
            by_class[rc] = {"total": 0, "recovered": 0, "failed": 0}
        by_class[rc]["total"] += 1
        if s["status"] == "RECOVERED":
            by_class[rc]["recovered"] += 1
        else:
            by_class[rc]["failed"] += 1

    return {
        "batch_size": len(data),
        "recovery_rate": round(len(recovered) / len(data) * 100, 1) if data else 0,
        "amount_recovered": recovered_amount,
        "per_class": by_class,
    }


@router.get("/metrics/bandit")
async def get_bandit_learning():
    """Get bandit learning curves for dashboard."""
    data = await bandit.get_learning_data()
    return {"contexts": data}
