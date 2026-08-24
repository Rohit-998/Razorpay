"""Payments API — list and detail views."""
from fastapi import APIRouter, HTTPException
from app.db.database import get_supabase

router = APIRouter()


@router.get("/payments")
async def list_payments(limit: int = 50, offset: int = 0):
    """List failed payments with pagination."""
    db = get_supabase()
    result = db.table("payments").select("*").order("created_at", desc=True).range(offset, offset + limit - 1).execute()
    return {"payments": result.data or [], "count": len(result.data or [])}


@router.get("/payments/{payment_id}")
async def get_payment(payment_id: str):
    """Get payment detail with recovery session and audit trail."""
    db = get_supabase()

    try:
        result = db.table("payments").select("*").eq("payment_id", payment_id).execute()
        payment_data = result.data[0] if result and result.data else None
    except Exception:
        payment_data = None

    if not payment_data:
        raise HTTPException(status_code=404, detail=f"Payment {payment_id} not found")

    try:
        session = db.table("recovery_sessions").select("*").eq("payment_id", payment_id).execute()
        session_data = session.data[0] if session and session.data else None
    except Exception:
        session_data = None

    try:
        audit = db.table("audit_events").select("*").eq("payment_id", payment_id).order("created_at").execute()
        audit_data = audit.data if audit and audit.data else []
    except Exception:
        audit_data = []

    return {
        "payment": payment_data,
        "session": session_data,
        "audit_trail": audit_data,
    }
