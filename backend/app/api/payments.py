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

    # Use maybe_single to avoid error when no row found
    payment = db.table("payments").select("*").eq("payment_id", payment_id).maybe_single().execute()
    if not payment.data:
        raise HTTPException(status_code=404, detail=f"Payment {payment_id} not found")

    session = db.table("recovery_sessions").select("*").eq("payment_id", payment_id).execute()
    audit = db.table("audit_events").select("*").eq("payment_id", payment_id).order("created_at").execute()

    return {
        "payment": payment.data,
        "session": session.data[0] if session.data else None,
        "audit_trail": audit.data or [],
    }
