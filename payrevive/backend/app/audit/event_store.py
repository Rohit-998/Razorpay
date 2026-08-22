"""Audit Event Store — append-only, immutable decision log."""

from datetime import datetime
from uuid import uuid4
from app.db.database import get_supabase
from app.models.schemas import AuditEvent
import structlog

logger = structlog.get_logger()


class EventStore:
    """
    Append-only audit trail. Every state change gets an immutable event.
    
    NO UPDATEs. NO DELETEs. Only INSERTs.
    
    Event types:
      INGESTED, CLASSIFIED, STRATEGY_SELECTED, COMPLIANCE_CHECKED,
      CIRCUIT_CHECKED, RETRY_ATTEMPTED, PAYMENT_LINK_SENT,
      RETRY_SCHEDULED, RECOVERED, RETRY_FAILED, MAX_RETRIES_EXCEEDED,
      ESCALATED, EXCEPTION_LOGGED, BANDIT_UPDATED
    """

    def log(self, session_id: str, payment_id: str, event_type: str, data: dict = None):
        """Log an audit event (synchronous for Supabase client)."""
        event = {
            "recovery_session_id": session_id,
            "payment_id": payment_id,
            "event_type": event_type,
            "event_data": data or {},
            "created_at": datetime.utcnow().isoformat(),
        }

        try:
            db = get_supabase()
            db.table("audit_events").insert(event).execute()
            logger.info("audit.logged", event_type=event_type, payment_id=payment_id)
        except Exception as e:
            logger.error("audit.log_failed", event_type=event_type, error=str(e))

    def log_classification(
        self, session_id: str, payment_id: str,
        root_cause: str, confidence: float,
        shap_explanations: list[dict], summary: str,
    ):
        """Log a classification decision with SHAP values."""
        self.log(session_id, payment_id, "CLASSIFIED", {
            "root_cause": root_cause,
            "confidence": confidence,
            "shap_explanations": shap_explanations,
            "explanation_summary": summary,
        })

    def log_strategy(
        self, session_id: str, payment_id: str,
        strategy: str, decided_by: str, reasoning: str,
        confidence: float,
    ):
        """Log a strategy selection decision."""
        self.log(session_id, payment_id, "STRATEGY_SELECTED", {
            "strategy": strategy,
            "decided_by": decided_by,
            "reasoning": reasoning,
            "confidence": confidence,
        })

    def log_compliance_check(
        self, session_id: str, payment_id: str,
        approved: bool, blocked_by: list[str] = None,
    ):
        """Log compliance engine result."""
        self.log(session_id, payment_id, "COMPLIANCE_CHECKED", {
            "approved": approved,
            "blocked_by": blocked_by or [],
        })

    def log_recovery_attempt(
        self, session_id: str, payment_id: str,
        action: str, details: dict = None,
    ):
        """Log a recovery attempt (retry, payment link, etc)."""
        self.log(session_id, payment_id, "RETRY_ATTEMPTED", {
            "action": action,
            **(details or {}),
        })

    def log_recovery_success(
        self, session_id: str, payment_id: str,
        method: str, amount_recovered: int,
    ):
        """Log successful recovery."""
        self.log(session_id, payment_id, "RECOVERED", {
            "method": method,
            "amount_recovered": amount_recovered,
        })

    def log_escalation(
        self, session_id: str, payment_id: str, reason: str,
    ):
        """Log escalation to merchant."""
        self.log(session_id, payment_id, "ESCALATED", {"reason": reason})

    def log_exception(
        self, session_id: str, payment_id: str,
        reason: str, category: str,
    ):
        """Log unrecoverable payment to exception list."""
        self.log(session_id, payment_id, "EXCEPTION_LOGGED", {
            "reason": reason,
            "category": category,
        })

    def get_trail(self, session_id: str) -> list[dict]:
        """Get full audit trail for a recovery session."""
        try:
            db = get_supabase()
            result = db.table("audit_events") \
                .select("*") \
                .eq("recovery_session_id", session_id) \
                .order("created_at") \
                .execute()
            return result.data or []
        except Exception as e:
            logger.error("audit.get_trail_failed", error=str(e))
            return []

    def get_trail_by_payment(self, payment_id: str) -> list[dict]:
        """Get audit trail by payment ID."""
        try:
            db = get_supabase()
            result = db.table("audit_events") \
                .select("*") \
                .eq("payment_id", payment_id) \
                .order("created_at") \
                .execute()
            return result.data or []
        except Exception as e:
            logger.error("audit.get_trail_failed", error=str(e))
            return []


# Singleton
event_store = EventStore()
