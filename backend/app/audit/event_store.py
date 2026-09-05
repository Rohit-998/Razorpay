"""Audit Event Store — append-only, immutable decision log."""

from datetime import datetime
from uuid import uuid4
from app.db.database import get_supabase
from app.models.schemas import AuditEvent
import structlog

logger = structlog.get_logger()

PAYMENT_LINK_SENT = "PAYMENT_LINK_SENT"
"""The one action that puts a link carrying `reference_id` in front of the customer, and so the
only one a `SYSTEM_RECOVERED` verdict can rest on."""

ACTION_EVENTS = (
    PAYMENT_LINK_SENT,
    "RETRY_SCHEDULED",
    "IMMEDIATE_RETRY_MOCKED",
    "DELAYED_RETRY_WOKE_UP",
    "ESCALATED",
)
"""Every event type that means an action was actually executed against a payment.

This tuple exists because the writer and its readers had drifted apart in silence.
`log_recovery_attempt` stamped **every** action `RETRY_ATTEMPTED` and filed the real name under
`event_data["action"]`, so `PAYMENT_LINK_SENT` — named in this class's own docstring, matched on
by `webhooks._last_contact_at`, and the sole precondition for a `SYSTEM_RECOVERED` verdict — did
not occur once in 2084 audit rows. Eight payment links were created against Razorpay's API in a
single batch and all eight were recorded as retries.

Nothing failed. `/batch/run` reported `processed: 190, errored: 0`, the links were real and
payable, and the audit trail was plausible. What it cost was the entire provable-recovery path:
the attribution split could reach `CUSTOMER_SELF_RECOVERED` and `AMBIGUOUS` but never the one
verdict that credits the system, so the dashboard's headline figure was structurally pinned at
₹0 by a mislabelled string.
"""

CUSTOMER_FACING_EVENTS = (PAYMENT_LINK_SENT, "ESCALATED")
"""The subset a human on the other end actually experiences: a message with a link, or an agent
telephoning. A retry is server-side — it spends no attention and, per `compliance.py`, none of
the customer's daily contact budget."""

LEGACY_ACTION_EVENT = "RETRY_ATTEMPTED"
"""What every action was called before the fix above, with its real name in `event_data["action"]`.

Readers have to keep matching it. This log is append-only by design and nothing rewrites history,
so the 118 rows written under the old name are the only record that those payments were ever acted
on. Dropping the name from the read filters would make them retroactively untouched — which is the
same class of bug as the one being fixed, arrived at from the other direction.
"""

ACTION_EVENTS_READ = ACTION_EVENTS + (LEGACY_ACTION_EVENT,)
"""What a reader filters `event_type` on: both naming eras. Writers use `ACTION_EVENTS`."""


def action_of(event: dict) -> str:
    """The action an audit row records, whichever naming era wrote it.

    New rows carry the action as the event type. Old ones carry `RETRY_ATTEMPTED` and the real
    name under `event_data["action"]`, so a row for a payment link written last week still
    resolves to `PAYMENT_LINK_SENT` here.
    """
    kind = event.get("event_type") or ""
    if kind == LEGACY_ACTION_EVENT:
        return (event.get("event_data") or {}).get("action") or kind
    return kind


def is_customer_facing(event: dict) -> bool:
    """Whether this row cost the customer some attention, as opposed to being server-side."""
    return action_of(event) in CUSTOMER_FACING_EVENTS


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
        """Log a recovery attempt under the name of the action that was taken.

        The event type is the action, not the constant `RETRY_ATTEMPTED` this used to write.
        See `ACTION_EVENTS` for what the constant cost. `action` stays in `event_data` as well,
        because rows written before this fix carry it there and nothing rewrites history in an
        append-only log.
        """
        self.log(session_id, payment_id, action, {
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
