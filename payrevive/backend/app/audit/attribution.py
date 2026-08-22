"""Recovery Attribution Engine — honest counting of what PayRevive actually recovered."""

from datetime import datetime
from app.db.database import get_supabase
from app.models.schemas import AttributionType
from app.audit.event_store import event_store
import structlog

logger = structlog.get_logger()


class AttributionEngine:
    """
    When a payment gets recovered, determine WHO actually recovered it:
    
    SYSTEM_RECOVERED:
      - Our retry attempt succeeded
      - Customer clicked our payment link and paid
      
    CUSTOMER_SELF_RECOVERED:
      - Customer paid on their own BEFORE our action
      - Customer paid via a different session/device
      
    AMBIGUOUS:
      - Customer paid AFTER our link was sent, but via different channel
      - Can't prove causation
    
    Only SYSTEM_RECOVERED counts in our recovery rate.
    This is the "honest metrics" differentiator.
    """

    def attribute(
        self,
        payment_id: str,
        session_id: str,
        capture_source: str | None = None,
        capture_notes: dict | None = None,
    ) -> AttributionType:
        """Determine recovery attribution for a captured payment."""

        capture_notes = capture_notes or {}

        # Check if payment was recovered via our payment link
        if capture_notes.get("source") == "payrevive":
            attribution = AttributionType.SYSTEM_RECOVERED
            method = "payment_link"
        else:
            # Check the audit trail: did we take any action?
            trail = event_store.get_trail(session_id)
            action_events = [
                e for e in trail
                if e["event_type"] in (
                    "RETRY_ATTEMPTED", "PAYMENT_LINK_SENT", "RETRY_SCHEDULED"
                )
            ]

            if not action_events:
                # We didn't even take action yet — customer paid on their own
                attribution = AttributionType.CUSTOMER_SELF_RECOVERED
                method = "self"
            else:
                last_action = action_events[-1]
                last_action_type = last_action["event_type"]

                if last_action_type == "RETRY_ATTEMPTED":
                    # Our retry likely caused the recovery
                    attribution = AttributionType.SYSTEM_RECOVERED
                    method = "retry"
                elif last_action_type == "PAYMENT_LINK_SENT":
                    # We sent a link but customer may have paid via other means
                    attribution = AttributionType.AMBIGUOUS
                    method = "ambiguous_after_link"
                else:
                    attribution = AttributionType.AMBIGUOUS
                    method = "ambiguous"

        # Log the attribution decision
        event_store.log(session_id, payment_id, "ATTRIBUTION_DETERMINED", {
            "attribution": attribution.value,
            "method": method,
            "capture_source": capture_source,
        })

        # Update the session
        try:
            db = get_supabase()
            db.table("recovery_sessions").update({
                "attribution": attribution.value,
            }).eq("id", session_id).execute()
        except Exception as e:
            logger.error("attribution.update_failed", error=str(e))

        logger.info(
            "attribution.determined",
            payment_id=payment_id,
            attribution=attribution.value,
            method=method,
        )

        return attribution

    def get_attribution_summary(self) -> dict:
        """Get attribution breakdown for batch report."""
        try:
            db = get_supabase()
            sessions = db.table("recovery_sessions") \
                .select("attribution, amount_recovered") \
                .eq("status", "RECOVERED") \
                .execute()

            summary = {
                "SYSTEM_RECOVERED": {"count": 0, "amount": 0},
                "CUSTOMER_SELF_RECOVERED": {"count": 0, "amount": 0},
                "AMBIGUOUS": {"count": 0, "amount": 0},
            }

            for s in (sessions.data or []):
                attr = s.get("attribution", "AMBIGUOUS")
                if attr in summary:
                    summary[attr]["count"] += 1
                    summary[attr]["amount"] += s.get("amount_recovered", 0) or 0

            return summary
        except Exception as e:
            logger.error("attribution.summary_failed", error=str(e))
            return {}


# Singleton
attribution_engine = AttributionEngine()
