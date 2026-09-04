"""Metrics API — batch outcomes by root cause, and the bandit's posteriors.

`/metrics/batch` used to return `recovery_rate` and a per-cause `{total, recovered, failed}`
split. Two problems, both of which the eval report addresses at length. The rate counted
self-recoveries as wins. And the per-cause breakdown was keyed on the *predicted* root
cause, which makes it a picture of what the classifier believed rather than of what happened
— on a cause the model confuses, the row describes a mixture of two populations and the rate
in it is not a rate of anything.

The predicted-cause keying is kept, because in production the predicted label is the only
one that exists and the breakdown is genuinely useful for spotting a cause the policy is
losing on. But it is named as a prediction, and the rate is replaced by the attribution
split, so a row cannot claim a win it did not earn. The true-cause version of this table is
in `/eval/causes`, where the simulator knows the answer.
"""

from fastapi import APIRouter

from app.api.dashboard import ATTRIBUTION_LABELS, ATTRIBUTION_ORDER
from app.db.database import get_supabase
from app.strategy.bandit import bandit

router = APIRouter()


@router.get("/metrics/batch")
async def get_batch_report():
    """Closed sessions, split by attribution and by the cause the classifier predicted."""
    db = get_supabase()
    sessions = db.table("recovery_sessions").select("*").neq("status", "OPEN").execute()
    data = sessions.data or []

    def _empty() -> dict:
        return {
            "sessions": 0,
            "amount_recovered_paise": 0,
            "attributed": {v: 0 for v in ATTRIBUTION_ORDER},
            "unattributed": 0,
            "escalated": 0,
            "closed_without_recovery": 0,
        }

    overall = _empty()
    by_predicted_cause: dict[str, dict] = {}

    for row in data:
        cause = row.get("root_cause") or "UNCLASSIFIED"
        bucket = by_predicted_cause.setdefault(cause, _empty())
        for target in (overall, bucket):
            target["sessions"] += 1
            if row.get("status") == "RECOVERED":
                target["amount_recovered_paise"] += row.get("amount_recovered") or 0
                verdict = row.get("attribution")
                if verdict in target["attributed"]:
                    target["attributed"][verdict] += 1
                else:
                    target["unattributed"] += 1
            else:
                target["closed_without_recovery"] += 1
                if row.get("status") == "ESCALATED":
                    target["escalated"] += 1

    return {
        "batch_size": len(data),
        "overall": overall,
        "by_predicted_cause": by_predicted_cause,
        "attribution_labels": ATTRIBUTION_LABELS,
        "keyed_on": (
            "The cause the classifier predicted, not the true one — in production there is "
            "no other label available. Accuracy against the true cause is bounded at 65.78% "
            "for error-fields-only inference, so treat a row on a confusable cause as a "
            "mixture. The true-cause breakdown is at /api/v1/eval/causes."
        ),
    }


@router.get("/metrics/bandit")
async def get_bandit_learning():
    """Get bandit learning curves for dashboard."""
    data = await bandit.get_learning_data()
    return {"contexts": data}
