"""Live-traffic stats, reported the way the evaluation says they have to be.

This endpoint used to return `recovery_rate: recovered / total * 100`. That number is the
one thing `reports/REPORT.md` spends a page arguing against, for a reason that is not
pedantic: a recovery rate counts every customer who would have paid anyway as a win. On a
batch where a third of people come back on their own — which is roughly what happens — a
system that did nothing at all would have posted a 33% "recovery rate" and looked like it
was working.

The product cannot fix that by computing lift instead. Lift needs a counterfactual, and
live traffic has no twin batch running under `do_nothing`; that is exactly why the harness
exists and why it runs in a simulator. What live traffic *does* have is attribution:
Razorpay names our payment link in the `payment_link.paid` webhook, so for each recovery we
know whether the customer paid on our link, paid through their own channel, or paid so soon
after our message that the two are inseparable.

So this returns the split, and the split is the honest live metric. `SYSTEM_RECOVERED` is
ours. `AMBIGUOUS` is not a win and is never counted as one. The counterfactual number — the
measured ₹17.52 L per batch with its interval — is served from `/eval/ladder`, computed
offline under fixed seeds, and the two are deliberately different endpoints because they are
different kinds of claim.
"""

from fastapi import APIRouter

from app.db.database import get_supabase, select_all
from app.execution import attribution

router = APIRouter()

# The verdicts, in the order a reader should meet them: what we can claim, what we cannot,
# what we are not allowed to decide either way.
ATTRIBUTION_ORDER = ["SYSTEM_RECOVERED", "CUSTOMER_SELF_RECOVERED", "AMBIGUOUS"]

ATTRIBUTION_LABELS = {
    "SYSTEM_RECOVERED": "Paid on our link — ours",
    "CUSTOMER_SELF_RECOVERED": "Came back on their own — not ours",
    "AMBIGUOUS": "Paid soon after we messaged — unprovable either way",
}

UNATTRIBUTED_WHY = (
    "No RECOVERY_OBSERVED event, so nothing decided this. Either the callback has not "
    "arrived, or the row predates the attribution rule and its verdict column was written "
    "by a coin flip."
)


def audit_backed_sessions() -> set[str]:
    """Session ids whose attribution verdict an audit event stands behind.

    One read, one place, shared by `/dashboard/stats` and `/metrics/batch` so the two cannot
    disagree about which recoveries are claimable — they were both reading the `attribution`
    column directly, and both reporting the legacy coin flips as wins.
    """
    return {
        event["recovery_session_id"]
        for event in select_all(
            "audit_events",
            "recovery_session_id",
            event_type=attribution.OBSERVATION_EVENT,
        )
        if event.get("recovery_session_id")
    }


@router.get("/dashboard/stats")
async def get_stats():
    """Session counts and the attribution split, in paise.

    No rate is returned that divides recoveries by failures. The closest thing to a
    headline here is `attributed.SYSTEM_RECOVERED.amount_paise`, which is money we can
    point at a link for — and even that is a gross figure, not a lift.

    `provably_ours.share_of_recovered` is served rather than left to the caller because the
    dashboard needs a percentage for its progress bar, and a percentage the browser derives
    is a percentage nobody reviewed. Its denominator is deliberately recovered rupees, not
    failed payments, which is the distinction the whole endpoint exists to keep.

    A verdict is counted only where an audit event justifies it — see
    `attribution.verdict_of`. Reading the column on its own had this endpoint reporting 100%
    of recovered rupees as provably ours on 217 rows a deleted `random.random()` wrote, which
    is the overclaim the rest of this file argues against.
    """
    rows = select_all(
        "recovery_sessions", "id, status, amount_recovered, root_cause, attribution"
    )

    by_status: dict[str, int] = {}
    for row in rows:
        status = row.get("status") or "UNKNOWN"
        by_status[status] = by_status.get(status, 0) + 1

    observed = audit_backed_sessions()

    attributed = {
        verdict: {"label": ATTRIBUTION_LABELS[verdict], "sessions": 0, "amount_paise": 0}
        for verdict in ATTRIBUTION_ORDER
    }
    # A recovered session whose verdict no audit event backs is its own category rather
    # than a silent zero, and rather than a claim. The webhook that decides causation may
    # not have arrived yet, or — for the legacy rows — may never have run at all.
    unattributed = {
        "label": "Recovered, cause not established",
        "sessions": 0,
        "amount_paise": 0,
        "why": UNATTRIBUTED_WHY,
    }
    for row in rows:
        if row.get("status") != "RECOVERED":
            continue
        amount = row.get("amount_recovered") or 0
        bucket = attributed.get(attribution.verdict_of(row, observed)) or unattributed
        bucket["sessions"] += 1
        bucket["amount_paise"] += amount

    at_risk_paise = sum(p["amount"] for p in select_all("payments", "amount"))

    # The one share worth putting on a dashboard, computed here so the browser does not have
    # to. Denominator is every rupee that came back, however it came back — so the figure
    # answers "of the money recovered, how much can we show was us?" and falls when the
    # ambiguous bucket grows. A rate over *failed* payments would be the recovery rate this
    # endpoint refuses to serve; this is a share of recoveries, not a claim about failures.
    recovered_paise = sum(b["amount_paise"] for b in attributed.values()) + unattributed[
        "amount_paise"
    ]
    ours_paise = attributed["SYSTEM_RECOVERED"]["amount_paise"]

    return {
        "sessions_total": len(rows),
        "by_status": by_status,
        "open": by_status.get("OPEN", 0) + by_status.get("IN_PROGRESS", 0),
        "at_risk_paise": at_risk_paise,
        # Ingested minus returned. Subtracted here because the dashboard would otherwise do
        # it over whatever page of payments it happens to have fetched, which is a different
        # and smaller number wearing the same label.
        "unrecovered_paise": max(0, at_risk_paise - recovered_paise),
        "attributed": attributed,
        "unattributed": unattributed,
        "attribution_order": ATTRIBUTION_ORDER,
        "provably_ours": {
            "amount_paise": ours_paise,
            "sessions": attributed["SYSTEM_RECOVERED"]["sessions"],
            "recovered_paise": recovered_paise,
            "share_of_recovered": (
                round(ours_paise / recovered_paise, 4) if recovered_paise else 0.0
            ),
            # What the share is missing, so a low one can be read correctly. A 0% here means
            # one of two very different things — nothing worked, or nothing was measured —
            # and the dashboard cannot tell them apart from a percentage alone.
            "unestablished_sessions": unattributed["sessions"],
            "unestablished_paise": unattributed["amount_paise"],
            "caveat": (
                (
                    f"{unattributed['sessions']} of "
                    f"{unattributed['sessions'] + sum(b['sessions'] for b in attributed.values())} "
                    "recoveries in this table have no attribution event, so they are "
                    "excluded from the numerator rather than assumed to be ours. Run "
                    "`python -m app.eval` for the measured figure, which does have a "
                    "counterfactual."
                )
                if unattributed["sessions"]
                else None
            ),
            "note": (
                "Share of recovered rupees that Razorpay's webhook named our payment link "
                "for. Not a recovery rate: the denominator is money that came back, not "
                "payments that failed."
            ),
        },
        "counterfactual": {
            "available_at": "/api/v1/eval/ladder",
            "note": (
                "Lift needs a batch that ran under do_nothing on the same customers and the "
                "same coin flips. Live traffic has no such twin, so the measured figure is "
                "produced offline by the eval harness under fixed seeds rather than "
                "estimated from this table."
            ),
        },
        "on_recovery_rates": (
            "A recovery rate counts customers who would have paid anyway. The split above "
            "separates them out instead: only SYSTEM_RECOVERED is money we can attribute "
            "to an action we took."
        ),
    }


@router.get("/dashboard/exceptions")
async def get_exceptions(limit: int = 50):
    """The payments a human has to work, and why each one arrived here.

    This list is the other half of the stopping rule. `GIVE_UP` closes a session, and if the
    only record of that were a status of `FAILED` then an abandoned payment would read
    identically to an unrecoverable one — which is precisely the collapse the report refuses
    to make. Every give-up writes an `EXCEPTION_LOGGED` event carrying the sentence that
    justified it, and this endpoint is that queue.

    Read from `audit_events` rather than from a status column because the reason is the
    point. `compliance block with no reachable remedy` and `recovery window closed` both
    produce a closed session and mean entirely different things about whether the system
    behaved correctly.
    """
    db = get_supabase()
    events = (
        db.table("audit_events")
        .select("payment_id, recovery_session_id, event_data, created_at")
        .eq("event_type", "EXCEPTION_LOGGED")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = events.data or []

    by_category: dict[str, int] = {}
    items = []
    for row in rows:
        data = row.get("event_data") or {}
        category = data.get("category") or "UNKNOWN"
        by_category[category] = by_category.get(category, 0) + 1
        items.append(
            {
                "payment_id": row.get("payment_id"),
                "session_id": row.get("recovery_session_id"),
                "category": category,
                "reason": data.get("reason"),
                "logged_at": row.get("created_at"),
            }
        )

    return {
        "count": len(items),
        "by_category": by_category,
        "exceptions": items,
        "note": (
            "A payment on this list has run out of legal moves, not out of value. It is "
            "handed to a person with the reason attached rather than dissolved into a rate."
        ),
    }

