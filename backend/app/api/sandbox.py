"""A stand-in for Razorpay's callbacks, so the live demo can close a session honestly.

`/batch/run` takes actions and stops. It has to: outcomes are not ours to decide, and the
version of that endpoint which decided them called `random.random()`, wrote the coin flip to
the database as `SYSTEM_RECOVERED`, and fed it to the bandit as a reward. Deleting that left a
real gap. On a sandbox key no customer ever pays, so every session stays `OPEN`, the
attribution split is three zeros, and the dashboard has nothing to show — which is why this
project's own database holds 217 recoveries with verdicts nothing decided.

This endpoint fills the gap without reopening it, and the distinction is worth being precise
about because it is the whole argument:

  It decides **whether the customer paid, and through which channel**. That is what a payment
  processor observes and reports. Nothing else.

  It does **not** decide the verdict, the amount, or the reward. Those come from
  `_record_recovery` — the same handler a real `payment_link.paid` lands in — which reads the
  clock, compares it against the last contact in the audit trail, and calls
  `attribution.attribute()`. The audit event, the `amount_recovered` column and the bandit
  update are all produced by production code paths that cannot tell they were driven from
  here.

  Its draws come from `app.sim.customer.PERSONAS` — the same behavioural parameters the eval
  harness measured the ₹17.52 L of lift under. The old coin flip drew against a table keyed on
  the classifier's own prediction, which is what made the bandit's "learning" circular. A
  persona is assigned by hashing the payment id, so it is stable: replaying this endpoint on
  the same batch produces the same customers making the same decisions, and a demo that
  behaves differently on the second run is a demo nobody can check.

Named `sandbox` and refused when `app_env == "production"`, because a synthetic capture in a
real merchant's books is fraud, not a demo.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, HTTPException

from app.api.webhooks import _record_recovery
from app.audit.event_store import (
    ACTION_EVENTS_READ,
    PAYMENT_LINK_SENT,
    action_of,
    is_customer_facing,
)
from app.config import get_settings
from app.db.database import select_all
from app.sim.customer import PERSONAS

logger = structlog.get_logger()
router = APIRouter()

CONTACT_EVENTS = ACTION_EVENTS_READ
"""The audit events that mean we actually acted. A session with none of these has had nothing
done to it, so there is no outcome to observe — it is not eligible.

Imported rather than restated. This tuple used to be a local literal, and the moment
`event_store` stopped writing `RETRY_ATTEMPTED` for everything, a private copy of the vocabulary
would have gone stale the same way `webhooks._last_contact_at`'s did — silently, while every
count kept moving. `ACTION_EVENTS_READ` is also what attribution filters on, which matters here
for a second reason: the instant this endpoint anchors `paid_at` to has to be the same instant
`attribute()` measures the gap from, or the two disagree about the window.
"""

LINK_EVENT = PAYMENT_LINK_SENT
"""The only action that can produce a `SYSTEM_RECOVERED` verdict, because it is the only one
that puts a link carrying `reference_id` in front of the customer. A retry that succeeds is a
capture on our side; a customer who pays after an escalation pays on their own channel."""


def _persona_for(payment_id: str):
    """A stable behavioural archetype for this payment.

    Hashed rather than drawn, so the endpoint is idempotent in the sense that matters for a
    demo: run it twice and the same customers make the same decisions. `hashlib` rather than
    `hash()` because Python salts string hashing per process, which would make the second
    uvicorn restart disagree with the first.
    """
    digest = hashlib.sha256(payment_id.encode("utf-8")).digest()
    return PERSONAS[digest[0] % len(PERSONAS)]


def _draw(payment_id: str, salt: str) -> float:
    """A uniform in [0, 1) derived from the payment id — the same one every time."""
    digest = hashlib.sha256(f"{payment_id}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _responsiveness(persona, messages_before: int) -> float:
    """The chance this customer acts on a message we sent.

    The channel is not read from the audit trail because nothing records it — the executor
    logs `PAYMENT_LINK_SENT` with the link id and the short url, and no event anywhere carries
    the channel the message went out on. So this takes the persona's *best* channel, which
    makes the sandbox mildly optimistic about click-through rather than silently arbitrary,
    and `channel_not_recorded` in the response says so.

    Fatigue is real and applied: the third message lands at `fatigue_decay ** 2` of the first,
    which is what stops the feed from rewarding a system that contacts people repeatedly.
    `messages_before` counts only the customer-facing actions, because a retry is a server-side
    call to the gateway. Counting retries as fatigue would have this endpoint punish a strategy
    the customer never saw, and it would disagree with `compliance.py`, which does not spend a
    contact slot on one either.
    """
    best = max(persona.channel_response.values())
    return best * (persona.fatigue_decay ** max(0, messages_before))


def _parse(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _payload(event: str, *, payment_id: str, order_id: str | None, paid_at: datetime) -> dict:
    """A Razorpay callback body, shaped the way the real one is.

    Built here rather than posted over HTTP so the signature check is not in the way, but the
    body is the same body — `reference_id` on a paid link, `order_id` on a bare capture — so
    `_recovered_reference` does the same join it does in production.

    `paid_at` is naive UTC, which is what `datetime.utcnow()` produces and therefore what
    every timestamp in this service means. It has to be encoded as UTC explicitly:
    `paid_at.timestamp()` would read the naive value as *local* time, and on an IST machine
    that is a 5.5-hour shift. The attribution window is six hours wide, so a skew of five and a
    half would land almost every self-recovery on the wrong side of it — and the resulting
    verdicts would look entirely plausible. `app/sim/customer.py` carries the same warning
    about the same class of bug.
    """
    stamp = int(paid_at.replace(tzinfo=timezone.utc).timestamp())
    if event == "payment_link.paid":
        return {
            "event": event,
            "payload": {
                "payment_link": {
                    "entity": {
                        "reference_id": payment_id,
                        "status": "paid",
                        "created_at": stamp,
                    }
                },
                "payment": {"entity": {"id": f"pay_sandbox_{payment_id[-8:]}", "created_at": stamp}},
            },
        }
    return {
        "event": event,
        "payload": {
            "payment": {
                "entity": {
                    "id": f"pay_sandbox_{payment_id[-8:]}",
                    "order_id": order_id,
                    "status": "captured",
                    "created_at": stamp,
                }
            }
        },
    }


@router.post("/sandbox/outcomes")
async def deliver_outcomes(limit: int = 200):
    """Let the customers respond, and route each response through the real webhook handler.

    One session at a time, and for each one only two questions are answered here: did this
    customer pay, and did they pay on our link or through a channel of their own. Everything
    that follows — the six-hour ambiguity comparison, the verdict, the reason sentence, the
    audit event, the money booked, the bandit update — is `_record_recovery` doing exactly
    what it does when Razorpay calls.

    Sessions with no contact in the trail are skipped rather than resolved. Nothing was done
    to them, so there is no outcome to observe, and closing them would be inventing one.
    """
    settings = get_settings()
    if settings.app_env == "production":
        raise HTTPException(
            status_code=403,
            detail=(
                "Refused: /sandbox/outcomes fabricates payment callbacks. Against a live "
                "merchant that is a false capture in their books. Set APP_ENV=development."
            ),
        )

    sessions = [
        row
        for row in select_all(
            "recovery_sessions", "id, payment_id, status, created_at"
        )
        if row.get("status") != "RECOVERED" and row.get("payment_id")
    ]
    if not sessions:
        return {"status": "no_data", "message": "No unresolved recovery sessions."}

    trail = select_all(
        "audit_events", "payment_id, event_type, created_at, event_data"
    )
    by_payment: dict[str, list[dict]] = {}
    for event in trail:
        if event.get("event_type") in CONTACT_EVENTS and event.get("payment_id"):
            by_payment.setdefault(event["payment_id"], []).append(event)

    # Eligibility is decided before the limit, not inside the loop, because the two orders give
    # different answers and one of them is wrong. This table holds 502 sessions that failed
    # outright and were never contacted; they sort ahead of the ones `/batch/run` had just
    # acted on. Slicing first spent the whole budget of 250 on rows with no outcome to observe
    # and reported `status: complete` with three zeros — a finished run in which nothing
    # happened. `never_contacted` below is that exclusion, counted against the full set rather
    # than against the page.
    eligible = [row for row in sessions if by_payment.get(row["payment_id"])]
    never_contacted = len(sessions) - len(eligible)
    considered = eligible[:limit]

    orders = {
        row["payment_id"]: row.get("order_id")
        for row in select_all("payments", "payment_id, order_id")
    }

    outcomes = {"paid_on_our_link": 0, "paid_their_own_way": 0, "no_response": 0}
    skipped = {"never_contacted": never_contacted, "already_closed_elsewhere": 0}
    verdicts: dict[str, int] = {}

    for session in considered:
        payment_id = session["payment_id"]
        contacts = sorted(
            by_payment.get(payment_id, []), key=lambda e: e.get("created_at") or ""
        )

        persona = _persona_for(payment_id)
        last = contacts[-1]
        contacted_at = _parse(last.get("created_at")) or datetime.utcnow()
        # Only the actions the customer could see decay their willingness to act on the next
        # one; `-1` because the message being responded to is not one of the ones that tired
        # them out.
        messages_before = sum(1 for event in contacts if is_customer_facing(event)) - 1

        # Question one: did our message work? Drawn against the persona's own click-through,
        # decayed by how many times we had already written to them.
        acts_on_us = _draw(payment_id, "click") < _responsiveness(persona, messages_before)
        # Question two, asked independently: would they have come back on their own? This is
        # the counterfactual the whole project is about, and it is deliberately not
        # conditioned on our actions — a customer with high self-recover propensity pays
        # whether or not we did anything, and the system must not be credited for it.
        self_recovers = _draw(payment_id, "self") < persona.self_recover_propensity

        if acts_on_us and action_of(last) == LINK_EVENT:
            event = "payment_link.paid"
            # Clicking through takes minutes to hours, not days.
            paid_at = contacted_at + timedelta(minutes=12 + int(_draw(payment_id, "lag") * 180))
            outcomes["paid_on_our_link"] += 1
        elif self_recovers:
            event = "payment.captured"
            # Their own schedule, spread across three days — which is what puts some of these
            # inside the six-hour ambiguity window and some far outside it. The verdict
            # difference is decided by `attribution.attribute()`, not here.
            paid_at = contacted_at + timedelta(hours=_draw(payment_id, "own") * 72)
            outcomes["paid_their_own_way"] += 1
        else:
            outcomes["no_response"] += 1
            continue

        result = await _record_recovery(
            event,
            _payload(
                event,
                payment_id=payment_id,
                order_id=orders.get(payment_id),
                paid_at=paid_at,
            ),
        )
        verdict = result.get("attribution") or f"not recorded: {result.get('reason', result.get('status'))}"
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        if not result.get("attribution"):
            skipped["already_closed_elsewhere"] += 1

    logger.info("sandbox.outcomes_delivered", **outcomes, verdicts=verdicts)
    return {
        "status": "complete",
        "sessions_unresolved": len(sessions),
        "sessions_eligible": len(eligible),
        "sessions_considered": len(considered),
        "customer_behaviour": outcomes,
        "skipped": skipped,
        "verdicts": verdicts,
        "decided_here": (
            "Whether the customer paid, and on which channel. Drawn from "
            "app.sim.customer.PERSONAS — the same parameters the eval harness measured lift "
            "under — and keyed on a hash of the payment id, so a replay reproduces it."
        ),
        "decided_by_production_code": (
            "The verdict, the amount booked and the bandit reward. This endpoint calls the "
            "same _record_recovery a real payment_link.paid lands in; attribution compares "
            "the payment time against the last contact in the audit trail."
        ),
        "channel_not_recorded": (
            "No event stores which channel a message went out on, so responsiveness uses the "
            "persona's best channel. That makes link click-through here an upper bound."
        ),
    }
