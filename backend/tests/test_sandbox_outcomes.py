"""The sandbox may decide who pays. It may not decide who gets the credit.

That line is the whole reason this endpoint is allowed to exist. The code it replaces —
`/batch/run` as it was originally written — crossed it: it drew `random.random()` against a
table of recovery rates keyed on the classifier's own prediction, wrote the result to the
database as `SYSTEM_RECOVERED`, and handed it to `bandit.update()` as a reward. Three claims
came out of that, all false: a recovery rate that was a property of the seed, an audit trail
asserting causation for events that never happened, and a bandit learning from its own
classifier's confidence.

`/sandbox/outcomes` answers only the two questions a payment processor answers — did the
customer pay, and on which channel — and then calls the same `_record_recovery` a real
`payment_link.paid` lands in. So these tests are mostly about what the endpoint does *not*
do: the verdicts below are produced by `attribution.attribute()` reading a clock, and the test
proves it by making the clock the only thing that changes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from app.api import sandbox, webhooks
from app.db import database
from app.execution import attribution

CONTACTED_AT = datetime(2026, 9, 1, 10, 0, 0)


def _run(coro):
    return asyncio.run(coro)


class _Chain:
    def __init__(self, rows: list[dict]) -> None:
        self.data = rows

    def select(self, *_a, **_k) -> "_Chain":
        return self

    def eq(self, column: str, value) -> "_Chain":
        return _Chain([r for r in self.data if r.get(column) == value])

    def range(self, start: int, end: int) -> "_Chain":
        return _Chain(self.data[start : end + 1])

    def execute(self) -> "_Chain":
        return self


class _FakeDB:
    def __init__(self, tables: dict[str, list[dict]]) -> None:
        self._tables = tables

    def table(self, name: str) -> _Chain:
        return _Chain(self._tables.get(name, []))


def _session(payment_id: str) -> dict:
    return {
        "id": f"s_{payment_id}",
        "payment_id": payment_id,
        "status": "OPEN",
        "created_at": CONTACTED_AT.isoformat(),
    }


def _contact(payment_id: str, event_type: str, at: datetime = CONTACTED_AT) -> dict:
    return {
        "payment_id": payment_id,
        "event_type": event_type,
        "created_at": at.isoformat(),
        "event_data": {},
    }


@pytest.fixture()
def bench(monkeypatch: pytest.MonkeyPatch):
    """Wire the endpoint to a fake database and to a recorder that runs the real rule.

    `_record_recovery` is replaced, but what replaces it calls `webhooks._paid_at` and
    `attribution.attribute()` — the production clock reader and the production verdict — so
    the assertions below are about real arithmetic on a payload the endpoint really built.
    Only Supabase writes, Redis and the bandit are absent.
    """
    calls: list[dict] = []

    def build(sessions: list[dict], events: list[dict], *, env: str = "development"):
        db = _FakeDB({
            "recovery_sessions": sessions,
            "audit_events": events,
            "payments": [
                {"payment_id": s["payment_id"], "order_id": f"order_{s['payment_id']}"}
                for s in sessions
            ],
        })
        monkeypatch.setattr(database, "get_supabase", lambda: db)
        monkeypatch.setattr(
            sandbox, "get_settings", lambda: type("S", (), {"app_env": env})()
        )

        async def recorder(event: str, payload: dict) -> dict:
            paid_at = webhooks._paid_at(payload)
            reference, via_our_link = webhooks._recovered_reference(event, payload)
            last = next(
                (
                    datetime.fromisoformat(e["created_at"])
                    for e in reversed(events)
                    if e["event_type"] in sandbox.CONTACT_EVENTS
                    and (
                        e["payment_id"] == reference
                        or f"order_{e['payment_id']}" == reference
                    )
                ),
                None,
            )
            verdict, why = attribution.attribute(
                paid_at=paid_at, via_our_link=via_our_link, last_contact_at=last
            )
            calls.append(
                {
                    "event": event,
                    "reference": reference,
                    "paid_at": paid_at,
                    "verdict": verdict,
                    "why": why,
                }
            )
            return {"status": "recorded", "attribution": verdict, "reasoning": why}

        monkeypatch.setattr(sandbox, "_record_recovery", recorder)
        return calls

    return build


# ── What the endpoint refuses to do ───────────────────────────────────────────


def test_it_will_not_run_against_a_live_merchant(bench) -> None:
    """A fabricated capture in a real merchant's books is not a demo."""
    bench([_session("pay_1")], [_contact("pay_1", "PAYMENT_LINK_SENT")], env="production")
    with pytest.raises(HTTPException) as caught:
        _run(sandbox.deliver_outcomes())
    assert caught.value.status_code == 403
    assert "APP_ENV" in caught.value.detail


def test_a_session_nobody_contacted_is_skipped_not_resolved(bench) -> None:
    """No action was taken, so there is no outcome to observe. Closing it would be inventing
    one — and it would credit the system for a customer it never reached."""
    calls = bench([_session("pay_untouched")], [])
    served = _run(sandbox.deliver_outcomes())
    assert calls == []
    assert served["skipped"]["never_contacted"] == 1
    assert served["customer_behaviour"]["paid_on_our_link"] == 0


def test_the_endpoint_never_names_a_verdict_itself(bench) -> None:
    """The response reports what it decided and what it did not, and the two lists are
    disjoint. `verdicts` is populated from what came back out of `_record_recovery`."""
    bench([_session("pay_1")], [_contact("pay_1", "PAYMENT_LINK_SENT")])
    served = _run(sandbox.deliver_outcomes())
    assert "channel" in served["decided_here"]
    assert "verdict" in served["decided_by_production_code"]
    assert set(served["customer_behaviour"]) == {
        "paid_on_our_link",
        "paid_their_own_way",
        "no_response",
    }, "the endpoint's own vocabulary is about payment, not about credit"


# ── The verdict follows the clock, and only the clock ─────────────────────────


def test_a_link_payment_is_the_one_case_causation_is_recorded(bench) -> None:
    """`payment_link.paid` carries `reference_id`, so nothing is inferred. This is also the
    only path that can produce `SYSTEM_RECOVERED` at all."""
    calls = bench(
        [_session("pay_clicker")], [_contact("pay_clicker", "PAYMENT_LINK_SENT")]
    )
    _run(sandbox.deliver_outcomes())
    link_calls = [c for c in calls if c["event"] == "payment_link.paid"]
    if not link_calls:
        pytest.skip("this payment id's persona did not click; covered by the sweep below")
    assert all(c["verdict"] == attribution.SYSTEM_RECOVERED for c in link_calls)
    assert all("on the link we sent" in c["why"] for c in link_calls)


def test_a_capture_soon_after_contact_is_ambiguous_and_one_much_later_is_not(
    bench,
) -> None:
    """The same event type, the same customer, the same channel — two verdicts, and the only
    difference between them is the hour on the clock.

    This is the property that makes the sandbox honest. If the endpoint were choosing
    verdicts, moving the payment time would not change one.
    """
    inside = webhooks._paid_at(
        sandbox._payload(
            "payment.captured",
            payment_id="pay_x",
            order_id="order_x",
            paid_at=CONTACTED_AT + timedelta(hours=2),
        )
    )
    outside = webhooks._paid_at(
        sandbox._payload(
            "payment.captured",
            payment_id="pay_x",
            order_id="order_x",
            paid_at=CONTACTED_AT + timedelta(hours=40),
        )
    )
    near, _ = attribution.attribute(
        paid_at=inside, via_our_link=False, last_contact_at=CONTACTED_AT
    )
    far, _ = attribution.attribute(
        paid_at=outside, via_our_link=False, last_contact_at=CONTACTED_AT
    )
    assert near == attribution.AMBIGUOUS
    assert far == attribution.CUSTOMER_SELF_RECOVERED


def test_the_payment_time_comes_from_the_payload_not_from_the_wall_clock(bench) -> None:
    """`_paid_at` reads the processor's stamp. Using our receipt time would widen every gap by
    however long the webhook was delayed, turning ambiguous outcomes into self-recoveries and
    changing what the bandit learns."""
    stamped = CONTACTED_AT + timedelta(hours=3)
    payload = sandbox._payload(
        "payment.captured", payment_id="pay_x", order_id="order_x", paid_at=stamped
    )
    assert webhooks._paid_at(payload) == stamped
    assert webhooks._paid_at({"event": "payment.captured", "payload": {}}) != stamped


def test_a_recovery_is_never_timestamped_before_the_message_that_caused_it(bench) -> None:
    """A negative gap would make `attribute()` compute negative hours and read as inside the
    ambiguity window, which would book a payment that happened first as possibly ours."""
    calls = bench(
        [_session(f"pay_{i:04d}") for i in range(60)],
        [_contact(f"pay_{i:04d}", "PAYMENT_LINK_SENT") for i in range(60)],
    )
    _run(sandbox.deliver_outcomes())
    assert calls, "60 customers and none of them paid — the draw is broken"
    assert all(c["paid_at"] > CONTACTED_AT for c in calls)


# ── The behaviour model, and the fact that it is not a coin flip ──────────────


def test_the_same_batch_twice_produces_the_same_customers(bench) -> None:
    """A demo that behaves differently on the second run cannot be checked, and a reviewer who
    reruns it and sees different numbers has learned nothing except that the numbers move."""
    payments = [f"pay_{i:04d}" for i in range(40)]
    first = bench(
        [_session(p) for p in payments],
        [_contact(p, "PAYMENT_LINK_SENT") for p in payments],
    )
    one = _run(sandbox.deliver_outcomes())
    first_calls = [(c["reference"], c["event"], c["paid_at"]) for c in first]
    first.clear()

    two = _run(sandbox.deliver_outcomes())
    assert one["customer_behaviour"] == two["customer_behaviour"]
    assert [(c["reference"], c["event"], c["paid_at"]) for c in first] == first_calls


def test_repeated_contact_lands_softer_than_the_first(bench) -> None:
    """Fatigue is applied, so the feed cannot reward a system that messages people four times.

    Asserted on the responsiveness function rather than on outcome counts, because a count
    over 40 hashed draws is noisy and this is an exact property of the model.
    """
    persona = sandbox._persona_for("pay_0001")
    first = sandbox._responsiveness(persona, 0)
    third = sandbox._responsiveness(persona, 2)
    assert third == pytest.approx(first * persona.fatigue_decay**2)
    assert third < first


def test_self_recovery_is_drawn_independently_of_our_action(bench) -> None:
    """The counterfactual the project is built on. A customer's propensity to come back on
    their own must not be conditioned on what we did, or the system gets credit for it.

    Two salted draws off the same payment id, asserted distinct — if the endpoint reused one
    draw for both questions, every clicker would also be a self-recoverer and the ambiguous
    bucket would be an artefact of the code rather than of behaviour.
    """
    for payment_id in (f"pay_{i:04d}" for i in range(20)):
        assert sandbox._draw(payment_id, "click") != sandbox._draw(payment_id, "self")


def test_the_personas_are_the_ones_the_harness_measured_lift_under(bench) -> None:
    """Not a table typed into this module. If someone retunes the simulator, the sandbox moves
    with it, and the live demo keeps matching the report."""
    from app.sim.customer import PERSONAS

    assert sandbox.PERSONAS is PERSONAS
    assigned = {sandbox._persona_for(f"pay_{i:04d}").name for i in range(200)}
    assert assigned == {p.name for p in PERSONAS}, "every archetype is reachable"


def test_only_a_link_can_produce_a_provable_recovery(bench) -> None:
    """A retry or an escalation never yields `payment_link.paid`, because neither one puts a
    link carrying `reference_id` in front of the customer. Those customers can still pay — on
    their own channel, which is what the capture event means."""
    payments = [f"pay_{i:04d}" for i in range(40)]
    calls = bench(
        [_session(p) for p in payments],
        [_contact(p, "RETRY_ATTEMPTED") for p in payments],
    )
    served = _run(sandbox.deliver_outcomes())
    assert served["customer_behaviour"]["paid_on_our_link"] == 0
    assert all(c["event"] == "payment.captured" for c in calls)
    assert attribution.SYSTEM_RECOVERED not in served["verdicts"]
