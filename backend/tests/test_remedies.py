"""Whether a compliance block actually leads somewhere.

`compliance._recommendation` answers every block with a specific instruction —
`DEFER_TO_MORNING`, `WAIT_FOR_INTERVAL`, `SWITCH_METHOD`, one per rule. The worker used
to read that instruction, compare it against two values (one of which the engine never
returns), and close the session as FAILED for everything else. Five of the seven
remedies were discarded, so a payment the system had been told exactly how to recover
was written off at the moment it was told.

These tests are about the seam between those two modules, which is the kind of thing no
aggregate catches: the recovery rate looks the same whether a remedy was honoured or
silently dropped, because a dropped remedy just moves the payment into the failure
bucket it was already sitting in.

The last test is the general form. It enumerates what the engine can return and asserts
the worker has somewhere to put each one, so adding an eighth rule to the engine breaks
a test here rather than quietly deleting money in production.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta

import pytest

from app import worker
from app.config import get_settings
from app.execution import compliance as C
from app.execution.compliance import IST, is_quiet_hour, method_ceiling_paise
from app.models.schemas import (
    ComplianceCheck,
    ErrorSource,
    FailedPayment,
    PaymentMethod,
    RecoveryDecision,
    RecoveryStrategy,
)

SETTINGS = get_settings()

LAKH_AND_THREE_QUARTERS = 175_000_00
"""Above UPI's ₹1 lakh ceiling, so `SWITCH_METHOD` has something real to do."""


def _payment(
    amount: int = LAKH_AND_THREE_QUARTERS,
    method: PaymentMethod = PaymentMethod.UPI,
) -> FailedPayment:
    return FailedPayment(
        payment_id="pay_remedy",
        order_id="order_remedy",
        amount=amount,
        method=method,
        bank="HDFC",
        error_code="BAD_REQUEST_ERROR",
        error_source=ErrorSource.CUSTOMER,
        error_step="payment_authorization",
        error_reason="payment_failed",
        customer_contact="+919000000000",
        created_at=datetime(2026, 3, 5, 14, 30, tzinfo=IST),
    )


def _decision(
    strategy: RecoveryStrategy = RecoveryStrategy.IMMEDIATE_RETRY,
) -> RecoveryDecision:
    return RecoveryDecision(
        strategy=strategy,
        reasoning="test",
        confidence=0.8,
        decided_by="rule",
    )


def _blocked(recommendation: str, reason: str = "a limit") -> ComplianceCheck:
    return ComplianceCheck(
        approved=False, blocked_by=[reason], recommendation=recommendation
    )


# ---------------------------------------------------------------------------
# Doubles. The remedy logic is the only thing under test; Supabase, ARQ, the
# executor and the audit trail are all recorders.
# ---------------------------------------------------------------------------

class _Chain:
    """The one Supabase call shape the worker uses: table().update().eq().execute()."""

    def __init__(self, sink: list, table: str) -> None:
        self._sink, self._table, self._patch = sink, table, {}

    def update(self, patch: dict) -> "_Chain":
        self._patch = patch
        return self

    def eq(self, *_args) -> "_Chain":
        return self

    def execute(self) -> "_Chain":
        self._sink.append((self._table, self._patch))
        return self


class _FakeDB:
    def __init__(self) -> None:
        self.writes: list[tuple[str, dict]] = []

    def table(self, name: str) -> _Chain:
        return _Chain(self.writes, name)

    @property
    def session(self) -> dict:
        """The net effect on the session row, in write order."""
        merged: dict = {}
        for table, patch in self.writes:
            if table == "recovery_sessions":
                merged.update(patch)
        return merged


class _World:
    def __init__(self) -> None:
        self.db = _FakeDB()
        self.events: list[tuple[str, dict]] = []
        self.executed: list[tuple[RecoveryStrategy, PaymentMethod | None]] = []
        self.rechecks: list[tuple[RecoveryStrategy, int]] = []
        self.requeued: list[tuple[str, int]] = []
        self.execute_succeeds = True
        self.recheck_approves = True
        self.requeue_succeeds = True

    def event(self, event_type: str) -> dict | None:
        for name, data in self.events:
            if name == event_type:
                return data
        return None


@pytest.fixture
def world(monkeypatch) -> _World:
    """Everything `_honour_the_remedy` reaches outside itself, replaced by recorders."""
    w = _World()

    class _Events:
        def log(self, session_id, payment_id, event_type, data=None):
            w.events.append((event_type, data or {}))

        def log_exception(self, session_id, payment_id, reason, category):
            w.events.append(
                ("EXCEPTION_LOGGED", {"reason": reason, "category": category})
            )

    class _Executor:
        async def execute(self, payment, session_id, decision):
            w.executed.append((decision.strategy, decision.preferred_method))
            return w.execute_succeeds

    class _Engine:
        async def check(self, payment, decision, retry_count=0, **_kw):
            w.rechecks.append((decision.strategy, retry_count))
            return ComplianceCheck(
                approved=w.recheck_approves,
                blocked_by=[] if w.recheck_approves else ["still blocked"],
                recommendation=None if w.recheck_approves else "LOG_EXCEPTION",
            )

    async def _requeue(payment_id, minutes):
        w.requeued.append((payment_id, minutes))
        return w.requeue_succeeds

    monkeypatch.setattr(worker, "event_store", _Events())
    monkeypatch.setattr(worker, "executor", _Executor())
    monkeypatch.setattr(worker, "compliance_engine", _Engine())
    monkeypatch.setattr(worker, "_requeue", _requeue)
    return w


def _honour(
    world: _World,
    remedy: str,
    payment: FailedPayment | None = None,
    decision: RecoveryDecision | None = None,
    retry_count: int = 0,
) -> None:
    """Run the remedy handler to completion.

    `asyncio.run` rather than a pytest-asyncio marker: the suite has no other async
    tests and no pytest config, so an event-loop mode set in neither place is one more
    thing that can differ between this machine and a reviewer's.
    """
    import asyncio

    asyncio.run(
        worker._honour_the_remedy(
            world.db,
            "sess-1",
            payment or _payment(),
            decision or _decision(),
            _blocked(remedy),
            retry_count,
        )
    )


# ---------------------------------------------------------------------------
# `_minutes_until` — the three remedies that are "not now"
# ---------------------------------------------------------------------------

def test_waiting_out_the_gateway_interval_waits_exactly_that_long() -> None:
    assert worker._minutes_until("WAIT_FOR_INTERVAL") == (
        SETTINGS.min_retry_interval_minutes
    )


@pytest.mark.parametrize("remedy", ["DEFER_TO_MORNING", "WAIT_NEXT_DAY"])
def test_a_deferral_wakes_up_outside_quiet_hours(remedy: str) -> None:
    """The property that makes the remedy a remedy, asserted against the real clock.

    A fixed backoff cannot promise this. Sleeping 30 minutes at 23:50 wakes inside quiet
    hours again, and the payment ping-pongs between the queue and the same block until
    the recovery window closes — which is why the wake-up is computed from the wall clock
    rather than from a table of delays.
    """
    minutes = worker._minutes_until(remedy)
    assert minutes is not None and minutes >= 1
    wake = datetime.now(IST) + timedelta(minutes=minutes)
    assert not is_quiet_hour(wake, SETTINGS.quiet_hours_start, SETTINGS.quiet_hours_end)


def test_waiting_for_the_next_day_actually_crosses_into_it() -> None:
    """`WAIT_NEXT_DAY` answers a *daily* contact budget, so it has to land on a later
    IST date — waking at 08:05 the same morning would find the same spent ledger."""
    minutes = worker._minutes_until("WAIT_NEXT_DAY")
    now = datetime.now(IST)
    assert (now + timedelta(minutes=minutes)).date() > now.date()


@pytest.mark.parametrize(
    "remedy", ["SWITCH_METHOD", "SWITCH_TO_PAYMENT_LINK", "ESCALATE_TO_AGENT",
               "LOG_EXCEPTION", "ESCALATE"]
)
def test_the_remedies_that_are_not_waits_report_no_delay(remedy: str) -> None:
    """`None` is what routes these to the substitution branches rather than the queue."""
    assert worker._minutes_until(remedy) is None


# ---------------------------------------------------------------------------
# `_switched_rail` — the remedy that is "not that way"
# ---------------------------------------------------------------------------

def test_a_rail_too_small_for_the_amount_is_swapped_for_one_that_fits() -> None:
    """₹1.75 L will not move over UPI, and that is the network's limit, not the
    customer's unwillingness — so the answer is a different rail, not a phone call."""
    rail = worker._switched_rail(_payment(), _decision())
    assert rail is not None
    ceiling = method_ceiling_paise(rail)
    assert ceiling is None or LAKH_AND_THREE_QUARTERS <= ceiling


def test_the_rail_that_just_failed_is_never_offered_back() -> None:
    """Suggesting the customer retry the thing that just refused them is the one
    substitution guaranteed to achieve nothing."""
    decision = _decision()
    decision.preferred_method = PaymentMethod.CARD
    assert worker._switched_rail(_payment(method=PaymentMethod.CARD), decision) is not (
        PaymentMethod.CARD
    )


def test_the_decisions_own_rail_outranks_the_one_that_failed() -> None:
    """`preferred_method` is what compliance judged, so it is what has to be replaced."""
    decision = _decision()
    decision.preferred_method = PaymentMethod.UPI
    rail = worker._switched_rail(_payment(method=PaymentMethod.CARD), decision)
    assert rail is not PaymentMethod.UPI


# ---------------------------------------------------------------------------
# `_honour_the_remedy` — what the session looks like afterwards
# ---------------------------------------------------------------------------

def test_a_deferred_payment_stays_open(world: _World) -> None:
    """The bug in one line. A payment asleep until 08:05 has not failed, and closing it
    FAILED is not a status error — it is the write-off, and nothing reopens it."""
    _honour(world, "DEFER_TO_MORNING")
    assert world.db.session["status"] == "OPEN"
    assert world.requeued and world.requeued[0][0] == "pay_remedy"


def test_a_deferral_records_when_it_will_wake_up(world: _World) -> None:
    """In the audit trail, not on the session row: it is a thing that happened at an
    instant, and `recovery_sessions` has no column for it. The dashboard's countdown
    reads the latest event."""
    _honour(world, "WAIT_FOR_INTERVAL")
    remedy = world.event("COMPLIANCE_REMEDY")
    assert remedy is not None
    assert remedy["defer_minutes"] == SETTINGS.min_retry_interval_minutes
    assert datetime.fromisoformat(remedy["wake_at"]) > datetime.utcnow()
    assert remedy["queued"] is True


def test_a_payment_that_cannot_be_requeued_is_the_one_real_failure(world: _World) -> None:
    """Redis is down, so nothing will ever pick this payment up again. That is a dead
    session rather than a sleeping one, and recording it OPEN would leave it stuck in the
    queue depth forever with nothing coming to collect it."""
    world.requeue_succeeds = False
    _honour(world, "DEFER_TO_MORNING")
    assert world.db.session["status"] == "FAILED"
    assert world.event("COMPLIANCE_REMEDY")["queued"] is False


def test_a_deferral_never_spends_an_action(world: _World) -> None:
    """Waiting is free, and it has to stay free: a remedy that also messaged the customer
    would spend a contact slot to answer a block on spending contact slots."""
    _honour(world, "DEFER_TO_MORNING")
    assert world.executed == []


def test_switching_the_rail_sends_a_link_on_one_that_can_carry_the_amount(
    world: _World,
) -> None:
    _honour(world, "SWITCH_METHOD")
    assert len(world.executed) == 1
    strategy, rail = world.executed[0]
    assert strategy is RecoveryStrategy.LINK_ALT_METHOD
    ceiling = method_ceiling_paise(rail)
    assert ceiling is None or LAKH_AND_THREE_QUARTERS <= ceiling
    assert world.db.session["strategy"] == "LINK_ALT_METHOD"


def test_a_spent_retry_budget_becomes_a_question_to_the_customer(world: _World) -> None:
    """Out of automatic attempts is not out of options — it is the end of what we may do
    without asking, which is precisely what a payment link is."""
    _honour(world, "SWITCH_TO_PAYMENT_LINK")
    assert world.executed == [(RecoveryStrategy.LINK_SAME_METHOD, None)]
    assert world.db.session["strategy"] == "LINK_SAME_METHOD"


def test_escalation_closes_the_session_as_escalated(world: _World) -> None:
    """A human now owns it, so it is out of the automated pipeline — and that has to be a
    distinct status, because ESCALATED counted as FAILED is a queue nobody works."""
    _honour(world, "ESCALATE_TO_AGENT")
    assert world.executed == [(RecoveryStrategy.ESCALATE, None)]
    assert world.db.session["status"] == "ESCALATED"
    assert world.db.session["closed_at"]


def test_every_substituted_action_is_checked_before_it_runs(world: _World) -> None:
    """The remedy is a suggestion from the engine, not a licence to skip it.

    A `SWITCH_METHOD` at 02:00 IST would answer an amount-ceiling breach by sending an SMS
    inside quiet hours. The recheck is what stops one rule being satisfied at the cost of
    another, and `retry_count` is threaded in so it sees the same history the first check
    did.
    """
    _honour(world, "SWITCH_METHOD", retry_count=3)
    assert world.rechecks == [(RecoveryStrategy.LINK_ALT_METHOD, 3)]


def test_a_substitution_the_engine_refuses_is_not_executed(world: _World) -> None:
    world.recheck_approves = False
    _honour(world, "SWITCH_TO_PAYMENT_LINK")
    assert world.executed == []
    assert world.db.session["status"] == "FAILED"


def test_a_dead_end_is_recorded_as_an_exception_not_a_statistic(world: _World) -> None:
    """`LOG_EXCEPTION` is the engine saying the 72-hour window has closed. There is
    genuinely nothing legal left, and the honest place for that is the exception list a
    human reads — not dissolved into a recovery rate where it reads as underperformance."""
    _honour(world, "LOG_EXCEPTION")
    exc = world.event("EXCEPTION_LOGGED")
    assert exc is not None and exc["category"] == "COMPLIANCE"
    assert world.db.session["status"] == "FAILED"
    assert world.executed == []


def test_a_remedy_whose_action_fails_to_execute_closes_the_session(world: _World) -> None:
    """Razorpay refused the link. The remedy was reachable and we could not take it, which
    is a different thing from the remedy not existing — but the same outcome for the
    payment, and it must not be left OPEN with nothing scheduled to revisit it."""
    world.execute_succeeds = False
    _honour(world, "SWITCH_METHOD")
    assert world.db.session["status"] == "FAILED"
    assert world.event("EXCEPTION_LOGGED") is not None


def test_every_remedy_the_engine_can_return_has_somewhere_to_go() -> None:
    """The general form of the bug, and the reason this file exists.

    `_recommendation` is a ladder of string returns, and the worker used to compare that
    string against two values — one of which (`ESCALATE_TO_MERCHANT`) the engine has never
    returned. Five remedies fell through to a FAILED write. Nothing failed loudly; the
    payments simply stopped being worked on, which is invisible in every aggregate because
    an abandoned payment and an unrecoverable one look identical in a recovery rate.

    So the invariant is read off both modules' source rather than asserted case by case:
    every literal the engine can hand back is either a wait or is named in the handler. An
    eighth rule with a new remedy breaks this test on the day it is added.
    """
    engine = re.findall(r'return "([A-Z_]+)"', inspect.getsource(C._recommendation))
    assert len(engine) >= 7, f"expected the full ladder, found {engine}"

    handler = inspect.getsource(worker._honour_the_remedy)
    unhandled = [
        remedy
        for remedy in engine
        if worker._minutes_until(remedy) is None and f'"{remedy}"' not in handler
    ]
    assert not unhandled, (
        "the compliance engine can recommend "
        + ", ".join(unhandled)
        + " and _honour_the_remedy has no branch for it, so those payments are written "
        "off at the moment the system is told how to recover them"
    )
