"""Properties the deployed compliance engine must hold.

This module is the one the brief names directly — "compliant escalation, stopping
rules" — and it had no tests at all. Each test below guards a claim the report
makes on its behalf:

  * the verdict is a function of its arguments  → an audit entry can be re-derived
  * quiet hours bind contact, not machine calls → the overnight window is not forfeit
  * amount ceilings bind rails, not messages    → every block has a reachable remedy
  * budgets roll on IST days                    → no double-contact at 05:30 IST
  * mixed tz-awareness does not raise           → the check actually runs in production
  * every strategy maps to an action            → nothing falls through the rules
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.config import get_settings
from app.execution import compliance as C
from app.models.schemas import RecoveryStrategy
from app.sim.types import ActionType

SETTINGS = get_settings()
FAILED_AT = datetime(2026, 3, 5, 14, 0)
SMALL = 50_000
"""₹500 — under every amount ceiling, so amount rules stay out of the way unless a
test is about them."""


def _at(hour: int, minute: int = 0, day: int = 5) -> datetime:
    return datetime(2026, 3, day, hour, minute)


def _check(action: ActionType, **kwargs) -> C.ComplianceCheck:
    params: dict = {
        "action": action,
        "amount_paise": SMALL,
        "at": _at(14),
        "failed_at": FAILED_AT,
    }
    params.update(kwargs)
    return C.evaluate(**params)


def test_the_verdict_depends_only_on_its_arguments() -> None:
    """Same inputs, same answer, whatever the wall clock says.

    This is what makes the audit trail worth keeping. `_is_quiet_hours` used to read
    `datetime.now()`, so a stored "approved" could not be checked later: replaying it
    at a different hour gave a different verdict.
    """
    args = dict(action=ActionType.SEND_LINK, at=_at(23), contacts_today=0)
    first = _check(**args)
    second = _check(**args)
    assert first.approved is second.approved is False
    assert first.blocked_by == second.blocked_by


def test_quiet_hours_block_contact_and_not_machine_retries() -> None:
    """A retry at 02:00 wakes nobody, and blocking it forfeits the outage window.

    The old engine blocked every action during quiet hours. Bank outages are
    disproportionately overnight, and a server-side charge on a stored mandate is a
    machine calling a machine — the customer learns about it from a receipt.
    """
    assert _check(ActionType.RETRY, at=_at(2)).approved
    assert not _check(ActionType.SEND_LINK, at=_at(2)).approved
    assert not _check(ActionType.ESCALATE, at=_at(2)).approved, (
        "an agent telephoning at 02:00 is more intrusive than the SMS we refuse"
    )


@pytest.mark.parametrize("hour", [22, 23, 0, 3, 7])
def test_quiet_hours_wrap_midnight(hour: int) -> None:
    assert C.is_quiet_hour(_at(hour), SETTINGS.quiet_hours_start, SETTINGS.quiet_hours_end)


@pytest.mark.parametrize("hour", [8, 12, 21])
def test_working_hours_are_not_quiet(hour: int) -> None:
    assert not C.is_quiet_hour(_at(hour), SETTINGS.quiet_hours_start, SETTINGS.quiet_hours_end)


def test_quiet_hours_agree_with_the_simulator() -> None:
    """The eval and the product must judge the same instant the same way.

    If they drift, the report is scoring a policy against limits production does not
    enforce, which is worse than reporting no number.
    """
    from app.sim.environment import RecoveryEnv

    for hour in range(24):
        assert C.is_quiet_hour(
            _at(hour), SETTINGS.quiet_hours_start, SETTINGS.quiet_hours_end
        ) == RecoveryEnv._is_quiet_hour(_at(hour)), f"disagreement at {hour}:00"


def test_an_aware_instant_is_converted_rather_than_read_as_local() -> None:
    """18:30 UTC is midnight IST — quiet — and 02:00 UTC is 07:30 IST, also quiet.

    Reading a UTC timestamp as local time shifts every hour by 5:30 and silently
    reclassifies the evening. Supabase returns aware timestamps, so this path is the
    production one.
    """
    from datetime import timezone

    utc_evening = datetime(2026, 3, 5, 18, 30, tzinfo=timezone.utc)
    utc_midday = datetime(2026, 3, 5, 6, 30, tzinfo=timezone.utc)
    assert C.is_quiet_hour(utc_evening, 22, 8)
    assert not C.is_quiet_hour(utc_midday, 22, 8), "12:00 IST is not quiet"


def test_a_mixed_awareness_subtraction_does_not_raise() -> None:
    """Aware `at` against naive `failed_at`, which is what production actually had.

    Supabase returns aware timestamps and the simulator produces naive ones, and both
    reach this module. The old engine subtracted them directly; the `TypeError` was
    swallowed by a broad `except` in the batch runner and counted as a failed payment,
    so the compliance check simply never ran on any live payment.
    """
    from datetime import timezone

    verdict = _check(
        ActionType.RETRY,
        at=datetime(2026, 3, 5, 15, 0, tzinfo=timezone.utc),
        failed_at=FAILED_AT,
    )
    assert verdict.approved


def test_the_recovery_window_closes() -> None:
    late = FAILED_AT + timedelta(hours=SETTINGS.max_recovery_window_hours + 1)
    verdict = _check(ActionType.RETRY, at=late)
    assert not verdict.approved
    assert verdict.recommendation == "LOG_EXCEPTION"
    assert any("window expired" in b.lower() for b in verdict.blocked_by)


def test_the_retry_budget_binds_and_suggests_a_link() -> None:
    spent = _check(ActionType.RETRY, retries_made=SETTINGS.max_retries_per_payment)
    assert not spent.approved
    assert spent.recommendation == "SWITCH_TO_PAYMENT_LINK"
    assert _check(
        ActionType.RETRY, retries_made=SETTINGS.max_retries_per_payment - 1
    ).approved


def test_the_retry_budget_does_not_bind_a_link() -> None:
    """Budgets are per action type. A payment out of retries can still be messaged —
    conflating the two is how a policy runs out of moves while three legal ones
    remain."""
    assert _check(
        ActionType.SEND_LINK, retries_made=SETTINGS.max_retries_per_payment + 5
    ).approved


def test_the_contact_budget_binds_only_contacting_actions() -> None:
    over = SETTINGS.max_contacts_per_day
    assert not _check(ActionType.SEND_LINK, contacts_today=over).approved
    assert _check(ActionType.RETRY, contacts_today=over).approved


def test_the_minimum_retry_interval_binds() -> None:
    too_soon = _check(
        ActionType.RETRY,
        minutes_since_last_retry=SETTINGS.min_retry_interval_minutes - 1,
    )
    assert not too_soon.approved
    assert too_soon.recommendation == "WAIT_FOR_INTERVAL"
    assert _check(
        ActionType.RETRY, minutes_since_last_retry=SETTINGS.min_retry_interval_minutes
    ).approved


def test_an_unknown_last_retry_time_does_not_block() -> None:
    """`None` means "we have no record", not "it was a second ago". Treating a missing
    timestamp as a violation would block the first retry on every payment."""
    assert _check(ActionType.RETRY, minutes_since_last_retry=None).approved


def test_a_rails_own_transaction_ceiling_is_enforced() -> None:
    """NPCI declines a UPI collect above ₹1 lakh, so proposing one is not a long shot,
    it is an action the network refuses. The remedy is another rail, not an agent."""
    from app.models.schemas import PaymentMethod

    over = SETTINGS.upi_transaction_ceiling_paise + 1
    verdict = _check(ActionType.SEND_LINK, amount_paise=over, method=PaymentMethod.UPI)
    assert not verdict.approved
    assert verdict.recommendation == "SWITCH_METHOD"
    assert _check(
        ActionType.SEND_LINK,
        amount_paise=SETTINGS.upi_transaction_ceiling_paise,
        method=PaymentMethod.UPI,
    ).approved


def test_the_ceiling_follows_the_rail_and_not_the_kind_of_action() -> None:
    """It is a limit on how much UPI can move, so it binds a mandate retry exactly as
    hard as a link, and does not bind a card at all.

    The rule it replaced was a flat ceiling on payment *links* of any method, which had
    the risk the wrong way round: a link is the customer paying, present and
    authenticated, so refusing one for ₹75,000 while permitting an unattended charge
    for the same amount is not defensible. It also had no reachable remedy — the agent
    bench is 20 slots against a 400-payment batch — and measurably dropped 35 of the
    largest failures in a batch untried.
    """
    from app.models.schemas import PaymentMethod

    over = SETTINGS.upi_transaction_ceiling_paise + 1
    assert not _check(
        ActionType.RETRY, amount_paise=over, method=PaymentMethod.UPI, has_mandate=True
    ).approved
    assert _check(
        ActionType.SEND_LINK, amount_paise=over, method=PaymentMethod.CARD
    ).approved, "a card link for ₹1 L is the ordinary way to collect a large payment"
    assert _check(ActionType.SEND_LINK, amount_paise=over).approved, (
        "with no rail named the customer chooses, and there is no ceiling to apply"
    )


def test_an_unnamed_rail_is_not_assumed_to_be_the_smallest_one() -> None:
    """A missing `method` means we do not know, and inventing a ceiling for it would
    refuse payments that would have gone through. Cards and netbanking have no entry
    for the same reason: their limits are per issuer and per customer, and we see
    neither."""
    from app.models.schemas import PaymentMethod

    huge = SETTINGS.upi_transaction_ceiling_paise * 100
    assert C.method_ceiling_paise(None) is None
    assert C.method_ceiling_paise(PaymentMethod.CARD) is None
    assert C.method_ceiling_paise(PaymentMethod.NETBANKING) is None
    assert _check(ActionType.SEND_LINK, amount_paise=huge).approved


def test_a_large_amount_may_not_be_retried_without_the_customer() -> None:
    over = SETTINGS.require_action_above_paise + 1
    assert not _check(ActionType.RETRY, amount_paise=over).approved
    assert _check(ActionType.SEND_LINK, amount_paise=over).approved, (
        "the rule is that a large charge needs the customer present, so the link is "
        "the remedy and must not be blocked by the same limit"
    )


def test_a_standing_mandate_is_the_customer_action_the_ceiling_asks_for() -> None:
    """The ceiling exists so a large sum is not charged without the customer's say-so.

    A mandate is that say-so, given in advance and for a stated amount. Applying the
    ceiling to mandate-backed retries anyway would block every autopay subscription
    over ₹10,000 — and it blocked a third of the retries the measured policy makes,
    which would have made the report quote money production refuses to collect.
    """
    over = SETTINGS.require_action_above_paise + 1
    assert _check(ActionType.RETRY, amount_paise=over, has_mandate=True).approved
    assert not _check(ActionType.RETRY, amount_paise=over, has_mandate=False).approved


def test_a_mandate_does_not_excuse_any_other_limit() -> None:
    """It answers one question — whether the amount was authorised — and no others."""
    from app.models.schemas import PaymentMethod

    over = SETTINGS.require_action_above_paise + 1
    spent = _check(
        ActionType.RETRY,
        amount_paise=over,
        has_mandate=True,
        retries_made=SETTINGS.max_retries_per_payment,
    )
    assert not spent.approved
    assert not _check(
        ActionType.SEND_LINK,
        amount_paise=SETTINGS.upi_transaction_ceiling_paise + 1,
        method=PaymentMethod.UPI,
        has_mandate=True,
    ).approved


def test_waiting_and_stopping_are_always_permitted() -> None:
    """A stopping rule that compliance can block is a policy with no way to stop, and
    the brief asks for stopping rules by name."""
    hopeless = dict(
        amount_paise=SETTINGS.require_action_above_paise * 10,
        at=_at(3),
        failed_at=FAILED_AT - timedelta(days=30),
        retries_made=99,
        contacts_today=99,
        minutes_since_last_retry=0.0,
    )
    assert _check(ActionType.WAIT, **hopeless).approved
    assert _check(ActionType.GIVE_UP, **hopeless).approved


def test_quiet_hours_outrank_a_spent_budget_in_the_recommendation() -> None:
    """Both bind, and the advice must be the free remedy. Telling the caller to
    escalate costs an agent slot; telling it to wait until morning costs nothing."""
    verdict = _check(
        ActionType.SEND_LINK, at=_at(23), contacts_today=SETTINGS.max_contacts_per_day
    )
    assert len(verdict.blocked_by) == 2
    assert verdict.recommendation == "DEFER_TO_MORNING"


def test_ist_days_roll_at_midnight_ist_not_at_utc_midnight() -> None:
    """A budget keyed on the UTC date rolls over at 05:30 IST, which splits an Indian
    morning across two days and permits a second contact twenty minutes later."""
    from datetime import timezone

    before = datetime(2026, 3, 5, 4, 0, tzinfo=timezone.utc)  # 09:30 IST, 5 Mar
    after = datetime(2026, 3, 5, 6, 0, tzinfo=timezone.utc)   # 11:30 IST, 5 Mar
    assert C.ist_day(before) == C.ist_day(after) == "2026-03-05"
    late = datetime(2026, 3, 5, 19, 0, tzinfo=timezone.utc)   # 00:30 IST, 6 Mar
    assert C.ist_day(late) == "2026-03-06"


def test_every_strategy_maps_to_an_action() -> None:
    """A strategy missing from the table would fall through to a default and be
    checked against the wrong rules."""
    missing = [
        s.value for s in RecoveryStrategy if s.value not in C.STRATEGY_ACTIONS
    ]
    assert not missing, f"strategies with no compliance mapping: {missing}"


def test_the_contact_key_is_one_identity_per_customer() -> None:
    """Counting a phone budget and an email budget separately lets the same customer
    be messaged twice under a limit of one."""
    from app.models.schemas import ErrorSource, FailedPayment, PaymentMethod

    def build(contact: str | None, email: str | None) -> FailedPayment:
        return FailedPayment(
            payment_id="pay_1", amount=SMALL, method=PaymentMethod.CARD,
            error_code="X", error_source=ErrorSource.GATEWAY, error_step="authorization",
            error_reason="payment_failed", customer_contact=contact,
            customer_email=email, created_at=FAILED_AT,
        )

    key = C.ComplianceEngine.contact_key
    assert key(build("+919000000001", "a@b.com")) == "+919000000001"
    assert key(build(None, "a@b.com")) == "a@b.com"
    assert key(build(None, None)) == ""
