"""The rulebook endpoint has to report the rules that are actually enforced.

`/compliance/policy` exists so the dashboard's "Safety checks applied" panel can name real
limits instead of the three invented sentences it used to print. That only helps if the
numbers it serves are the ones `evaluate()` checks against — a rulebook that drifts from the
rules is worse than no rulebook, because it is a compliance claim that looks sourced.

So these tests read `Settings` and the action sets straight out of `compliance.py` and demand
the endpoint agree with them. Nothing here asserts on wording; the sentences are free to be
rewritten, the values are not.
"""

from fastapi.testclient import TestClient

from app.config import get_settings
from app.execution import compliance as C
from app.main import app
from app.models.schemas import PaymentMethod
from app.sim.types import ActionType

client = TestClient(app)


def policy() -> dict:
    response = client.get("/api/v1/compliance/policy")
    assert response.status_code == 200
    return response.json()


def limits() -> dict[str, dict]:
    return {row["key"]: row for row in policy()["limits"]}


def test_every_limit_reports_the_setting_it_enforces() -> None:
    """The served value equals the configured value, for each limit."""
    s = get_settings()
    rows = limits()
    assert rows["contacts_per_day"]["value"] == s.max_contacts_per_day
    assert rows["max_retries"]["value"] == s.max_retries_per_payment
    assert rows["mandate_ceiling"]["value"] == s.require_action_above_paise
    assert rows["rail_ceiling"]["value"] == C.method_ceiling_paise(PaymentMethod.UPI)
    assert rows["recovery_window"]["value"] == f"{s.max_recovery_window_hours}h"
    assert rows["retry_interval"]["value"] == f"{s.min_retry_interval_minutes}m"
    assert rows["quiet_hours"]["value"] == (
        f"{s.quiet_hours_start}:00–{s.quiet_hours_end}:00 IST"
    )


def test_applies_to_comes_from_the_frozensets_the_rules_use() -> None:
    """The scope of a limit is read from `compliance.py`, not restated by hand.

    If someone adds an action to `CONTACTING_ACTIONS` — the way `ESCALATE` was added once a
    telephone call was recognised as contact — quiet hours start applying to it, and this
    endpoint has to say so without anyone editing a list.
    """
    rows = limits()
    contacting = sorted(a.value for a in C.CONTACTING_ACTIONS)
    charging = sorted(a.value for a in C.CHARGING_ACTIONS)
    retrying = sorted(a.value for a in C.RETRY_ACTIONS)
    assert rows["quiet_hours"]["applies_to"] == contacting
    assert rows["contacts_per_day"]["applies_to"] == contacting
    assert rows["rail_ceiling"]["applies_to"] == charging
    assert rows["max_retries"]["applies_to"] == retrying
    assert rows["retry_interval"]["applies_to"] == retrying
    assert rows["mandate_ceiling"]["applies_to"] == retrying


def test_the_actions_it_calls_unconditionally_allowed_really_are() -> None:
    """`WAIT` and `GIVE_UP` approve under conditions that block everything else.

    The endpoint claims spending nothing cannot breach a limit on spending. That is checked
    here against `evaluate` itself, with every dial set to a breach: outside the recovery
    window, in quiet hours, over the contact limit, over the retry limit, over the amount
    ceiling, seconds after the last retry.
    """
    from datetime import datetime, timedelta

    s = get_settings()
    failed = datetime(2026, 3, 1, 2, 0, tzinfo=C.IST)
    now = failed + timedelta(hours=s.max_recovery_window_hours + 10)

    hostile = dict(
        amount_paise=99_999_999,
        at=now,
        failed_at=failed,
        retries_made=s.max_retries_per_payment + 5,
        contacts_today=s.max_contacts_per_day + 5,
        minutes_since_last_retry=0.0,
        has_mandate=False,
        method=PaymentMethod.UPI,
    )

    for name in policy()["always_allowed"]["actions"]:
        verdict = C.evaluate(action=ActionType[name], **hostile)
        assert verdict.approved, f"{name} was blocked: {verdict.blocked_by}"

    # The same conditions must block a charging action, or the test above proves nothing.
    blocked = C.evaluate(action=ActionType.RETRY, **hostile)
    assert not blocked.approved
    assert blocked.blocked_by
