"""The audit trail's vocabulary, and the drift that made a whole capability unreachable.

`log_recovery_attempt` used to stamp every action `RETRY_ATTEMPTED` and file the real name under
`event_data["action"]`. Nothing failed. Eight payment links were created against Razorpay's API,
the batch reported `errored: 0`, and 2084 audit rows contained no `PAYMENT_LINK_SENT` at all — so
`SYSTEM_RECOVERED`, the only verdict that credits the system, was unreachable and the dashboard's
headline was pinned at ₹0 by a mislabelled string.

The bug was not inside any one function. It was two modules holding private copies of the same
vocabulary and drifting apart in silence, so most of these tests assert agreement *between* files
rather than behaviour within one.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from app.api import sandbox, webhooks
from app.audit import event_store as store

APP = Path(store.__file__).resolve().parents[1]


class _Recorder:
    """Captures the row the store would insert, and the `in_` filter a reader asks for."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.filters: list[list[str]] = []
        self.data: list[dict] = []

    def table(self, _name: str) -> "_Recorder":
        return self

    def insert(self, row: dict) -> "_Recorder":
        self.rows.append(row)
        return self

    def select(self, *_a, **_k) -> "_Recorder":
        return self

    def eq(self, *_a) -> "_Recorder":
        return self

    def in_(self, _column: str, values) -> "_Recorder":
        self.filters.append(list(values))
        return self

    def order(self, *_a, **_k) -> "_Recorder":
        return self

    def limit(self, _n: int) -> "_Recorder":
        return self

    def execute(self) -> "_Recorder":
        return self


# ── What the writer writes ────────────────────────────────────────────────────


def test_an_action_is_logged_under_its_own_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The event type is the action. This is the one-line fix, asserted at the writer."""
    db = _Recorder()
    monkeypatch.setattr(store, "get_supabase", lambda: db)

    store.event_store.log_recovery_attempt(
        "s_1", "pay_1", store.PAYMENT_LINK_SENT, {"link_id": "plink_1"}
    )

    assert [row["event_type"] for row in db.rows] == [store.PAYMENT_LINK_SENT]
    # The name stays in `event_data` as well. Rows written before the fix carry it only there,
    # and nothing rewrites an append-only log, so both shapes have to remain readable.
    assert db.rows[0]["event_data"]["action"] == store.PAYMENT_LINK_SENT
    assert db.rows[0]["event_data"]["link_id"] == "plink_1"


def test_an_escalation_is_logged_under_a_name_the_readers_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`log_escalation` names the event itself, and the readers used to look for `ESCALATION`.

    Nothing writes that string. A filter naming it matched zero rows for as long as it existed,
    which is why the assertion is membership in the shared tuples rather than a literal.
    """
    db = _Recorder()
    monkeypatch.setattr(store, "get_supabase", lambda: db)

    store.event_store.log_escalation("s_1", "pay_1", "no contact channel on file")

    assert db.rows[0]["event_type"] in store.ACTION_EVENTS
    assert db.rows[0]["event_type"] in store.CUSTOMER_FACING_EVENTS


def _logged_action_names() -> set[str]:
    """Every string literal the app passes to `log_recovery_attempt`, read out of the source."""
    names: set[str] = set()
    for path in APP.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "log_recovery_attempt" or len(node.args) < 3:
                continue
            action = node.args[2]
            if isinstance(action, ast.Constant) and isinstance(action.value, str):
                names.add(action.value)
    return names


def test_every_action_the_app_logs_is_one_the_readers_look_for() -> None:
    """The guard that would have caught the original bug.

    A new strategy that logs `SMS_NUDGE_SENT` without adding it to `ACTION_EVENTS` is invisible
    to attribution: the payment has been acted on and `_last_contact_at` returns `None`, so every
    capture lands outside the ambiguity window and is credited to the customer. That failure has
    no symptom — the counts all move — so it has to be caught here.
    """
    logged = _logged_action_names()
    assert logged, "the scan found no call sites, so it is not proving anything"
    assert logged <= set(store.ACTION_EVENTS), (
        f"logged but unread: {sorted(logged - set(store.ACTION_EVENTS))}"
    )


# ── What the readers read ─────────────────────────────────────────────────────


def test_the_sandbox_and_the_store_share_one_list() -> None:
    """Identity, not equality. An equal copy is exactly what drifted last time."""
    assert sandbox.CONTACT_EVENTS is store.ACTION_EVENTS_READ
    assert sandbox.LINK_EVENT is store.PAYMENT_LINK_SENT


def test_the_attribution_clock_reads_the_shared_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_last_contact_at` is the instant the six-hour window is measured from.

    Its filter used to be `["PAYMENT_LINK_SENT", "ESCALATION"]` and matched neither name, so it
    returned `None` for every payment and the window — the whole mechanism for not claiming an
    unprovable recovery — was never once applied.
    """
    db = _Recorder()
    monkeypatch.setattr(webhooks, "get_supabase", lambda: db)

    assert asyncio.run(webhooks._last_contact_at("pay_1")) is None
    assert db.filters == [list(store.ACTION_EVENTS_READ)]


def test_a_row_written_under_the_old_name_resolves_to_its_action() -> None:
    legacy = {
        "event_type": store.LEGACY_ACTION_EVENT,
        "event_data": {"action": store.PAYMENT_LINK_SENT},
    }
    assert store.action_of(legacy) == store.PAYMENT_LINK_SENT
    assert store.is_customer_facing(legacy)


def test_a_legacy_row_with_nothing_filed_stays_what_it_says() -> None:
    """A retry recorded as a retry is a retry. The fallback must not promote it."""
    bare = {"event_type": store.LEGACY_ACTION_EVENT, "event_data": {}}
    assert store.action_of(bare) == store.LEGACY_ACTION_EVENT
    assert not store.is_customer_facing(bare)


def test_the_old_blanket_name_is_read_but_never_written() -> None:
    """Readers accept both eras; writers only produce the new one."""
    assert store.LEGACY_ACTION_EVENT not in store.ACTION_EVENTS
    assert store.LEGACY_ACTION_EVENT in store.ACTION_EVENTS_READ
    assert set(store.ACTION_EVENTS) < set(store.ACTION_EVENTS_READ)


def test_a_server_side_retry_costs_the_customer_nothing() -> None:
    """Three of the five actions never reach a human, so they must not spend attention.

    `compliance.py` charges only customer-facing actions against the daily contact budget, and
    the sandbox decays responsiveness on the same basis. If a retry counted, both would punish a
    strategy the customer never saw.
    """
    for name in ("RETRY_SCHEDULED", "IMMEDIATE_RETRY_MOCKED", "DELAYED_RETRY_WOKE_UP"):
        assert name in store.ACTION_EVENTS
        assert not store.is_customer_facing({"event_type": name})

    for name in store.CUSTOMER_FACING_EVENTS:
        assert name in store.ACTION_EVENTS, "a message the readers do not count as an action"
        assert store.is_customer_facing({"event_type": name})


