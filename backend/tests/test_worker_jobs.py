"""The stubs, and the two places where the audit trail recorded something that never happened.

Three defects of the same shape, all of which a test suite reading only aggregates would miss.

`executor` logged `RETRY_SCHEDULED` with the enqueue commented out, so a delayed retry was a
line in the audit trail and nothing else. The session stayed `OPEN` waiting for a wake-up that
did not exist, and the bandit was rewarded or punished for a strategy with no mechanism behind
it.

`execute_delayed_retry` logged `DELAYED_RETRY_EXECUTED` and returned. Even had the enqueue been
live, the wake-up did nothing.

`razorpay_client.create_payment_link` returned `{"id": "plink_mock123"}` when unconfigured. The
link id is the one field that carries causation — `payment_link.paid` names it, which is how a
recovery becomes `SYSTEM_RECOVERED` rather than `AMBIGUOUS` — so a constant placeholder is both
counterfeit and shared by every payment in the batch.

What links the three is that none of them errors. Each returns success, writes a plausible
audit row, and leaves the money unrecovered.
"""

from __future__ import annotations

import asyncio

import pytest

from app import queue as queue_module
from app import worker
from app.execution import executor as executor_module
from app.execution import razorpay_client as client_module
from app.models.schemas import (
    ErrorSource,
    FailedPayment,
    PaymentMethod,
    RecoveryDecision,
    RecoveryStrategy,
)
from datetime import datetime


def _run(coro):
    return asyncio.run(coro)


def _payment(**overrides) -> FailedPayment:
    fields = dict(
        payment_id="pay_delayed_1",
        order_id="order_1",
        amount=50_000_00,
        method=PaymentMethod.CARD,
        bank="HDFC",
        error_code="BAD_REQUEST_ERROR",
        error_source=ErrorSource.GATEWAY,
        error_step="payment_authorization",
        error_reason="insufficient_funds",
        error_description="not enough balance",
        customer_contact="+919000000001",
        customer_email="someone@example.com",
        created_at=datetime(2026, 3, 24, 12, 0),
    )
    fields.update(overrides)
    return FailedPayment(**fields)


def _decision(strategy: RecoveryStrategy, **overrides) -> RecoveryDecision:
    fields = dict(
        strategy=strategy,
        decided_by="bandit",
        confidence=0.7,
        reasoning="because the account is short and payday is tomorrow",
    )
    fields.update(overrides)
    return RecoveryDecision(**fields)


class _Recorder:
    """What the executor did, in the order it did it."""

    def __init__(self) -> None:
        self.enqueued: list[tuple] = []
        self.attempts: list[tuple[str, dict]] = []
        self.exceptions: list[tuple[str, str]] = []
        self.contacts: list[str] = []
        self.enqueue_succeeds = True

    def event(self, name: str) -> dict | None:
        for attempt, data in self.attempts:
            if attempt == name:
                return data
        return None


@pytest.fixture()
def acted(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()

    async def fake_enqueue(job: str, *args, defer_minutes: int = 0) -> bool:
        rec.enqueued.append((job, args, defer_minutes))
        return rec.enqueue_succeeds

    class _Events:
        def log_recovery_attempt(self, _s, _p, name, data=None):
            rec.attempts.append((name, data or {}))

        def log_exception(self, _s, _p, reason, category):
            rec.exceptions.append((reason, category))

        def log_escalation(self, *_a, **_k):
            pass

    class _Compliance:
        def contact_key(self, payment) -> str:
            return payment.customer_contact or ""

        async def record_contact(self, key: str) -> None:
            rec.contacts.append(key)

    monkeypatch.setattr(executor_module, "enqueue", fake_enqueue)
    monkeypatch.setattr(executor_module, "event_store", _Events())
    monkeypatch.setattr(executor_module, "compliance_engine", _Compliance())
    return rec


# ── A scheduled retry is now actually scheduled ───────────────────────────────


@pytest.mark.parametrize(
    "strategy", [RecoveryStrategy.DELAYED_RETRY, RecoveryStrategy.SCHEDULED_RETRY]
)
def test_scheduling_a_retry_puts_a_job_on_the_queue(
    acted: _Recorder, strategy: RecoveryStrategy
) -> None:
    """The whole defect, in one assertion: the delay has to exist somewhere other than a log."""
    assert _run(
        executor_module.executor.execute(
            _payment(), "sess-1", _decision(strategy, delay_minutes=90)
        )
    )
    assert acted.enqueued == [
        ("execute_delayed_retry", ("pay_delayed_1", "sess-1"), 90)
    ]
    assert acted.event("RETRY_SCHEDULED") == {"delay_minutes": 90}


def test_the_delay_the_policy_asked_for_is_the_delay_that_is_used(acted: _Recorder) -> None:
    """`delay_minutes` is the policy's judgement about when the world will have changed —
    payday, or the end of an outage. Substituting a default would discard it silently."""
    _run(
        executor_module.executor.execute(
            _payment(), "sess-1", _decision(RecoveryStrategy.DELAYED_RETRY, delay_minutes=1440)
        )
    )
    assert acted.enqueued[0][2] == 1440


def test_an_unreachable_queue_is_reported_as_an_action_not_taken(acted: _Recorder) -> None:
    """The failure mode that mattered. Redis Cloud is a network hop, and a retry that was
    never queued must not be credited: `execute` returning True increments the retry count and
    feeds the bandit's posterior, so a phantom retry teaches the policy that waiting works."""
    acted.enqueue_succeeds = False
    assert not _run(
        executor_module.executor.execute(
            _payment(), "sess-1", _decision(RecoveryStrategy.DELAYED_RETRY)
        )
    )
    assert acted.event("RETRY_SCHEDULED") is None, "not recorded as scheduled"
    assert acted.exceptions and "queue unreachable" in acted.exceptions[0][0]


def test_scheduling_a_retry_does_not_spend_a_contact(acted: _Recorder) -> None:
    """A retry is server-side. Charging it against the customer's two-messages-a-day budget
    would make the policy quieter than compliance requires and lose recoverable money."""
    _run(
        executor_module.executor.execute(
            _payment(), "sess-1", _decision(RecoveryStrategy.DELAYED_RETRY)
        )
    )
    assert acted.contacts == []


# ── The wake-up re-runs the pipeline rather than charging blind ────────────────


class _Session:
    def __init__(self, status: str) -> None:
        self.data = {"status": status}

    def table(self, _name):
        return self

    def select(self, *_a):
        return self

    def eq(self, *_a):
        return self

    def single(self):
        return self

    def execute(self):
        return self


@pytest.fixture()
def woken(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {"reprocessed": [], "attempts": []}

    async def fake_process(_ctx, payment_id: str) -> None:
        seen["reprocessed"].append(payment_id)

    class _Events:
        def log_recovery_attempt(self, _s, _p, name, data=None):
            seen["attempts"].append(name)

    monkeypatch.setattr(worker, "process_failed_payment", fake_process)
    monkeypatch.setattr(worker, "event_store", _Events())
    return seen


def test_a_wake_up_re_enters_the_pipeline(
    woken: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It does not charge the card. The reason for waiting was that the world was wrong —
    bank down, account short, details stale — so the only useful thing to do on waking is to
    look again: fresh features, fresh classification, fresh compliance check against the
    retry count as it now stands."""
    monkeypatch.setattr(worker, "get_supabase", lambda: _Session("OPEN"))
    _run(worker.execute_delayed_retry({}, "pay_1", "sess-1"))
    assert woken["reprocessed"] == ["pay_1"]
    assert woken["attempts"] == ["DELAYED_RETRY_WOKE_UP"]


@pytest.mark.parametrize("status", ["RECOVERED", "FAILED", "ESCALATED"])
def test_a_payment_settled_while_asleep_is_left_alone(
    woken: dict, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """A session is `OPEN` only while the outcome is unknown. Retrying one that closed while
    we were asleep charges a customer who has already paid."""
    monkeypatch.setattr(worker, "get_supabase", lambda: _Session(status))
    _run(worker.execute_delayed_retry({}, "pay_1", "sess-1"))
    assert woken["reprocessed"] == []
    assert woken["attempts"] == []


def test_the_recovery_window_is_not_re_implemented_in_the_worker() -> None:
    """Deliberate absence, asserted so nobody helpfully adds it.

    A payment that wakes up past its 72 hours must not be retried — and that rule already
    lives in `compliance.evaluate`, which returns `LOG_EXCEPTION` and gets written off with
    its reason. A second copy of the arithmetic in the worker is how the limit the eval
    measures and the limit production enforces drift apart.
    """
    import inspect

    source = inspect.getsource(worker.execute_delayed_retry)
    assert "max_recovery_window_hours" not in source


# ── Bank health is polled, and an unanswered poll is not good news ─────────────


class _Store:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str, bool]] = []

    async def set_bank_downtime(self, bank: str, severity: str, is_down: bool) -> None:
        self.writes.append((bank, severity, is_down))


@pytest.fixture()
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    s = _Store()

    class _Extractor:
        pass

    extractor = _Extractor()
    extractor.store = s
    monkeypatch.setattr(worker, "feature_extractor", extractor)
    return s


def _feed(monkeypatch: pytest.MonkeyPatch, result) -> None:
    async def fake_fetch():
        return result

    monkeypatch.setattr(worker.razorpay_client, "fetch_downtimes", fake_fetch)


def test_a_declared_outage_is_written_to_the_feature_store(
    store: _Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one input to the decision no single payment can supply.

    A failure on HDFC says almost nothing alone. HDFC being in a declared outage says that
    retrying now throws an attempt off a budget of three, and that waiting is not
    procrastination. The classifier's bank-health feature reads what this writes, so the
    empty stub meant every payment was scored as though the world were healthy.
    """
    _feed(monkeypatch, [
        {"bank": "HDFC", "method": "netbanking", "severity": "high", "status": "started"},
        {"bank": "ICIC", "method": "card", "severity": None, "status": "started"},
    ])
    _run(worker.poll_bank_downtimes({}))
    assert ("HDFC", "high", True) in store.writes
    # Razorpay does not always send a severity. Defaulting to `medium` keeps the bank flagged;
    # defaulting to nothing would drop the outage entirely.
    assert ("ICIC", "medium", True) in store.writes


def test_a_bank_that_came_back_has_its_flag_cleared(
    store: _Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A downtime never cleared is worse than one never recorded: it teaches the policy to
    wait forever on a bank that is fine, and the payment ages out of its window asleep."""
    ctx: dict = {}
    _feed(monkeypatch, [{"bank": "HDFC", "severity": "high", "status": "started"}])
    _run(worker.poll_bank_downtimes(ctx))
    assert ctx["banks_down"] == {"HDFC"}

    store.writes.clear()
    _feed(monkeypatch, [])
    _run(worker.poll_bank_downtimes(ctx))
    assert store.writes == [("HDFC", "none", False)]
    assert ctx["banks_down"] == set()


def test_a_failed_poll_does_not_declare_the_world_healthy(
    store: _Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`None` and `[]` are different answers and this is why the client distinguishes them.

    An unreachable API means we do not know. Treating that as "nothing is down" would clear
    every outage flag at exactly the moment the system has no information, which is the one
    time being wrong is most expensive.
    """
    ctx = {"banks_down": {"HDFC"}}
    _feed(monkeypatch, None)
    _run(worker.poll_bank_downtimes(ctx))
    assert store.writes == []
    assert ctx["banks_down"] == {"HDFC"}, "still believed down, because we did not find out"


def test_the_poller_runs_on_a_schedule_rather_than_being_a_registered_job() -> None:
    """It was in `functions` with the cron commented out — so it was callable and never
    called. A five-minute cron, and an outage noticed twenty minutes late has already cost
    the retries taken during it."""
    names = [f.__name__ for f in worker.WorkerSettings.functions]
    assert "poll_bank_downtimes" not in names
    assert worker.WorkerSettings.cron_jobs, "the cron is enabled, not commented out"


# ── A payment link id is either real or absent ─────────────────────────────────


def test_an_unconfigured_client_refuses_rather_than_inventing_a_link_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`plink_mock123` was a constant, and the link id is what carries causation.

    `payment_link.paid` names the link that was paid, which is how a recovery becomes
    `SYSTEM_RECOVERED` instead of `AMBIGUOUS`. A placeholder id is written to the audit trail
    as though it were real, can never be paid, and is shared by every payment in the batch —
    so the field the attribution verdict rests on would have been counterfeit and
    non-unique at the same time.
    """
    client = client_module.RazorpayClient()
    monkeypatch.setattr(client.settings, "razorpay_key_id", "")
    link, refusal = _run(
        client.create_payment_link(
            amount=50_000_00, currency="INR", reference_id="pay_1",
            description="d", customer_contact="+919000000001",
            customer_email="someone@example.com",
        )
    )
    assert link is None
    assert "RAZORPAY_KEY_ID" in (refusal or ""), "the refusal names what is missing"


def test_no_placeholder_link_id_survives_anywhere_in_the_client() -> None:
    """The literal itself, because the failure it caused was silent and a helpful
    reinstatement would be too.

    Docstrings are stripped before the scan: the module explains at length what
    `plink_mock123` was and why returning it was worse than returning nothing, and that
    explanation is the reason the id is unlikely to come back. What must not exist is a
    placeholder in code that some caller could receive.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(client_module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is not None:
                node.body = node.body[1:]
    assert "plink_mock123" not in ast.unparse(tree)


def test_a_link_that_could_not_be_created_is_not_credited_as_an_action(
    acted: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`None` from the client has to reach the caller as a failed action.

    Otherwise the session is closed as though a link had gone out, the contact is charged
    against the customer's daily budget, and the bandit updates on an arm that never fired.
    """
    async def no_link(**_kwargs):
        return None, "Razorpay answered 429: too many requests (gave up after 4 attempts)"

    monkeypatch.setattr(client_module.razorpay_client, "create_payment_link", no_link)
    assert not _run(
        executor_module.executor.execute(
            _payment(), "sess-1", _decision(RecoveryStrategy.LINK_SAME_METHOD)
        )
    )
    assert acted.contacts == [], "a link that does not exist did not contact anyone"
    assert acted.exceptions
    assert "429" in acted.exceptions[-1][0], (
        "the exception ledger carries the reason, not just the fact — a throttle is a thing a "
        "reviewer can fix and an unset key is a different thing"
    )


# ── The rate limit that made the provable path unreachable ────────────────────
#
# A live batch of 191 payments sent forty link creations inside four seconds. Razorpay answered
# 429 to all but the first few, the client returned `None`, the executor logged an exception, and
# `/batch/run` reported `processed: 191, errored: 0`. Nothing looked wrong anywhere. But
# `PAYMENT_LINK_SENT` is the only audit event that can produce a `SYSTEM_RECOVERED` verdict, so
# the throttle had quietly deleted the one recovery path the system can prove — and the
# dashboard's headline sat at ₹0 while every other number moved.


class _Reply:
    """Just enough of `httpx.Response` for the client's own branching."""

    def __init__(self, status: int, body: dict | None = None, headers: dict | None = None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.reason_phrase = "Too Many Requests" if status == 429 else "Error"
        self.text = str(self._body)

    def json(self) -> dict:
        return self._body


def _serving(replies: list[_Reply], monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Point the client at a scripted sequence of replies. Returns the call log."""
    calls: list[int] = []

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def post(self, *_a, **_k):
            reply = replies[min(len(calls), len(replies) - 1)]
            calls.append(reply.status_code)
            return reply

    monkeypatch.setattr(client_module.httpx, "AsyncClient", lambda *a, **k: _FakeClient())
    # The waits are the point of the fix but not of the test; asserting on them here would
    # spend fifteen real seconds proving arithmetic.
    monkeypatch.setattr(client_module.asyncio, "sleep", _instant)
    return calls


async def _instant(_seconds):
    return None


def _link(client) -> tuple:
    return _run(
        client.create_payment_link(
            amount=50_000_00, currency="INR", reference_id="pay_1", description="d",
            customer_contact="+919000000001", customer_email="someone@example.com",
        )
    )


def test_a_throttled_link_is_retried_rather_than_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole defect in one test. Before the retry, this returned `(None, ...)` and the
    payment's only provable recovery path was gone for the rest of the batch."""
    client = client_module.RazorpayClient()
    calls = _serving(
        [_Reply(429), _Reply(429), _Reply(200, {"id": "plink_real", "short_url": "https://x"})],
        monkeypatch,
    )
    link, refusal = _link(client)
    assert refusal is None
    assert link and link["id"] == "plink_real"
    assert calls == [429, 429, 200], "it kept asking, and it stopped as soon as it succeeded"


def test_a_persistent_throttle_gives_up_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bounded, because a queue of retries against a limit that is not lifting is a slower way
    to fail. The reason names the status so the ledger is actionable."""
    client = client_module.RazorpayClient()
    calls = _serving([_Reply(429, {"error": {"description": "too many requests"}})], monkeypatch)
    link, refusal = _link(client)
    assert link is None
    assert len(calls) == client_module.MAX_ATTEMPTS
    assert "429" in refusal and "too many requests" in refusal
    assert str(client_module.MAX_ATTEMPTS) in refusal


def test_a_rejected_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 is a malformed request; sending it again spends another slot on the same
    rejection — and under a rate limit those slots are what the next link needs."""
    client = client_module.RazorpayClient()
    calls = _serving(
        [_Reply(400, {"error": {"description": "Recurring digits in customer contact are disallowed"}})],
        monkeypatch,
    )
    link, refusal = _link(client)
    assert link is None
    assert calls == [400], "one attempt, because the second would be rejected identically"
    assert "Recurring digits" in refusal, "Razorpay's own sentence, not ours"


def test_writes_are_spaced_so_a_batch_does_not_arrive_as_a_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry recovers from a throttle; the spacing is what keeps the batch from causing one.
    Asserted on the gate itself, because the observable effect is elapsed time."""
    client = client_module.RazorpayClient()
    waited: list[float] = []

    async def record(seconds):
        waited.append(seconds)

    monkeypatch.setattr(client_module.asyncio, "sleep", record)
    monkeypatch.setattr(client_module.time, "monotonic", lambda: 100.0)
    client._last_write = 100.0  # a write just happened
    _run(client._space_out())
    assert waited == [pytest.approx(client_module.MIN_INTERVAL_SECONDS)]


# ── The 429 that is not a rate limit ──────────────────────────────────────────
#
# Razorpay's test mode allows thirty payment links per account, ever, and answers the
# thirty-first with 429 — the same status it uses for "you are going too fast". The retry above
# was therefore spending four attempts and eight seconds of backoff per payment on an answer
# that cannot change, and filing the result as `EXECUTION_ERROR`, which reads as a bug in the
# executor rather than a credential that has run out. This was found on live traffic: the
# exception ledger held `no payment link: Razorpay answered 429: test mode limit of 30 reached
# for payment_link (gave up after 4 attempts)`, and it is the reason `provably_ours` stops
# moving on a sandbox key no matter how many batches are run.


def test_an_exhausted_allowance_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """One attempt, because the thirty-first link is refused exactly as the thirty-fourth is.

    The retry budget exists for a throttle that lifts. Spending it here delays every remaining
    payment in the batch to arrive at the same refusal.
    """
    client = client_module.RazorpayClient()
    calls = _serving(
        [_Reply(429, {"error": {"description": "test mode limit of 30 reached for payment_link"}})],
        monkeypatch,
    )
    link, refusal = _link(client)
    assert link is None
    assert calls == [429], "asked once, because asking again cannot change the answer"
    assert refusal.startswith(client_module.QUOTA_PREFIX)
    assert "test mode limit of 30" in refusal, "Razorpay's own sentence survives"
    assert "gave up after" not in refusal, "it did not give up; it was told no"


def test_a_real_throttle_is_still_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half, and the one that would hurt if the match were loose.

    `Too many requests` clears in a second. Reading it as a permanent refusal would drop links
    that a one-second pause would have created — and `PAYMENT_LINK_SENT` is the only event that
    can produce a provable recovery, so each one dropped is a recovery the system cannot claim.
    """
    client = client_module.RazorpayClient()
    calls = _serving(
        [
            _Reply(429, {"error": {"description": "Too many requests"}}),
            _Reply(200, {"id": "plink_real", "short_url": "https://x"}),
        ],
        monkeypatch,
    )
    link, refusal = _link(client)
    assert refusal is None and link["id"] == "plink_real"
    assert calls == [429, 200]


def test_the_quota_test_only_fires_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 503 whose body happens to mention a limit is an outage, not an allowance. The check is
    scoped to the status code that is actually ambiguous."""
    assert client_module.is_quota_exhausted(429, "test mode limit of 30 reached")
    assert not client_module.is_quota_exhausted(503, "test mode limit of 30 reached")
    assert not client_module.is_quota_exhausted(429, "Too many requests")


def test_an_exhausted_allowance_is_filed_under_its_own_name(
    acted: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The category is the point. `EXECUTION_ERROR` on 700 sessions says this code is broken;
    `GATEWAY_QUOTA_EXHAUSTED` says the key has spent its thirty links and the provable path is
    closed until a live key replaces it. One is a bug report, the other is a fact about the
    environment, and the demo has to be able to say which it is looking at."""
    async def spent(**_kwargs):
        return None, f"{client_module.QUOTA_PREFIX}: test mode limit of 30 reached"

    monkeypatch.setattr(client_module.razorpay_client, "create_payment_link", spent)
    assert not _run(
        executor_module.executor.execute(
            _payment(), "sess-1", _decision(RecoveryStrategy.LINK_SAME_METHOD)
        )
    )
    assert acted.contacts == [], "no link means no contact charged against the customer"
    reason, category = acted.exceptions[-1][0], acted.exceptions[-1][1]
    assert category == "GATEWAY_QUOTA_EXHAUSTED"
    assert "test mode limit of 30" in reason


def test_an_ordinary_execution_failure_keeps_the_ordinary_category(
    acted: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new category must not swallow the old one — a persistent throttle is still an
    execution error, and calling it a spent allowance would tell a reviewer to change
    credentials when what they need to do is slow down."""
    async def throttled(**_kwargs):
        return None, "Razorpay answered 429: Too many requests (gave up after 4 attempts)"

    monkeypatch.setattr(client_module.razorpay_client, "create_payment_link", throttled)
    _run(
        executor_module.executor.execute(
            _payment(), "sess-1", _decision(RecoveryStrategy.LINK_SAME_METHOD)
        )
    )
    assert acted.exceptions[-1][1] == "EXECUTION_ERROR"
