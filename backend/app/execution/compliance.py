"""Hard limits every recovery action passes through, whatever the model wanted.

Three properties this module is built around, each of which was missing before.

**The decision instant is an argument, not `now()`.** A rule that reads the wall
clock cannot be replayed: re-run the same check tomorrow and it answers
differently, so an audit record of "compliance approved" proves nothing. It also
cannot price a *future* action, which is what a policy deciding whether to hold a
message until morning needs. Every rule here is a function of stated inputs.

**Rules are per action type, not per payment.** Quiet hours exist so we do not
wake people up. A server-side retry on a stored mandate does not wake anybody —
it is a machine calling a machine — and blocking it at 02:00 forfeits the whole
overnight window on exactly the payments a bank outage produced. Contacting the
customer at 02:00 is the thing that is forbidden, and that is what is blocked.
Amount ceilings work the same way: they follow the *rail* the money moves on, not
the kind of message we send about it.

**No rule may have a remedy that does not exist.** Every block here names something
else to do, and that alternative has to be reachable. The flat payment-link ceiling
this module used to enforce failed that test — its remedy was an agent call, the
bench is twenty slots against a four-hundred payment batch, and the measured effect
was thirty-five of the largest failures in a batch dropped untried. See
`method_ceiling_paise`.

**The verdict carries the inputs it was reached on.** `ComplianceCheck.blocked_by`
says which limit bound; `evidence` says what the counters were. Without that an
auditor reading the trail six weeks later can see the answer but cannot check it.

Quiet hours match `RecoveryEnv._is_quiet_hour` exactly — 22:00–08:00 IST, judged
on the decision instant. If the two drifted apart the evaluation would be scoring
a policy against limits the product does not actually enforce.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

from app.audit.event_store import event_store
from app.cache.redis_client import redis_client
from app.config import get_settings
from app.models.schemas import (
    ComplianceCheck,
    FailedPayment,
    PaymentMethod,
    RecoveryDecision,
)
from app.sim.types import ActionType

logger = structlog.get_logger()

IST = timezone(timedelta(hours=5, minutes=30))

CHARGING_ACTIONS = frozenset({ActionType.RETRY, ActionType.SEND_LINK})
"""Actions that put an amount on a rail, and therefore inherit that rail's limits."""

CONTACTING_ACTIONS = frozenset({ActionType.SEND_LINK, ActionType.ESCALATE})
"""Actions a customer experiences. `ESCALATE` is here because an agent telephones
them, which at 02:00 is more intrusive than the SMS we refuse to send."""

RETRY_ACTIONS = frozenset({ActionType.RETRY})

STRATEGY_ACTIONS = {
    "IMMEDIATE_RETRY": ActionType.RETRY,
    "DELAYED_RETRY": ActionType.RETRY,
    "SCHEDULED_RETRY": ActionType.RETRY,
    "LINK_SAME_METHOD": ActionType.SEND_LINK,
    "LINK_ALT_METHOD": ActionType.SEND_LINK,
    "ESCALATE": ActionType.ESCALATE,
    "NO_ACTION": ActionType.GIVE_UP,
}
"""Bridge from the product's strategy vocabulary to the action space the simulator
and the policies use. One table rather than string matching at each call site, so
adding a strategy cannot silently fall through the compliance rules."""


def is_quiet_hour(at: datetime, start: int, end: int) -> bool:
    """Whether `at` falls in quiet hours, judged in IST.

    Naive datetimes are read as IST already, which is what the simulator produces;
    aware ones are converted. Getting this wrong in the other direction — reading a
    UTC timestamp as local — moves every hour by 5:30 and would quietly reclassify
    the entire evening as quiet.
    """
    local = at if at.tzinfo is None else at.astimezone(IST)
    if start > end:  # wraps midnight, e.g. 22 → 8
        return local.hour >= start or local.hour < end
    return start <= local.hour < end


def ist_day(at: datetime) -> str:
    """The IST calendar day `at` belongs to, for per-day budgets.

    Per-day counters keyed on the UTC date roll over at 05:30 IST, which splits an
    Indian morning across two budgets and lets a customer be contacted twice inside
    twenty minutes without either day noticing.
    """
    local = at if at.tzinfo is None else at.astimezone(IST)
    return local.strftime("%Y-%m-%d")


def method_ceiling_paise(method: PaymentMethod | None) -> int | None:
    """The rail's own per-transaction limit, or `None` where we cannot know it.

    UPI is capped by NPCI at ₹1 lakh per P2M transaction, so a collect request above
    it is declined by the network whatever we think of the customer. Cards and
    netbanking are absent deliberately rather than by omission: their limits are set
    per issuer and per customer, we cannot see either, and inventing a number here
    would refuse payments that would have gone through.

    This replaced a flat ceiling that applied to payment *links* of any method, and the
    replacement is a correction rather than a loosening. That rule had the ceilings the
    wrong way round — a payment link is the customer paying, present and authenticated
    on the gateway's own page, so refusing one for ₹75,000 while permitting an
    unattended mandate charge for the same amount is not a defensible risk posture.
    It also left the largest failures with no legal action at all: no retry without a
    mandate, no link over the ceiling, and an agent bench of 20 slots against a
    400-payment batch. Measured, that silently dropped 35 of the biggest payments in a
    batch untried. A limit whose stated remedy does not exist is not a compliance
    posture, it is a way of losing money quietly.
    """
    if method is PaymentMethod.UPI:
        return get_settings().upi_transaction_ceiling_paise
    return None


def evaluate(
    *,
    action: ActionType,
    amount_paise: int,
    at: datetime,
    failed_at: datetime,
    retries_made: int = 0,
    contacts_today: int = 0,
    minutes_since_last_retry: float | None = None,
    has_mandate: bool = False,
    method: PaymentMethod | None = None,
) -> ComplianceCheck:
    """Decide whether one action is permitted. Pure, and replayable from its record.

    No clock, no Redis, no database. Everything the verdict depends on is in the
    signature, so the same arguments always produce the same answer and an audit
    entry can be re-derived from what it stored. The async `ComplianceEngine.check`
    below is only responsible for gathering these arguments.

    `WAIT` and `GIVE_UP` are always approved: spending nothing cannot breach a limit
    on spending, and a stopping rule that could be blocked would be a policy with no
    way to stop.

    `has_mandate` gates the auto-retry amount ceiling rather than the amount alone.
    The ceiling exists so a large sum is not charged without the customer's say-so,
    and a standing mandate *is* their say-so — it is the instrument autopay runs on,
    and re-charging it after a failure is the most ordinary recovery action there is.
    Applying the ceiling regardless would block every subscription above ₹10,000 and,
    measured against the simulator, a third of the retries the policy makes.

    `method` is the rail the money would move on, and it is optional because the
    caller does not always know it — an escalation has no rail, and a link that lets
    the customer choose has no single one either. When it is known, the rail's own
    per-transaction ceiling applies; see `method_ceiling_paise`.
    """
    if action in (ActionType.WAIT, ActionType.GIVE_UP):
        return ComplianceCheck(approved=True)

    settings = get_settings()
    blocked: list[str] = []
    hours_since = (_aware(at) - _aware(failed_at)).total_seconds() / 3600.0

    if hours_since > settings.max_recovery_window_hours:
        blocked.append(
            f"Recovery window expired ({hours_since:.0f}h > "
            f"{settings.max_recovery_window_hours}h max)"
        )

    if action in CONTACTING_ACTIONS:
        if is_quiet_hour(at, settings.quiet_hours_start, settings.quiet_hours_end):
            blocked.append(
                f"Quiet hours ({settings.quiet_hours_start}:00–"
                f"{settings.quiet_hours_end}:00 IST)"
            )
        if contacts_today >= settings.max_contacts_per_day:
            blocked.append(
                f"Daily contact limit reached ({contacts_today}/"
                f"{settings.max_contacts_per_day} per day)"
            )

    if action in CHARGING_ACTIONS:
        ceiling = method_ceiling_paise(method)
        if ceiling is not None and amount_paise > ceiling:
            blocked.append(
                f"₹{amount_paise / 100:,.0f} exceeds the ₹{ceiling / 100:,.0f} "
                f"per-transaction ceiling on {method.value if method else '?'}"
            )

    if action in RETRY_ACTIONS:
        if retries_made >= settings.max_retries_per_payment:
            blocked.append(
                f"Max retries exceeded ({retries_made}/"
                f"{settings.max_retries_per_payment})"
            )
        if (
            minutes_since_last_retry is not None
            and minutes_since_last_retry < settings.min_retry_interval_minutes
        ):
            blocked.append(
                f"Retried {minutes_since_last_retry:.0f}m ago, minimum interval is "
                f"{settings.min_retry_interval_minutes}m"
            )
        if amount_paise > settings.require_action_above_paise and not has_mandate:
            blocked.append(
                f"₹{amount_paise / 100:,.0f} exceeds the auto-retry limit with no "
                "standing mandate — requires customer action"
            )

    if not blocked:
        return ComplianceCheck(approved=True)
    return ComplianceCheck(
        approved=False, blocked_by=blocked, recommendation=_recommendation(blocked)
    )


def _aware(t: datetime) -> datetime:
    """Read a naive timestamp as IST rather than crashing on the subtraction.

    Supabase hands back aware timestamps and the simulator produces naive ones, so
    both reach this module. Mixing them raised `TypeError` inside a broad `except`
    in the batch runner, where it was counted as a failed payment — a compliance
    check that silently never ran.
    """
    return t if t.tzinfo is not None else t.replace(tzinfo=IST)


def _recommendation(blocked: list[str]) -> str:
    """What to do instead. The ordering matters: a payment blocked both by quiet
    hours and by a spent retry budget should be told to wait until morning, not to
    escalate, because waiting is free and an agent call is not."""
    joined = " ".join(blocked).lower()
    if "quiet hours" in joined:
        return "DEFER_TO_MORNING"
    if "contact limit" in joined:
        return "WAIT_NEXT_DAY"
    if "minimum interval" in joined:
        return "WAIT_FOR_INTERVAL"
    if "per-transaction ceiling" in joined:
        # The rail is too small for the amount, not the customer unwilling. Another
        # rail the customer holds will carry it, and there is no reason to spend an
        # agent slot before trying one.
        return "SWITCH_METHOD"
    if "auto-retry limit" in joined:
        return "ESCALATE_TO_AGENT"
    if "max retries" in joined:
        return "SWITCH_TO_PAYMENT_LINK"
    if "window expired" in joined:
        return "LOG_EXCEPTION"
    return "ESCALATE"


class ComplianceEngine:
    """Gathers the counters `evaluate` needs, then defers to it.

    The split is deliberate. Everything that can fail — Redis, a clock, a missing
    contact identifier — lives here, and none of it is allowed to influence the
    verdict except by being passed in as an argument.
    """

    async def check(
        self,
        payment: FailedPayment,
        decision: RecoveryDecision,
        session_retry_count: int = 0,
        *,
        at: datetime | None = None,
        minutes_since_last_retry: float | None = None,
    ) -> ComplianceCheck:
        """Check one decision against the limits, at `at` or else right now."""
        action = STRATEGY_ACTIONS.get(decision.strategy.value, ActionType.SEND_LINK)
        instant = at or datetime.now(IST)
        contacts = 0
        if action in CONTACTING_ACTIONS:
            contacts = await self.contacts_today(self.contact_key(payment), instant)

        verdict = evaluate(
            action=action,
            amount_paise=payment.amount,
            at=instant,
            failed_at=payment.created_at,
            retries_made=session_retry_count,
            contacts_today=contacts,
            minutes_since_last_retry=minutes_since_last_retry,
            has_mandate=payment.is_recurring,
            # The rail the money would actually move on: whatever the decision asked
            # for, falling back to the one that just failed. A retry has no choice —
            # it re-charges the instrument on file — and a link that suggests nothing
            # in particular leaves the method to the customer, which is why the
            # ceiling is only checked when a rail is named.
            method=decision.preferred_method or payment.method,
        )
        logger.info(
            "compliance.blocked" if not verdict.approved else "compliance.approved",
            payment_id=payment.payment_id,
            action=action.value,
            blocked_by=verdict.blocked_by or None,
        )
        return verdict

    async def record_contact(self, identifier: str, at: datetime | None = None) -> None:
        """Spend one of today's contact slots. Expires on its own IST day."""
        if not identifier:
            return
        key = self._counter_key(identifier, at or datetime.now(IST))
        await redis_client.incr(key)
        await redis_client.expire(key, 172_800)

    async def contacts_today(self, identifier: str, at: datetime | None = None) -> int:
        """How many times this customer has already been contacted on `at`'s IST day.

        A missing identifier returns 0 rather than blocking. A payment with no phone
        number and no email cannot be contacted at all, so the contact budget is not
        the rule that should stop it; the executor has nothing to send to.
        """
        if not identifier:
            return 0
        raw = await redis_client.get(self._counter_key(identifier, at or datetime.now(IST)))
        return int(raw or 0)

    @staticmethod
    def contact_key(payment: FailedPayment) -> str:
        """One identity for the contact budget, phone first.

        Budgets used to be counted against whichever field the caller happened to
        pass, so the same customer could spend a phone budget and an email budget on
        the same day and be messaged twice under a limit of one.
        """
        return payment.customer_contact or payment.customer_email or ""

    @staticmethod
    def _counter_key(identifier: str, at: datetime) -> str:
        return f"contact_count:{identifier}:{ist_day(at)}"


compliance_engine = ComplianceEngine()



