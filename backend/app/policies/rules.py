"""A competent hand-written policy. The real bar to beat.

Every rule below is one an experienced payments engineer would write on a
whiteboard, using only fields a live system actually has. It waits out apparent
outages, refuses to retry an instrument that looks dead, holds off on
insufficient-funds cases until money is likely to have landed, respects quiet
hours, and stops. On an average week it is genuinely good.

Keeping it strong is the point. A learned policy that only beats a strawman has
demonstrated nothing, so this file is written to win. Where it is structurally
weak is worth naming in advance, because those are the gaps the learned policy
has to exploit to justify existing:

  It cannot tell an abandoned OTP from stale card details beyond a crude history
  threshold, so it treats two causes with opposite correct treatments alike.

  It picks channel and tone from amount and response rate, which are weak proxies
  for who someone is. It has no way to learn that a particular kind of customer
  answers Hinglish on WhatsApp and ignores formal English on email.

  Its thresholds are fixed. It cannot notice that this week's outages recover in
  twenty minutes rather than ninety, and it cannot spend its contact budget where
  the marginal rupee is highest.

  It allocates the scarce agent bench first-come, first-served. Any payment over
  a fixed rupee floor takes a slot the moment it meets one, so a Tuesday morning
  of ₹30,000 failures can spend the whole week's calls before a ₹2,00,000 one
  arrives. Ranking a batch requires seeing the batch, which a rule written per
  payment structurally cannot do.
"""

from __future__ import annotations

from app.models.schemas import PaymentMethod
from app.policies.base import BasePolicy
from app.sim.types import Action, ActionType, Channel, Observation, Tone

TRANSIENT_REASONS = {"timeout", "network_error", "gateway_technical_error", "upi_psp_error"}
DEAD_INSTRUMENT_REASONS = {"card_blocked", "invalid_card", "mandate_expired"}
FUNDS_REASONS = {"insufficient_funds", "limit_exceeded"}
AUTH_REASONS = {"authentication_failed", "payment_cancelled"}
CONFIG_REASONS = {"bank_not_enabled"}

MAX_RETRIES = 2
MAX_CONTACTS = 2
ESCALATION_FLOOR_RUPEES = 25_000.0
"""Below this, a ₹90 agent call cannot pay for itself often enough to be worth one
of the bench's few slots. A fixed floor is the honest version of what a human
would write — and it is also the weakness, because it allocates first-come,
first-served and cannot tell that a later ₹80,000 payment is coming."""

SELF_RECOVERY_GRACE_MINUTES = 25.0
"""Transient failures mostly fix themselves. Spending inside this window buys
payments that were already on their way."""


class RulesPolicy(BasePolicy):
    """Expert heuristics over observable fields only."""

    name = "rules"
    description = (
        "Hand-written expert policy: waits out apparent outages, never retries a "
        "dead-looking instrument, defers funds failures, respects quiet hours, "
        "and stops after two retries and two contacts."
    )

    # ── Channel and tone, from what little is observable ──────────────────

    def _channel(self, obs: Observation) -> Channel:
        if obs.customer_signal.response_rate >= 0.35:
            return Channel.WHATSAPP
        if obs.amount_rupees >= 1_000:
            return Channel.SMS
        return Channel.EMAIL

    def _tone(self, obs: Observation) -> Tone:
        if obs.amount_rupees >= 10_000:
            return Tone.INFORMATIONAL
        if obs.minutes_since_failure > 720:
            return Tone.URGENT
        return Tone.FRIENDLY

    def _alternative(self, obs: Observation) -> PaymentMethod | None:
        others = [m for m in obs.available_methods if m is not obs.method]
        if not others:
            return None
        if PaymentMethod.UPI in others:
            return PaymentMethod.UPI
        return others[0]

    def _quiet_hour_wait(self, obs: Observation) -> Action | None:
        """Hold a message until 08:00 rather than burning it overnight."""
        hour = obs.now.hour
        if 8 <= hour < 22:
            return None
        minutes = (8 - hour) * 60 if hour < 8 else (32 - hour) * 60
        return Action(
            ActionType.WAIT,
            wait_minutes=minutes - obs.now.minute,
            reason="holding outreach until 08:00 — quiet-hours messages go unread",
        )

    def _send(self, obs: Observation, method: PaymentMethod | None, why: str) -> Action:
        return Action(
            ActionType.SEND_LINK,
            channel=self._channel(obs),
            method=method or obs.method,
            tone=self._tone(obs),
            reason=why,
        )

    def _escalation_available(self, obs: Observation) -> bool:
        """Whether an agent call is worth asking for, and whether one is even left."""
        return (
            not obs.escalated
            and obs.agent_calls_remaining > 0
            and obs.amount_rupees >= ESCALATION_FLOOR_RUPEES
        )

    # ── The decision ─────────────────────────────────────────────────────

    def act(self, obs: Observation) -> Action:
        reason = obs.error_reason
        elapsed = obs.minutes_since_failure
        exhausted = obs.attempts_made >= MAX_RETRIES and obs.contacts_made >= MAX_CONTACTS

        if exhausted:
            if self._escalation_available(obs):
                return Action(
                    ActionType.ESCALATE,
                    reason=f"₹{obs.amount_rupees:,.0f} justifies an agent after automation failed",
                )
            return Action(ActionType.GIVE_UP, reason="retry and contact budgets are spent")

        # 1. A dead-looking instrument. Retrying it is the classic value
        #    destruction: it cannot work, and it can get the card blocked.
        if reason in DEAD_INSTRUMENT_REASONS:
            alternative = self._alternative(obs)
            if alternative is None:
                return Action(
                    ActionType.GIVE_UP,
                    reason=f"'{reason}' with no alternative method on file",
                )
            if obs.contacts_made == 0:
                quiet = self._quiet_hour_wait(obs)
                return quiet or self._send(
                    obs, alternative, f"'{reason}' suggests the instrument is gone; offering {alternative.value}"
                )
            return Action(ActionType.GIVE_UP, reason="alternative rail already offered")

        # 2. Our own configuration looks broken. Same treatment, different cause.
        if reason in CONFIG_REASONS:
            alternative = self._alternative(obs)
            if alternative is not None and obs.contacts_made == 0:
                quiet = self._quiet_hour_wait(obs)
                return quiet or self._send(
                    obs, alternative, "method appears misconfigured on our side; switching rails"
                )
            return Action(ActionType.GIVE_UP, reason="no working method available for this payment")

        # 3. Correlated failures on this bank. Retrying into a wall is wasted
        #    spend; the same retry after recovery works. Bounded, because an
        #    outage that has run for hours may not be an outage at all — and
        #    waiting forever is indistinguishable from doing nothing.
        if obs.bank_signal.concurrent_failure_spike and elapsed < 4 * 60:
            return Action(
                ActionType.WAIT,
                wait_minutes=45,
                reason=(
                    f"{obs.bank_signal.observed_failure_count_1h} failures on {obs.bank} in the "
                    "last hour — waiting for the bank rather than retrying into an outage"
                ),
            )

        # 4. No money in the account. Nothing to do but wait for it, ideally
        #    until the next morning when salary credits land.
        if reason in FUNDS_REASONS:
            if elapsed < 12 * 60:
                return Action(
                    ActionType.WAIT,
                    wait_minutes=int(min(720, max(60, 12 * 60 - elapsed))),
                    reason="balance was short; retrying now cannot succeed",
                )
            if obs.has_mandate and obs.attempts_made < MAX_RETRIES:
                return Action(
                    ActionType.RETRY,
                    method=obs.method,
                    reason="enough time has passed for a credit to have landed",
                )
            if obs.contacts_made < MAX_CONTACTS:
                quiet = self._quiet_hour_wait(obs)
                return quiet or self._send(obs, None, "asking the customer to complete when funded")
            return Action(ActionType.GIVE_UP, reason="funds may still be short and outreach is spent")

        # 5. An authentication failure. The same error text covers an abandoned
        #    OTP and details that are simply wrong, and the correct treatments are
        #    opposites: one wants a nudge, the other must never be retried
        #    server-side because the stored details cannot authenticate. History is
        #    the only observable that separates them, and it is a blunt instrument.
        if reason in AUTH_REASONS:
            if elapsed < SELF_RECOVERY_GRACE_MINUTES:
                return Action(
                    ActionType.WAIT,
                    wait_minutes=int(SELF_RECOVERY_GRACE_MINUTES - elapsed) + 1,
                    reason="most abandoned authentications complete unaided within the half hour",
                )
            details_look_stale = (
                obs.customer_signal.success_rate_90d < 0.75
                or obs.customer_signal.failure_count_7d >= 3
            )
            if details_look_stale:
                if obs.contacts_made < MAX_CONTACTS:
                    quiet = self._quiet_hour_wait(obs)
                    return quiet or self._send(
                        obs, None, "history suggests the details on file are stale; asking for fresh ones"
                    )
                return Action(ActionType.GIVE_UP, reason="stale details and the customer is not responding")
            if obs.has_mandate and obs.attempts_made < MAX_RETRIES:
                return Action(
                    ActionType.RETRY,
                    method=obs.method,
                    reason="reliable payer, so this looks like an abandoned challenge rather than bad details",
                )
            if obs.contacts_made < MAX_CONTACTS:
                quiet = self._quiet_hour_wait(obs)
                return quiet or self._send(obs, None, "nudging the customer to finish authenticating")
            return Action(ActionType.GIVE_UP, reason="authentication nudges exhausted")

        # 6. Transient technical failure. Wait briefly — a lot of these fix
        #    themselves — then retry, which is the cheapest action there is.
        if reason in TRANSIENT_REASONS:
            if elapsed < SELF_RECOVERY_GRACE_MINUTES:
                return Action(
                    ActionType.WAIT,
                    wait_minutes=int(SELF_RECOVERY_GRACE_MINUTES - elapsed) + 1,
                    reason="transient failures largely resolve on their own; not spending yet",
                )
            if obs.has_mandate and obs.attempts_made < MAX_RETRIES:
                return Action(
                    ActionType.RETRY, method=obs.method, reason="transient failure, conditions look normal"
                )
            if obs.contacts_made < MAX_CONTACTS:
                quiet = self._quiet_hour_wait(obs)
                return quiet or self._send(obs, None, "no mandate on file, so the customer must complete it")
            return Action(ActionType.GIVE_UP, reason="transient recovery attempts exhausted")

        # 7. Everything else — `payment_failed` and `other`, which together are the
        #    largest bucket in production and carry almost no information. Try the
        #    cheap thing once, ask once, then stop.
        if obs.has_mandate and obs.attempts_made < 1:
            return Action(
                ActionType.RETRY, method=obs.method, reason="uninformative error; one cheap attempt"
            )
        if obs.contacts_made < 1:
            quiet = self._quiet_hour_wait(obs)
            return quiet or self._send(obs, None, "uninformative error; asking the customer directly")
        if self._escalation_available(obs):
            return Action(
                ActionType.ESCALATE,
                reason=f"₹{obs.amount_rupees:,.0f} at stake and no diagnosis from the error fields",
            )
        return Action(ActionType.GIVE_UP, reason="no remaining action with positive expected value")
