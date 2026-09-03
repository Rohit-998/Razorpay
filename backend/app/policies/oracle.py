"""The upper bound. Sees everything, including what no real system can.

This policy reads the hidden state directly: when the money lands, when the
outage ends, whether the card is already dead, and whether this customer was
going to pay unprompted anyway. It is not a proposal and it is not deployable.
It exists to answer the only question that makes a lift figure interpretable —
*how much money was actually on the table?* Without it, "+₹4.1 lakh recovered"
could be 90% of the achievable value or 30%, and nobody can tell which.

The most instructive thing it does is refuse to act. Perfect play spends nothing
on a customer who is already coming back, because that payment arrives either
way and any message sent first is pure cost. A policy whose recovery *rate*
approaches the oracle's while its *spend* stays far above it has not learned to
recover payments; it has learned to stand next to them.

What "optimal" means here, precisely: the oracle knows the latent state, not the
future coin flips. It never charges an instrument that is structurally incapable
of authorising, and it never waits past the moment one becomes capable. It still
loses the payments the bank happens to decline. That is the right ceiling to
report, because a policy that also knew the random draws would score 100% and
measure nothing.

Two honest limitations, worth naming rather than hiding:

  The metric counts money recovered inside the window, not how quickly. So when
  the oracle knows a customer self-recovers on day three it stops — optimal as
  scored, though a real merchant would still prefer the cash on day one.

  Its per-payment ladder is one-step greedy: free actions first, then one message,
  then an agent if a bench slot was reserved for it. The bench allocation itself
  *is* optimal — equal-cost slots ranked by expected value — but a reserved slot
  that goes unused because the cheap message worked is not re-offered to the next
  payment in line. A full backward induction over the window would beat it
  slightly. It is an upper bound on every policy in this repository, not a proof
  of optimality, and it is labelled as such everywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.models.schemas import PaymentMethod
from app.policies.base import BasePolicy
from app.sim.environment import (
    AGENT_COMPLETION_CEILING,
    AGENT_COMPLETION_FLOOR,
    AGENT_COMPLETION_INTENT_WEIGHT,
    AGENT_CONNECT_CEILING,
    AGENT_CONNECT_FLOOR,
    AGENT_CONNECT_INTENT_WEIGHT,
    ESCALATION_COST_PAISE,
    Episode,
    RecoveryEnv,
)
from app.sim.types import Action, ActionType, Channel, Observation, Tone

ESCALATION_COST_RUPEES = ESCALATION_COST_PAISE / 100.0

MAX_ORACLE_RETRIES = 2
"""Retries cost nothing in rupees but they are not risk-free: each failed one can
get the card blocked by its issuer, which destroys the link and agent routes too.
Two is roughly where the free option stops paying for itself."""


class OraclePolicy(BasePolicy):
    """Optimal play under full knowledge of the latent state."""

    name = "oracle"
    description = (
        "Cheats by construction: reads funds arrival, outage windows, dead "
        "instruments and the customer's unprompted-payment time. Reports the "
        "ceiling, not a result."
    )

    def __init__(self, env: RecoveryEnv) -> None:
        self.env = env
        self._by_id: dict[str, Episode] = {}
        self._agent_slots: set[str] = set()

    def begin(self, episode_count: int) -> None:
        self._by_id = {e.payment_id: e for e in self.env.episodes}
        self._agent_slots = self._allocate_agent_bench()

    def _allocate_agent_bench(self) -> set[str]:
        """Reserve the batch's few agent calls for the payments worth the most.

        The bench is a fixed number of identically priced slots, so ranking by
        expected value and taking the top of the list is exactly optimal — no
        dynamic programme needed. This is also the one thing no per-payment rule
        can do: `rules` commits a slot the moment it meets a payment over its
        floor, with no way to know a larger one lands on Thursday. Ranking a batch
        requires seeing the batch.
        """
        ranked: list[tuple[float, str]] = []
        for ep in self.env.episodes:
            recover_at = ep.customer.self_recover_at
            if recover_at is not None and recover_at <= ep.deadline:
                continue  # already coming back; a call buys nothing but cost
            value = self._agent_call_value(ep)
            if value > ESCALATION_COST_RUPEES:
                ranked.append((value, ep.payment_id))
        ranked.sort(reverse=True)
        return {payment_id for _, payment_id in ranked[: self.env.agent_capacity]}

    # ── Helpers over the hidden state ────────────────────────────────────

    def _best_channel_and_tone(self, ep: Episode) -> tuple[Channel, Tone]:
        """The channel this person actually reads, in the register they answer."""
        persona = ep.customer.persona
        channel = max(persona.channel_response, key=lambda c: persona.channel_response[c])
        tone = max(persona.tone_fit, key=lambda t: persona.tone_fit[t])
        return channel, tone

    def _earliest_workable(
        self, ep: Episode, rail: PaymentMethod, customer_present: bool
    ) -> datetime | None:
        """First moment a charge on `rail` could authorise, or None if never.

        A direct inversion of `RecoveryEnv._charge_succeeds` with the bank's coin
        flip left out. Each clause mirrors one gate in it, including the escapes a
        switched rail gets: a dead instrument and a broken merchant configuration
        are both specific to the method that failed, so moving off it clears them.
        """
        switched = rail is not ep.method
        if ep.instrument_blocked and not switched:
            return None

        at = ep.now
        if not switched and ep.merchant_broken_until is not None:
            at = max(at, ep.merchant_broken_until)
        # Stored credentials only gate a server-side charge. Someone following a
        # link or talking to an agent types their details in fresh.
        if not customer_present and ep.customer.credential_fix_at is not None:
            at = max(at, ep.customer.credential_fix_at)
        # A wallet carries its own float, so switching to one is the only way to
        # pay out of a drained bank account.
        wallet_escape = switched and rail is PaymentMethod.WALLET
        if not wallet_escape and ep.customer.funds_available_at is not None:
            at = max(at, ep.customer.funds_available_at)

        outage = self.env.world.downtime_at(ep.bank, at)
        if outage is not None:
            at = max(at, outage.end + timedelta(minutes=2))
        return at

    def _best_rail(
        self, ep: Episode, customer_present: bool
    ) -> tuple[PaymentMethod, datetime] | None:
        """The rail that becomes chargeable soonest, and the moment it does."""
        options: list[tuple[datetime, str, PaymentMethod]] = []
        for rail in ep.customer.alt_methods:
            at = self._earliest_workable(ep, rail, customer_present)
            if at is not None:
                # Rail name breaks ties so the choice never depends on dict order.
                options.append((at, rail.value, rail))
        if not options:
            return None
        at, _, rail = min(options)
        return rail, at

    def _escalation_value_rupees(self, ep: Episode, at: datetime | None = None) -> float:
        """Expected rupees an agent call placed at `at` brings in, before its fee.

        Three gates in sequence, mirroring `RecoveryEnv._do_escalate`: the agent has
        to reach the customer, the customer has to agree to pay on the call, and
        then at least one of the rails they hold has to authorise. All three lean on
        intent, which is why a call placed early is worth far more than the same
        call placed a day later — and why escalation is near-worthless during a bank
        outage however willing everyone is.
        """
        t = ep.now if at is None else at
        rails = [
            rail for rail in ep.customer.alt_methods
            if (workable := self._earliest_workable(ep, rail, customer_present=True)) is not None
            and workable <= t
        ]
        if not rails:
            return 0.0
        intent = ep.customer.intent_at(t, ep.failed_at)
        connect = min(
            AGENT_CONNECT_CEILING, AGENT_CONNECT_FLOOR + AGENT_CONNECT_INTENT_WEIGHT * intent
        )
        completion = min(
            AGENT_COMPLETION_CEILING,
            AGENT_COMPLETION_FLOOR + AGENT_COMPLETION_INTENT_WEIGHT * intent,
        )
        p = self.env.world.success_probability(ep.bank, t)
        any_rail_authorises = 1.0 - (1.0 - p) ** len(rails)
        return connect * completion * any_rail_authorises * ep.amount_rupees

    def _agent_call_value(self, ep: Episode) -> float:
        """Value of the best-timed agent call on this payment, for bench ranking."""
        present = self._best_rail(ep, customer_present=True)
        if present is None:
            return 0.0
        _, at = present
        if at > ep.deadline:
            return 0.0
        return self._escalation_value_rupees(ep, at)

    @staticmethod
    def _minutes_until(ep: Episode, t: datetime) -> int:
        """Whole minutes to `t`, never zero — a wait that does not move the clock
        burns a step and the harness would report it as a hung policy."""
        return max(1, int((t - ep.now).total_seconds() / 60.0) + 1)

    # ── The decision ─────────────────────────────────────────────────────

    def act(self, obs: Observation) -> Action:
        ep = self._by_id[obs.payment_id]
        customer = ep.customer

        # 1. Free money. They are coming back inside the window on their own, so
        #    every rupee spent chasing them is pure loss — and any message sent
        #    first would also make the recovery unattributable.
        if customer.self_recover_at is not None and customer.self_recover_at <= ep.deadline:
            return Action(
                ActionType.GIVE_UP,
                reason="customer will pay unprompted inside the window; spending would be waste",
            )

        # 2. A link is already open in front of them. Waiting costs nothing and
        #    resolves the only uncertainty that matters for the next decision.
        if ep.pending_click_at is not None and ep.pending_click_at <= ep.deadline:
            return Action(
                ActionType.WAIT,
                wait_minutes=self._minutes_until(ep, ep.pending_click_at),
                reason="our link is already open in front of them; nothing to buy by acting now",
            )

        # 3. The free option. A server-side retry costs zero rupees, so it is taken
        #    ahead of anything that costs money whenever one can actually authorise
        #    — including on a rail the customer holds but did not use, which is how
        #    a dead card or a broken configuration gets escaped for nothing.
        if customer.has_mandate and obs.attempts_made < MAX_ORACLE_RETRIES:
            server = self._best_rail(ep, customer_present=False)
            if server is not None:
                rail, at = server
                if at <= ep.now:
                    return Action(
                        ActionType.RETRY,
                        method=rail,
                        reason=f"a {rail.value} charge can authorise right now and costs nothing",
                    )
                if at <= ep.deadline:
                    minutes = self._minutes_until(ep, at)
                    return Action(
                        ActionType.WAIT,
                        wait_minutes=minutes,
                        reason=f"waiting {minutes}m until a free {rail.value} retry can work",
                    )

        # 4. Nothing free is left, so the customer has to be involved. One message,
        #    on the channel they read, in the register they answer, offering a rail
        #    that will actually authorise by the time they open it.
        present = self._best_rail(ep, customer_present=True)
        if present is None:
            return Action(
                ActionType.GIVE_UP,
                reason="no instrument this customer holds can ever authorise this payment",
            )
        rail, at = present
        if at > ep.deadline:
            return Action(
                ActionType.GIVE_UP,
                reason="nothing can authorise before the recovery window closes",
            )
        if at > ep.now:
            minutes = self._minutes_until(ep, at)
            return Action(
                ActionType.WAIT,
                wait_minutes=minutes,
                reason=f"a link sent now would be opened before {rail.value} works; waiting {minutes}m",
            )
        if obs.contacts_made == 0:
            channel, tone = self._best_channel_and_tone(ep)
            return Action(
                ActionType.SEND_LINK,
                channel=channel,
                method=rail,
                tone=tone,
                reason=f"one message on their best channel and register, offering {rail.value}",
            )

        # 5. The cheap message did not land it. An agent tries every rail with the
        #    customer on the line, which is the strongest action available — and the
        #    scarcest. The slot was reserved for this payment in `begin` by ranking
        #    the whole batch, so spending it here is known to be the best use of it
        #    rather than merely a positive-value use of it.
        if not ep.escalated and obs.payment_id in self._agent_slots:
            if obs.agent_calls_remaining > 0:
                value = self._escalation_value_rupees(ep)
                if value > ESCALATION_COST_RUPEES:
                    return Action(
                        ActionType.ESCALATE,
                        reason=(
                            f"₹{value:,.0f} expected from an agent call against a "
                            f"₹{ESCALATION_COST_RUPEES:,.0f} fee, and this payment ranks "
                            "inside the batch's agent bench"
                        ),
                    )

        return Action(ActionType.GIVE_UP, reason="every route with positive value is exhausted")


