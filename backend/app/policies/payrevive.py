"""`payrevive` — the policy this project is actually proposing.

`rules` is a good hand-written policy and it recovers about 46% of what is there
to recover. Its docstring names the four reasons it cannot do better, and every one
of them is structural rather than a matter of tuning:

  1. It allocates the scarce agent bench first-come, first-served over a fixed rupee
     floor, so an ordinary Tuesday can spend the week's calls before the large
     failures arrive. Ranking a batch requires seeing the batch.
  2. Its thresholds are fixed numbers. It waits 45 minutes for an outage whether the
     bank is clearing in twenty minutes or is still dark at hour three.
  3. It never re-charges on a *different* rail. It offers alternatives only inside a
     paid message and hopes the customer acts.
  4. It stops when a counter runs out, not when the next action stops being worth
     taking.

This policy replaces all four with one calculation. For every action it could take
right now it computes expected rupees:

    value = p_success × amount − cost − (1 − p_success) × collateral

and takes the best one, or stops if nothing clears zero. `p_success` is not written
here. It is read from `calibration.json`, fitted from action logs on seeds disjoint
from the ones this is scored on — see `app.policies.calibration`. That is the whole
difference in kind between this and `rules`: the thresholds are estimated from
observed outcomes rather than asserted, so they move when the world moves.

What it is *not* allowed to do, and does not: read `ep.true_cause`, `self_recover_at`,
`funds_available_at`, `credential_fix_at`, `instrument_blocked`, `merchant_broken_until`
or the outage schedule. It is handed the `RecoveryEnv` only so it can call
`env.observe()` on the batch — the same latent-stripping projection every other policy
sees, one payment at a time. `test_policies.py` asserts the oracle still bounds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from app.models.schemas import PaymentMethod
from app.policies import calibration
from app.policies.base import BasePolicy
from app.sim.environment import (
    CARD_BLOCK_PROBABILITY_PER_FAILED_RETRY,
    CHANNEL_COST_PAISE,
    ESCALATION_COST_PAISE,
    MANDATE_PAUSE_PROBABILITY_PER_FAILED_RETRY,
    RecoveryEnv,
)
from app.sim.types import Action, ActionType, Channel, Observation, Tone

QUIET_START_HOUR = 22
QUIET_END_HOUR = 8
"""22:00–08:00 IST. Enforced as a hard rule, not priced. The fitted model says
overnight messages are read at 0.22× the daytime rate, which on a large enough
payment would still pencil out — and that is exactly the trade a payments team
will not let a policy make. A compliance limit that a good enough expected value
can buy its way past is not a limit."""

HORIZON_HOURS = 48.0
"""How much recovery window to assume. The scenarios run 48–96 hours and the
`Observation` does not carry the deadline, so this takes the shortest of them:
being wrong about time in the pessimistic direction costs a little patience,
being wrong optimistically means planning around hours that do not exist."""

STORM_COLLATERAL_MULTIPLE = 5.0
"""What a blocked card or paused mandate costs, as a multiple of the payment.

Blocking an instrument loses this payment *and* removes the rail every later action
on it would have used *and* leaves the customer unable to pay the merchant tomorrow —
a harm that outlives the recovery window. `naive_retry` prices it at zero and blocks
instruments in bulk.

The number is picked from a sweep rather than from a theory of customer lifetime
value, because the sweep says something more useful than the theory would. Over 15
batches, raising the price from 1.6× to 9× moved lift by 0.7% (₹15.40 L → ₹15.30 L
per batch) and cut self-inflicted blocks by a third (22 → 14). Lift is flat and harm
is monotone across the whole range, which means the retries this price forgoes were
barely worth taking in the first place — so the harsh end is close to free and there
is no case for the lenient one. 5.0 sits where the harm curve has mostly bent and
the lift cost is still under a rounding error.

What the sweep also shows is that blocks are not reducible to zero by pricing. Most
of them come from retries with a genuinely high success rate that happened to fail;
at p=0.78 the risk term is 1.3% of the payment and no multiple within reason
outweighs it. Taking good retries costs some blocked cards. The report prints the
count next to the money for exactly that reason."""

MIN_ACTION_VALUE_PAISE = 500
"""Below ₹5 of expected gain, do nothing. A queue of actions each worth two rupees
is how a policy accumulates thousands of messages and a compliance problem, and the
estimate is not precise enough at that scale to be acting on anyway."""

CONTACT_FATIGUE_DECAY = 0.55
"""Each message after the first is worth less than the model's marginal rate implies.
The fitted `link_open` cells are conditioned on contacts already made, so most of
this is already in the data; this is the residual — a third message is an imposition
whether or not it is opened."""

WAIT_GAIN_THRESHOLD_PAISE = 2_000
"""How much better the picture has to get for waiting to be worth it — ₹20.

Waiting is not free: it burns window, and on a payment the customer might have paid
unaided it delays nothing but our own knowledge. It has to buy something."""

MAX_RETRIES_PER_PAYMENT = 3
MAX_CONTACTS_PER_DAY = 2
MIN_RETRY_INTERVAL_MINUTES = 15
UPI_TRANSACTION_CEILING_PAISE = 10_000_000
"""Operational limits, deliberately identical to `max_retries_per_payment`,
`max_contacts_per_day`, `min_retry_interval_minutes` and
`upi_transaction_ceiling_paise` in `app/config.py`, which is what the deployed
`ComplianceEngine` enforces on live traffic. `test_policies.py` asserts they still
match.

They are here because expected value alone does not stop. A retry on netbanking or a
wallet costs nothing in rupees and carries no instrument risk, so the arithmetic will
happily take a fortieth one at a 0.001% success rate on a large enough payment —
which is how 266 episodes ran to the step cap before this existed. That is not a
tuning problem, it is the correct answer to the wrong question: no issuer, risk team
or customer will accept forty authorisation attempts on one order however cheap each
one is. Measuring a policy that ignores the limits its own production compliance
engine enforces would report money that could never actually be collected.

The last two were added after measuring exactly that. With only the retry and contact
caps in the policy, 18% of its retries landed inside the gateway's minimum interval —
every one of them money the report claimed and production would have refused at the
door. The UPI ceiling is NPCI's, not ours, and it binds a *rail* rather than a
payment: above ₹1 lakh a collect request is declined by the network, so the policy
routes those to a rail that can carry the amount instead of dropping the payment."""


@dataclass(frozen=True)
class Candidate:
    """One action, priced. `value` is expected paise; `why` becomes the audit line."""

    action: Action
    value: float
    why: str


class PayRevivePolicy(BasePolicy):
    """Expected-value recovery over a fitted model, with a batch-ranked agent bench."""

    name = "payrevive"
    description = (
        "Prices every route from a log-fitted recoverability model and takes the best "
        "one, reserves the agent bench for the batch's most valuable calls, and stops "
        "when nothing left is worth doing."
    )

    def __init__(self, env: RecoveryEnv, model: calibration.Model | None = None) -> None:
        self.env = env
        self.model = model or calibration.load()
        self._agent_slots: set[str] = set()

    # ── Batch-level planning ────────────────────────────────────────────────

    def begin(self, episode_count: int) -> None:
        """Decide who the agent bench is for, before working any payment.

        This is the edge `rules` cannot have. The bench is a fixed number of
        identically priced slots, so ranking candidates by expected value and taking
        the top of the list is exactly optimal — but only if you can see the list.
        A rule evaluated one payment at a time has to commit to a slot before it
        knows whether something better is coming, and a fixed rupee floor is the
        best it can do.

        Every figure below comes from `env.observe()`, the same projection handed to
        every other policy. Nothing latent is read.
        """
        ranked: list[tuple[float, str]] = []
        for episode in self.env.episodes:
            obs = self.env.observe(episode)
            gain = self._agent_gain(obs)
            if gain > 0:
                ranked.append((gain, obs.payment_id))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        self._agent_slots = {pid for _, pid in ranked[: self.env.agent_capacity]}

    def _agent_gain(self, obs: Observation) -> float:
        """Expected paise from escalating this payment, net of everything cheaper.

        An agent call is only worth a slot for what it adds *over* the routes we would
        have taken anyway — every one of them, not just the free retry. Ranking on the
        raw agent rate hands the bench to easy timeouts a server-side retry recovers
        for nothing; ranking on the retry alone hands it to large payments with no
        mandate, where a WhatsApp link would have done the job and the call forecloses
        it. What a slot is worth is the *margin*, and the margin is over the best
        alternative that exists.
        """
        agent = self._rate("agent_paid", obs)
        alternatives = self._retry_candidates(obs)
        if self._may_contact(obs):
            alternatives += self._link_candidates(obs)
        best_free = max((c.value for c in alternatives), default=0.0)
        return agent * obs.amount - ESCALATION_COST_PAISE - max(0.0, best_free)

    # ── Per-payment decision ────────────────────────────────────────────────

    def act(self, obs: Observation) -> Action:
        """Take the most valuable legal action, or wait for a better one, or stop."""
        candidates = self._priced(obs)
        best = max(candidates, key=lambda c: c.value, default=None)

        hold = self._quiet_hold(obs)
        if hold is not None:
            return self._with_reason(hold)

        # Waiting is checked *before* acting, not as a fallback when nothing is worth
        # doing. That ordering is the difference between a policy that compares now
        # against later and one that takes the first thing above a floor: the earlier
        # version acted whenever any route cleared ₹5, which during an outage meant
        # retrying into a bank it could see was dark instead of sitting out the twenty
        # minutes until it cleared. `_wait_candidate` only returns something when
        # waiting beats acting now by a real margin, so putting it first is exactly
        # "take the better of the two" rather than a preference for patience.
        floor = max(best.value, 0.0) if best is not None else 0.0
        wait = self._wait_candidate(obs, floor)
        if wait is not None:
            return self._with_reason(wait)

        if best is not None and best.value > MIN_ACTION_VALUE_PAISE:
            return self._with_reason(best)

        return Action(
            ActionType.GIVE_UP,
            reason=(
                f"stopping: no action on {obs.error_reason} is worth more than "
                f"₹{MIN_ACTION_VALUE_PAISE / 100:.0f} in expectation after "
                f"{obs.attempts_made} attempts and {obs.contacts_made} messages, and "
                f"waiting does not improve it — further contact costs the customer "
                f"more than it is worth to us"
            ),
        )

    def _priced(self, obs: Observation, ahead_minutes: float = 0.0) -> list[Candidate]:
        """Every action available, priced as if it were `ahead_minutes` from now.

        One parameter rather than three, because the things that move with the clock
        move together: the elapsed-time bucket, the bank's forecast health, the hour
        quiet hours are judged on, and how many days of contact budget have accrued.
        Pricing a future action with today's contact ban was a real bug — a payment
        that failed at 23:00 with no mandate had no candidates at all, so the overnight
        hold could not justify itself and the policy gave up on it instead of waiting
        until 08:00. `rules` simply waited, and collected.

        Escalation is added only once the payment has actually spent a cheaper route.
        It is the one terminal action in the space — `run_episode` closes the episode on
        it — so its true cost is not the ₹35 call, it is every action the payment will
        never get. Pricing it as a peer of a retry had the policy phoning a ₹1.9 L
        failure at minute zero, missing at the 60% connect rate, and closing a payment
        `rules` recovered with a WhatsApp link. Ranking still decides *who* gets a slot;
        this decides *when*, and the answer is after the cheap routes have had their turn.

        `attempts_made or contacts_made` rather than "not the first step", because a
        payment that has only ever waited has spent nothing and still has its whole tail
        ahead of it. The exception is the case where that tail is empty anyway: when
        nothing cheaper clears the floor, foreclosing routes worth under ₹5 costs
        nothing, and for a ₹1.8 L failure on a dead instrument a human is the only thing
        left that might work.

        The gate used to be that floor alone, applied always, and it was far too tight.
        `_agent_gain` already subtracts the best cheaper route, so demanding the
        alternatives be worth under ₹5 *before* pricing the call made that subtraction a
        no-op and left the bench idle: 26 of 28 slots unused on `festival_spike`, 0 of 16
        on `outage_day`, against an oracle that spent 12 and 8. Two mechanisms were
        costing the same thing, and the cruder one won.
        """
        candidates = self._retry_candidates(obs, ahead_minutes)
        if self._may_contact(obs, ahead_minutes):
            candidates += self._link_candidates(obs, ahead_minutes)
        best = max((c.value for c in candidates), default=0.0)
        spent_something = obs.attempts_made > 0 or obs.contacts_made > 0
        if (
            (spent_something or best <= MIN_ACTION_VALUE_PAISE)
            and obs.payment_id in self._agent_slots
            and not obs.escalated
            and obs.agent_calls_remaining > 0
            # An agent call is customer contact, and the deployed engine charges it
            # against the same daily budget as a message. Without this the policy
            # phoned 235 customers it had already messaged twice that day — approved by
            # arithmetic, refused by production, and the most intrusive of the three
            # touches. Quiet hours come with it: a call at 02:00 is worse than the SMS
            # we refuse to send.
            and self._may_contact(obs, ahead_minutes)
        ):
            gain = self._agent_gain(obs)
            if gain > MIN_ACTION_VALUE_PAISE:
                candidates.append(Candidate(
                    Action(ActionType.ESCALATE),
                    gain,
                    f"agent call: {self._rate('agent_paid', obs):.0%} completion on "
                    f"{obs.error_reason} beats every cheaper route by "
                    f"{gain / 100:,.0f} rupees, and this payment is in the top "
                    f"{self.env.agent_capacity} of the batch by that margin",
                ))
        return candidates

    def _with_reason(self, candidate: Candidate) -> Action:
        """Attach the audit line. Every action this policy takes carries the number
        that justified it, so a decision can be argued with after the fact rather
        than merely observed."""
        return Action(
            type=candidate.action.type,
            wait_minutes=candidate.action.wait_minutes,
            method=candidate.action.method,
            channel=candidate.action.channel,
            tone=candidate.action.tone,
            reason=f"{candidate.why} (expected +₹{candidate.value / 100:,.0f})",
        )

    # ── Pricing each route ──────────────────────────────────────────────────

    def _retry_candidates(
        self, obs: Observation, ahead_minutes: float = 0.0
    ) -> list[Candidate]:
        """Every rail we could re-charge, priced with storm risk.

        A retry costs nothing in rupees — Razorpay does not charge for a failed
        payment — so the only thing standing between this policy and `naive_retry`
        is the collateral term. A card retry that fails carries a 6% chance of the
        issuer blocking the card outright, which is why a 2%-likely retry on a dead
        card is correctly refused here and taken 83,833 times by the incumbent.

        Bank health enters as a distribution rather than the bucket the bank is in, so
        the same code prices *now* — certainty about the present — and *after a wait*,
        where the fitted forecast applies. That is `rules`' fixed 45-minute outage wait
        replaced by a number: the logs say a dark bank is still dark 91% of the time an
        hour later and clear 93% of the time by hour six, so how long to wait has an
        answer rather than a default.

        A retry inside the gateway's minimum interval is not priced at all rather than
        priced badly. It is not a bad bet, it is an action that would be refused before
        it reached an issuer, and a rate fitted on retries that actually happened says
        nothing about one that cannot.
        """
        if not obs.has_mandate or obs.attempts_made >= MAX_RETRIES_PER_PAYMENT:
            return []
        if self._retry_interval_remaining(obs, ahead_minutes) > 0.0:
            return []
        age = self._age(obs, ahead_minutes)
        weights = (
            self._health_now(obs)
            if ahead_minutes <= 0.0
            else self._health_later(obs, ahead_minutes / 60.0)
        )
        factor = sum(
            weight * self.model.health(obs.error_reason, bucket)
            for bucket, weight in weights.items()
        )
        out: list[Candidate] = []
        for method in obs.available_methods:
            if not self._rail_carries(obs, method):
                continue
            same = method is obs.method
            route = "retry_same" if same else f"retry_to_{method.value}"
            fallback = self._rate("retry_switched", obs, age=age) if not same else 0.0
            p = self._rate(route, obs, default=fallback, age=age)
            # Past the last attempt bucket the model resolves, extrapolate down. The
            # fitted cells lump attempt 3 with attempt 10, so a policy that trusted
            # them retried until it ran out of steps — 266 episodes never closed. The
            # decay is measured from the logs, not chosen: consecutive attempt
            # buckets stand in a fitted ratio to each other.
            beyond = max(0, obs.attempts_made - calibration.tries_bucket(obs.attempts_made))
            p *= self.model.attempt_decay ** beyond
            p = min(1.0, p * factor)
            storm = {
                PaymentMethod.CARD: CARD_BLOCK_PROBABILITY_PER_FAILED_RETRY,
                PaymentMethod.UPI: MANDATE_PAUSE_PROBABILITY_PER_FAILED_RETRY,
            }.get(method, 0.0)
            collateral = (1.0 - p) * storm * STORM_COLLATERAL_MULTIPLE * obs.amount
            value = p * obs.amount - collateral
            where = "same rail" if same else f"switch to {method.value}"
            out.append(Candidate(
                Action(ActionType.RETRY, method=method),
                value,
                f"{where}: {p:.0%} of logged retries on {obs.error_reason} at this "
                f"age and attempt count authorised"
                + (
                    f" (discounted {self.model.attempt_decay:.2f}× per attempt past "
                    f"the {beyond + calibration.tries_bucket(obs.attempts_made)}th)"
                    if beyond else ""
                )
                + (
                    f", adjusted {factor:.2f}× for {obs.bank} clearing "
                    f"{obs.bank_signal.observed_success_rate_1h:.0%} of charges this hour"
                    if abs(factor - 1.0) > 0.02 else ""
                )
                + (
                    f", against a {storm:.0%} risk of killing the instrument "
                    f"(priced at {STORM_COLLATERAL_MULTIPLE:.1f}× the payment)"
                    if storm else ", with no instrument to lose"
                ),
            ))
        return out

    def _link_candidates(
        self, obs: Observation, ahead_minutes: float = 0.0
    ) -> list[Candidate]:
        """Messages, priced per channel and rail.

        Two things the fitted model supplies that no rule has: WhatsApp is read
        1.5× as often as the average channel and email 0.56× as often, and a
        friendly register beats an urgent one by 1.55×. Both are marginals over a
        uniformly random probe, so they are unconfounded — they would not be if
        they came from the incumbent's logs, where the channel is picked from the
        amount.

        The rails a link may suggest are filtered by what the rail can carry rather
        than by the amount alone, which is the difference between "this payment is too
        big to ask for" and "UPI is too small to ask on". A ₹1.5 L failure is offered a
        card link, and only a customer holding nothing but UPI is left with the agent
        bench as their one route.
        """
        age = self._age(obs, ahead_minutes)
        opened = self._rate("link_open", obs, age=age)
        if opened <= 0.0:
            return []
        fatigue = CONTACT_FATIGUE_DECAY ** obs.contacts_made
        tone = self._best_tone()
        out: list[Candidate] = []
        for channel in Channel:
            reach = opened * self.model.channel_lift.get(channel.value, 1.0)
            reach *= self.model.tone_lift.get(tone.value, 1.0) * fatigue
            for method in self._link_rails(obs):
                converts = self._pay_given_open(obs, method, age)
                cost = CHANNEL_COST_PAISE[channel]
                out.append(Candidate(
                    Action(ActionType.SEND_LINK, channel=channel, method=method, tone=tone),
                    reach * converts * obs.amount - cost,
                    f"{channel.value} link on {method.value}: {reach:.1%} read rate "
                    f"({self.model.channel_lift.get(channel.value, 1.0):.2f}× channel, "
                    f"{fatigue:.2f}× after {obs.contacts_made} prior messages) and "
                    f"{converts:.0%} pays once read",
                ))
        return out

    def _link_rails(self, obs: Observation) -> list[PaymentMethod]:
        """Which rail to put in the link. Only instruments the customer holds and that
        can carry the amount — anything else is refused by the gateway and teaches the
        customer we are broken."""
        rails = [m for m in obs.available_methods if self._rail_carries(obs, m)]
        if rails:
            return rails
        return [obs.method] if self._rail_carries(obs, obs.method) else []

    def _rail_carries(self, obs: Observation, method: PaymentMethod) -> bool:
        """Whether this rail can move this amount at all.

        NPCI declines a UPI collect above ₹1 lakh, so suggesting one is not a bet with
        poor odds — it is an action the network refuses, and the fitted rate for
        `retry_to_upi` was estimated on transactions that were allowed to happen. The
        remedy is a different rail, not a smaller ambition, which is why this filters
        methods rather than rejecting the payment.
        """
        return not (
            method is PaymentMethod.UPI
            and obs.amount > UPI_TRANSACTION_CEILING_PAISE
        )

    def _pay_given_open(
        self, obs: Observation, method: PaymentMethod, age: int | None = None
    ) -> float:
        """Given the customer opened the link, does the charge go through?

        A link is two independent things: reaching someone, and the money being
        there when they act. The second is the same physical question a retry asks,
        so it reuses the retry rates — with the customer present, which is why the
        rate for a rail the customer chooses themselves is at worst the rate we
        measured server-side.

        Unlike a retry this is *not* adjusted for the bank's current health, and
        deliberately: the charge happens whenever the customer gets round to opening
        the link, which is hours away and not now. Conditioning it on this hour's
        success rate would price a future event on present information.
        """
        same = method is obs.method
        route = "retry_same" if same else f"retry_to_{method.value}"
        fallback = self._rate("retry_switched", obs, age=age) if not same else 0.0
        return self._rate(route, obs, default=fallback, age=age)

    def _best_tone(self) -> Tone:
        """The register with the highest fitted read lift.

        Chosen globally because the `Observation` carries no persona, and the fitted
        tone lift is therefore a marginal over every customer type. `hinglish` scores
        0.78× not because it reads badly but because it fits only some people — the
        moment a real merchant has a language preference on file, this becomes a
        per-customer choice and that number stops being the right one to use.
        """
        return max(Tone, key=lambda t: self.model.tone_lift.get(t.value, 1.0))

    def _quiet_hold(self, obs: Observation) -> Candidate | None:
        """Sleep to 08:00 IST when outreach is the only thing left worth doing.

        Bounded by construction: it lands exactly on the boundary, so it can fire at
        most once per night, and it never fires while a retry is worth taking —
        server-side retries are not customer contact and quiet hours do not apply
        to them.

        The retry is priced at the moment it becomes *legal* rather than right now,
        which matters only when the gateway's interval is still running. A payment
        whose next retry is worth taking in twelve minutes must not be put to sleep
        for nine hours because the retry happens to be blocked at the instant we
        looked — bank outages are disproportionately overnight, so that is the window
        this policy least wants to forfeit.
        """
        if not self._is_quiet(obs):
            return None
        wait_for = self._minutes_until_retryable(obs)
        if any(
            c.value > MIN_ACTION_VALUE_PAISE
            for c in self._retry_candidates(obs, wait_for)
        ):
            return None
        minutes = self._minutes_until_open(obs)
        later = self._best_value(obs, minutes)
        if later <= MIN_ACTION_VALUE_PAISE:
            return None
        return Candidate(
            Action(ActionType.WAIT, wait_minutes=minutes),
            later,
            f"holding {minutes} minutes until 08:00 IST — overnight messages are read "
            f"at {self.model.quiet_lift:.2f}× the daytime rate, and quiet hours are a "
            f"hard limit here regardless of what the arithmetic says",
        )

    def _wait_candidate(self, obs: Observation, best_now: float) -> Candidate | None:
        """Advance the clock when time itself is the fix.

        `rules` waits a fixed 45 minutes for any outage. This considers every knot in
        the fitted model still ahead of this payment — the points at which the
        estimated rates actually change — prices the payment at each one, and takes
        the best, or none if none of them beats acting now by more than
        `WAIT_GAIN_THRESHOLD_PAISE`.

        Considering all of them rather than only the next one is what lets it sit out
        an outage. The fitted forecast says a dark bank is still dark 91% of the time
        an hour later, so a wait to the one-hour knot buys nothing and is correctly
        refused; by hour six it is clear 93% of the time. A policy that could only ever
        wait as far as the next knot would look at the first of those, conclude waiting
        does not pay, and retry into a bank it already knows is dark — which is exactly
        what it did before this loop, and why `outage_day` was the one scenario a
        hand-written rule still won.

        That is also what makes the policy terminate. Every target is strictly later
        than the payment's current age, so each wait moves it past a knot it can never
        revisit, and there are three of them.

        The retry interval is a fourth target, and the only one that is not a knot.
        Without it the shortest wait available is hours long, so a payment whose best
        route is a second retry twelve minutes from now had to either take a worse
        action or stop — and dropping those retries hands the money back rather than
        collecting it a quarter of an hour late. It cannot loop either: it exists only
        while a retry is interval-blocked, waiting past the interval clears it, and
        `MAX_RETRIES_PER_PAYMENT` bounds how many times a retry can arm it again.
        """
        hours = obs.minutes_since_failure / 60.0
        targets: list[tuple[int, str]] = []

        wait_for = self._minutes_until_retryable(obs)
        if wait_for:
            targets.append((
                wait_for,
                f"until the gateway's {MIN_RETRY_INTERVAL_MINUTES}-minute retry "
                "interval clears",
            ))
        for knot in calibration.ELAPSED_KNOTS_HOURS:
            if knot <= hours or knot > HORIZON_HOURS:
                continue
            targets.append((int((knot - hours) * 60.0) + 1, f"to the {knot:.0f}h mark"))

        best: Candidate | None = None
        for minutes, where in targets:
            later = self._best_value(obs, minutes)
            gain = later - best_now
            if gain <= WAIT_GAIN_THRESHOLD_PAISE or (best and gain <= best.value):
                continue
            forecast = self._health_later(obs, minutes / 60.0)
            health = obs.bank_signal.observed_success_rate_1h
            recovered = forecast.get(len(calibration.HEALTH_KNOTS), 0.0)
            best = Candidate(
                Action(ActionType.WAIT, wait_minutes=minutes),
                gain,
                f"waiting {minutes} minutes {where}: the best route is "
                f"worth ₹{later / 100:,.0f} there against ₹{best_now / 100:,.0f} now "
                f"({obs.bank} is clearing {health:.0%} of charges this hour"
                + (", with a concurrent failure spike"
                   if obs.bank_signal.concurrent_failure_spike else "")
                + (f", and banks this dark are clearing normally again "
                   f"{recovered:.0%} of the time after a gap this long"
                   if health < calibration.HEALTH_KNOTS[-1] else "")
                + ")",
            )
        return best

    def _best_value(self, obs: Observation, ahead_minutes: float) -> float:
        """The best action value this payment would have after waiting that long.

        Everything but the clock and the bank is held fixed, which understates the
        wait — a customer's salary may land in the meantime and this cannot see that.
        Understating is the right direction: it makes the policy impatient rather than
        hopeful.
        """
        return max((c.value for c in self._priced(obs, ahead_minutes)), default=0.0)

    def _age(self, obs: Observation, ahead_minutes: float = 0.0) -> int:
        """The elapsed-time bucket this payment will be in `ahead_minutes` from now."""
        return calibration.elapsed_bucket(
            (obs.minutes_since_failure + ahead_minutes) / 60.0
        )

    def _health_now(self, obs: Observation) -> dict[int, float]:
        """The bank's health bucket right now, as a point mass. Observable: it is the
        success rate over the merchant's own last hour of traffic on this bank."""
        return {calibration.health_bucket(obs.bank_signal.observed_success_rate_1h): 1.0}

    def _health_later(self, obs: Observation, gap_hours: float) -> dict[int, float]:
        """Where the bank's health will be after a wait of this length."""
        bucket = calibration.health_bucket(obs.bank_signal.observed_success_rate_1h)
        return self.model.forecast(bucket, gap_hours)

    # ── Compliance, which is not priced ─────────────────────────────────────

    def _retry_interval_remaining(
        self, obs: Observation, ahead_minutes: float = 0.0
    ) -> float:
        """Minutes until a server-side retry is legal again, 0.0 when it already is.

        `None` means no retry has happened on this payment, which is not the same as
        one having happened a long time ago — reading it as "unknown, assume blocked"
        would stop the first retry on every payment, and reading it as 0 would stop
        none of them.
        """
        if obs.minutes_since_last_retry is None:
            return 0.0
        return max(
            0.0,
            MIN_RETRY_INTERVAL_MINUTES - (obs.minutes_since_last_retry + ahead_minutes),
        )

    def _minutes_until_retryable(self, obs: Observation) -> int:
        """The same figure as a whole-minute wait, 0 when a retry is legal now.

        Rounded up rather than to nearest, so pricing the retry at this offset is
        pricing it at a moment it is definitely allowed. Asking for exactly the
        remaining fraction lands on the boundary, where the float subtraction can
        leave a residual and keep the retry blocked at the very instant the wait was
        chosen to unblock it.
        """
        remaining = self._retry_interval_remaining(obs)
        return int(remaining) + 1 if remaining > 0.0 else 0

    def _may_contact(self, obs: Observation, ahead_minutes: float = 0.0) -> bool:
        """Whether a message is allowed. Not a term in any expectation.

        Two limits, both hard: quiet hours, and the daily contact cap the deployed
        compliance engine enforces — counted on the IST calendar day the message would
        go out on, which is the day production's own counter is keyed to.

        It used to be counted as two per day *since failure*, on a running total,
        which is a cumulative budget rather than a daily one: a payment left alone on
        its first day arrived at its second with four messages available and spent
        them. Sixteen messages across the grid breached the deployed limit that way,
        every one of them approved by arithmetic production would have refused.

        `ahead_minutes` asks the same question of a future moment, which the wait
        look-ahead needs — and answering it is now exact rather than approximate. A
        wait that crosses midnight arrives with the day's budget untouched, so a
        payment contacted twice this evening is contactable again tomorrow morning and
        the valuation can say so. A payment that failed at 23:00 is not contactable now
        and is contactable at 08:00; a valuation that could not express that gave up on
        it overnight rather than holding.
        """
        if self._is_quiet(obs, ahead_minutes):
            return False
        then = obs.now + timedelta(minutes=ahead_minutes)
        spent = obs.contacts_today if then.date() == obs.now.date() else 0
        return spent < MAX_CONTACTS_PER_DAY

    def _is_quiet(self, obs: Observation, ahead_minutes: float = 0.0) -> bool:
        """Judged on decision time — `obs.now` — because that is when the message
        goes out. Judging it on when the effects settle charges a message sent
        legally at 21:59 as a violation."""
        hour = (obs.now + timedelta(minutes=ahead_minutes)).hour
        return hour >= QUIET_START_HOUR or hour < QUIET_END_HOUR

    def _minutes_until_open(self, obs: Observation) -> int:
        """Minutes to 08:00 IST, so the wait lands exactly on the boundary rather
        than sleeping a fixed block and waking up still inside quiet hours."""
        now = obs.now
        hours = (QUIET_END_HOUR - now.hour) % 24
        return max(15, hours * 60 - now.minute)

    # ── Model access ────────────────────────────────────────────────────────

    def _rate(
        self, route: str, obs: Observation, default: float = 0.0, age: int | None = None
    ) -> float:
        """The fitted rate for this route on this payment, at this age and count.

        The key is built only from things in the `Observation`: the webhook's error
        reason, hours since failure, and how many attempts or messages this payment
        has already had. `Model.rate` walks from the most specific cell to the
        coarsest until one has at least 40 trials, so a rate is never read off nine
        observations while the signal in those nine still reaches the decision.
        """
        if age is None:
            age = calibration.elapsed_bucket(obs.minutes_since_failure / 60.0)
        tries = obs.contacts_made if route == "link_open" else obs.attempts_made
        return self.model.rate(
            route,
            obs.error_reason,
            age,
            calibration.tries_bucket(tries),
            default=default,
        )
