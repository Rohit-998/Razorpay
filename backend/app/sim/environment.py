"""The recovery environment: a causal simulator that policies are scored against.

Contract, in one line: an outcome is never a function of a label.

A retry succeeds if the bank is actually up at that instant, the customer's money
has actually arrived, and the instrument is not actually dead. A link converts if
this particular person, at their current level of irritation, on that channel, at
that hour, actually opens it. Nothing here consults the root cause to decide what
happens — the root cause is a hidden variable that *causes* those conditions, and
policies have to infer it from noisy evidence.

Two design details do most of the analytical work:

  Common random numbers. Each episode carries several independent RNG streams,
  one per event type. A retry always draws from the retry stream no matter what
  the policy did beforehand, so two policies compared on the same seed face the
  same coin flips. That removes most of the variance from policy comparison and
  is why differences of a percentage point or two are trustworthy.

  Action-independent counterfactuals. `self_recover_at`, the downtime schedule
  and the funds-arrival time are all drawn at reset, before any policy runs. The
  `do_nothing` baseline therefore recovers precisely the money that was coming
  back anyway, and every other policy is measured as a delta against it.

Retry storms are modelled rather than assumed away: repeated failed attempts on a
card can get it blocked by the issuer, permanently. A policy that hammers three
retries at a dead instrument does not merely waste attempts, it destroys value —
which is exactly the failure mode a naive retry loop exhibits in production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from app.models.schemas import PaymentMethod
from app.sim import emission as em
from app.sim.customer import (
    PERSONAS,
    CustomerLatent,
    Persona,
    sample_customer,
)
from app.sim.types import (
    Action,
    ActionType,
    AttributionTruth,
    BankSignal,
    Channel,
    CustomerSignal,
    Observation,
    StepResult,
    Terminal,
)
from app.sim.world import BANKS, World
from app.sim.scenarios import Scenario

# Per-message cost in paise. Indian utility-template rates, rounded.
CHANNEL_COST_PAISE: dict[Channel, int] = {
    Channel.EMAIL: 2,
    Channel.SMS: 22,
    Channel.WHATSAPP: 38,
}

ESCALATION_COST_PAISE = 9_000
"""Fully loaded cost of a human agent working one payment (~₹90).

Not just the talk time. An outbound recovery contact in India takes two to three
dials to get one connect, six to eight minutes of agent time on the call, plus
wrap-up, dialler and supervision overhead. At a loaded agent cost of roughly
₹300/hour that lands near ₹90 per payment attempted — and it is charged whether
or not anyone picks up, because the cost is the attempt, not the outcome."""

AGENT_CONNECT_FLOOR = 0.22
AGENT_CONNECT_INTENT_WEIGHT = 0.40
AGENT_CONNECT_CEILING = 0.70
"""Chance the agent actually reaches the customer: `floor + weight × intent`,
capped. Outbound connect rates on Indian mobile numbers sit in the 25–45% band
even for warm lists, and someone who has already lost interest is the least
likely to answer an unknown number."""

AGENT_COMPLETION_FLOOR = 0.40
AGENT_COMPLETION_INTENT_WEIGHT = 0.50
AGENT_COMPLETION_CEILING = 0.90
"""Chance a customer who *is* reached, on a rail that *can* authorise, actually
pays on the call. The rest say 'not now' and mean it. Without this gate a
connect converts almost perfectly and the agent becomes a magic wand."""

AMBIGUITY_WINDOW_HOURS = 6.0
"""If a customer pays through their own channel within this long after we
contacted them, causation is unprovable and we decline to claim the win."""

QUIET_HOUR_CLICK_FACTOR = 0.28
"""Messages that land overnight mostly go unread. Not a compliance rule — this is
the effectiveness penalty that makes quiet-hours guardrails nearly free."""

MAX_CLICK_PROBABILITY = 0.95
"""Ceiling on click-through however perfect the message. Nobody opens everything,
and without a cap a high-intent persona on their best channel would be a certainty."""

CARD_BLOCK_PROBABILITY_PER_FAILED_RETRY = 0.06
"""Chance a failed card retry pushes the issuer into blocking the card outright."""

MANDATE_PAUSE_PROBABILITY_PER_FAILED_RETRY = 0.035
"""Same mechanic on UPI autopay: repeated failed debits get a mandate paused.
Lower than the card figure, but it means no rail is safe to hammer."""


@dataclass
class Episode:
    """One failed payment under recovery. Latent fields are policy-invisible."""

    index: int
    payment_id: str
    order_id: str
    amount: int
    method: PaymentMethod
    bank: str
    vpa: str | None
    wallet: str | None
    emission: em.Emission
    failed_at: datetime
    window_hours: int

    true_cause: str
    customer: CustomerLatent
    merchant_broken_until: datetime | None = None
    """For MERCHANT_ERROR: until this moment, the failed method is broken on our
    side. Retrying it cannot work; switching method or waiting can. `None` for
    every other cause."""

    now: datetime = field(init=False)
    attempts: int = 0
    contacts: int = 0
    escalated: bool = False
    terminal: Terminal = Terminal.OPEN
    paid: bool = False
    paid_at: datetime | None = None
    attribution: AttributionTruth = AttributionTruth.NOT_RECOVERED
    cost_paise: int = 0
    invalid_actions: int = 0
    last_contact_at: datetime | None = None
    contacts_by_day: dict[str, int] = field(default_factory=dict)
    """Messages sent, keyed by the IST calendar day they went out on.

    `contacts` alone cannot answer the question the daily cap asks. A policy holding a
    running total can only enforce a cumulative budget, which lets an unused day roll
    forward — and a payment that was quiet on Monday then sent four messages on
    Tuesday breached a limit of two while its total still looked legal. Production
    keys the same counter on the IST day in Redis, so this is the counter it has, not
    a convenience the simulator invented."""
    last_retry_at: datetime | None = None
    """When we last charged this payment server-side.

    Present because `min_retry_interval_minutes` is a real gateway-facing limit and a
    policy that cannot see this cannot comply with it. Measured against the simulator,
    18% of the proposal's retries came inside the interval before it was observable —
    money the report would have been claiming and production would have refused."""
    pending_click_at: datetime | None = None
    pending_click_method: PaymentMethod | None = None
    instrument_blocked: bool = False
    """The failed instrument cannot authorise, now or later. True from the start
    for a genuinely dead card or expired mandate; also set when a retry storm gets
    a live card blocked by its issuer. Irreversible either way — the only escape
    is a different payment method."""
    instrument_dead_at_failure: bool = False
    """Diagnostic split of the flag above: was the instrument already dead when we
    picked the payment up, or did our own retries kill it? Reported separately so
    a policy cannot hide self-inflicted damage inside the unrecoverable pile."""
    history: list[StepResult] = field(default_factory=list)

    rng_retry: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]
    rng_click: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]
    rng_pay: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]
    rng_agent: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.now = self.failed_at

    @property
    def deadline(self) -> datetime:
        return self.failed_at + timedelta(hours=self.window_hours)

    @property
    def amount_rupees(self) -> float:
        return self.amount / 100.0


# Which kinds of customer hit which failure. These correlations are what make
# customer history a genuinely useful feature rather than noise: a first-time
# UPI user fat-fingering their PIN and a premium card holder abandoning a 3DS
# challenge produce nearly identical error fields, and are told apart mainly by
# who they are.
PERSONA_BIAS: dict[str, dict[str, float]] = {
    "BANK_DOWNTIME": {"business_buyer": 1.3},
    "NETWORK_TRANSIENT": {"occasional_user": 1.4},
    "AUTH_TIMEOUT": {"premium_shopper": 2.2, "business_buyer": 0.7},
    "INSUFFICIENT_FUNDS": {"salary_regular": 2.4, "occasional_user": 1.3, "business_buyer": 0.4},
    "WRONG_CREDENTIALS": {"occasional_user": 2.5, "premium_shopper": 0.6},
    "PERMANENT_DECLINE": {"subscription_holder": 1.8, "premium_shopper": 1.6},
    "MERCHANT_ERROR": {"business_buyer": 2.2},
}

METHOD_BIAS: dict[str, dict[PaymentMethod, float]] = {
    "BANK_DOWNTIME": {PaymentMethod.NETBANKING: 0.34, PaymentMethod.UPI: 0.30,
                      PaymentMethod.CARD: 0.26, PaymentMethod.WALLET: 0.10},
    "NETWORK_TRANSIENT": {PaymentMethod.UPI: 0.48, PaymentMethod.CARD: 0.24,
                          PaymentMethod.NETBANKING: 0.18, PaymentMethod.WALLET: 0.10},
    "AUTH_TIMEOUT": {PaymentMethod.CARD: 0.52, PaymentMethod.NETBANKING: 0.26,
                     PaymentMethod.UPI: 0.16, PaymentMethod.WALLET: 0.06},
    "INSUFFICIENT_FUNDS": {PaymentMethod.UPI: 0.40, PaymentMethod.CARD: 0.34,
                           PaymentMethod.NETBANKING: 0.18, PaymentMethod.WALLET: 0.08},
    "WRONG_CREDENTIALS": {PaymentMethod.CARD: 0.44, PaymentMethod.NETBANKING: 0.32,
                          PaymentMethod.UPI: 0.18, PaymentMethod.WALLET: 0.06},
    "PERMANENT_DECLINE": {PaymentMethod.CARD: 0.62, PaymentMethod.UPI: 0.14,
                          PaymentMethod.NETBANKING: 0.14, PaymentMethod.WALLET: 0.10},
    "MERCHANT_ERROR": {PaymentMethod.NETBANKING: 0.42, PaymentMethod.CARD: 0.26,
                       PaymentMethod.UPI: 0.20, PaymentMethod.WALLET: 0.12},
}

AMOUNT_RANGES_PAISE: dict[str, tuple[int, int]] = {
    "micro": (1_000, 9_900),
    "small": (10_000, 99_900),
    "medium": (100_000, 999_900),
    "large": (1_000_000, 4_999_900),
    "premium": (5_000_000, 20_000_000),
}

# RNG stream identifiers. Separate streams per event type give common random
# numbers across policies: the k-th retry in an episode faces the same draw
# whatever the policy did before it.
_STREAM_WORLD, _STREAM_GEN = 0, 1
_STREAM_RETRY, _STREAM_CLICK, _STREAM_PAY, _STREAM_AGENT = 2, 3, 4, 5


class RecoveryEnv:
    """Builds a reproducible batch of failed payments and resolves actions on it.

    Two policies constructed with the same `(scenario, seed)` face byte-identical
    worlds: the same customers, the same outages, the same coin flips.
    """

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self.seed = seed
        self.world = World(
            rng=np.random.default_rng([seed, _STREAM_WORLD]),
            start=scenario.start,
            duration_days=scenario.duration_days,
            downtime_multiplier=scenario.downtime_multiplier,
        )
        self.episodes: list[Episode] = []
        self.agent_capacity: int = scenario.agent_capacity
        self.agent_calls_used: int = 0
        """Batch-level, not episode-level, and that is the point: the agent bench is
        shared, so an escalation spent on payment 12 is one that payment 300 cannot
        have. It makes escalation an allocation problem instead of a free upgrade."""

    # ── Batch construction ───────────────────────────────────────────────

    def reset(self) -> list[Episode]:
        """Generate the batch. Deterministic in `(scenario, seed)`."""
        rng = np.random.default_rng([self.seed, _STREAM_GEN])
        weights = self.scenario.normalised_weights()
        causes = list(weights)
        probs = np.array([weights[c] for c in causes], dtype=float)
        merchant_windows = self._sample_merchant_error_windows(rng)

        episodes: list[Episode] = []
        for index in range(self.scenario.n_payments):
            cause = causes[int(rng.choice(len(causes), p=probs / probs.sum()))]
            failed_at, bank, broken_until = self._sample_failure_time_and_bank(
                rng, cause, merchant_windows
            )
            persona = self._sample_persona_for_cause(rng, cause)
            method = self._sample_method(rng, cause, persona)
            amount = self._sample_amount(rng, persona)

            customer = sample_customer(
                rng=rng, index=index, persona=persona, cause=cause,
                failed_at=failed_at, method=method,
                window_hours=self.scenario.window_hours,
            )
            self._apply_history_signal(rng, customer, cause)

            episodes.append(Episode(
                index=index,
                payment_id=f"pay_{self.seed:02d}{index:05d}{int(rng.integers(0, 1_000)):03d}",
                order_id=f"order_{self.seed:02d}{index:05d}",
                amount=amount,
                method=method,
                bank=bank,
                vpa=self._vpa(customer, bank) if method is PaymentMethod.UPI else None,
                wallet="paytm" if method is PaymentMethod.WALLET else None,
                emission=em.emit(rng, cause),
                failed_at=failed_at,
                window_hours=self.scenario.window_hours,
                true_cause=cause,
                customer=customer,
                merchant_broken_until=broken_until,
                # "Permanent" has to mean permanent. A dead card or an expired
                # mandate cannot be retried into working, and an environment that
                # lets it is an environment that rewards hammering.
                instrument_blocked=cause == "PERMANENT_DECLINE",
                instrument_dead_at_failure=cause == "PERMANENT_DECLINE",
                rng_retry=np.random.default_rng([self.seed, index, _STREAM_RETRY]),
                rng_click=np.random.default_rng([self.seed, index, _STREAM_CLICK]),
                rng_pay=np.random.default_rng([self.seed, index, _STREAM_PAY]),
                rng_agent=np.random.default_rng([self.seed, index, _STREAM_AGENT]),
            ))

        episodes.sort(key=lambda e: e.failed_at)
        self.world.register_failures([(e.bank, e.failed_at) for e in episodes])
        self.episodes = episodes
        self.agent_calls_used = 0
        return episodes

    def _sample_merchant_error_windows(
        self, rng: np.random.Generator
    ) -> list[tuple[datetime, datetime]]:
        """Config breakages are bursty — a wrong setting breaks every payment on a
        method until someone notices. Clustering them makes MERCHANT_ERROR
        detectable from correlated failures rather than from its error text."""
        windows: list[tuple[datetime, datetime]] = []
        for _ in range(self.scenario.merchant_error_windows):
            offset = float(rng.uniform(0, self.scenario.duration_days * 1440))
            start = self.scenario.start + timedelta(minutes=offset)
            windows.append((start, start + timedelta(minutes=float(rng.uniform(90, 600)))))
        return windows

    def _sample_failure_time_and_bank(
        self,
        rng: np.random.Generator,
        cause: str,
        merchant_windows: list[tuple[datetime, datetime]],
    ) -> tuple[datetime, str, datetime | None]:
        """Pick when and where a failure happened, conditioned on its cause.

        This is the step that puts real signal into the auxiliary features:
        downtime failures land inside actual outage windows on the affected bank,
        which is the only thing distinguishing them from isolated network blips.

        The third return value is the moment a merchant-side config breakage gets
        fixed — latent, and the reason MERCHANT_ERROR needs a method switch rather
        than a retry.
        """
        if cause == "BANK_DOWNTIME" and self.world.episodes:
            outage = self.world.episodes[int(rng.integers(0, len(self.world.episodes)))]
            span = max(1.0, (outage.end - outage.start).total_seconds() / 60.0)
            at = outage.start + timedelta(minutes=float(rng.uniform(0, span)))
            return at, outage.bank, None

        if cause == "MERCHANT_ERROR" and merchant_windows:
            start, end = merchant_windows[int(rng.integers(0, len(merchant_windows)))]
            span = max(1.0, (end - start).total_seconds() / 60.0)
            at = start + timedelta(minutes=float(rng.uniform(0, span)))
            # Someone notices and ships a fix a while after the window closes.
            fixed = end + timedelta(minutes=float(rng.uniform(30, 900)))
            return at, self._sample_bank(rng), fixed

        if cause == "INSUFFICIENT_FUNDS":
            # Concentrated in the days before money arrives. Gives the temporal
            # features something to find.
            for _ in range(6):
                candidate = self._uniform_time(rng)
                if candidate.day >= 24 or candidate.day <= 3:
                    return candidate, self._sample_bank(rng), None

        return self._uniform_time(rng), self._sample_bank(rng), None

    def _uniform_time(self, rng: np.random.Generator) -> datetime:
        """A time in the batch window, shaped like real payment traffic:
        thin overnight, busy in the evening."""
        day = int(rng.integers(0, self.scenario.duration_days))
        hour_weights = np.array([
            0.4, 0.25, 0.15, 0.12, 0.12, 0.2, 0.5, 0.9,   # 00-07
            1.3, 1.6, 1.7, 1.7, 1.6, 1.5, 1.5, 1.6,       # 08-15
            1.7, 1.8, 2.0, 2.2, 2.1, 1.7, 1.1, 0.7,       # 16-23
        ])
        hour = int(rng.choice(24, p=hour_weights / hour_weights.sum()))
        return self.scenario.start + timedelta(
            days=day, hours=hour, minutes=float(rng.uniform(0, 60))
        )

    def _sample_bank(self, rng: np.random.Generator) -> str:
        shares = np.array([b.share for b in BANKS], dtype=float)
        return BANKS[int(rng.choice(len(BANKS), p=shares / shares.sum()))].code

    def _sample_persona_for_cause(self, rng: np.random.Generator, cause: str) -> Persona:
        """Draw a persona, tilted by which people hit which failure.

        Without this tilt, customer features would be pure noise and the
        AUTH_TIMEOUT / WRONG_CREDENTIALS pair would be genuinely inseparable.
        """
        bias = PERSONA_BIAS.get(cause, {})
        weights = np.array(
            [p.weight * bias.get(p.name, 1.0) for p in PERSONAS], dtype=float
        )
        return PERSONAS[int(rng.choice(len(PERSONAS), p=weights / weights.sum()))]

    def _sample_method(
        self, rng: np.random.Generator, cause: str, persona: Persona
    ) -> PaymentMethod:
        """Blend the cause's method profile with what this person actually uses.

        Both matter. 3DS timeouts are a card phenomenon, but a card timeout on a
        business buyer is rarer than on a premium shopper, and the mix is what
        keeps `method` from being a giveaway on its own.
        """
        profile = METHOD_BIAS.get(cause, {})
        methods = list(PaymentMethod)
        weights = np.array(
            [
                profile.get(m, 0.05) * (2.4 if m is persona.preferred_method else 1.0)
                for m in methods
            ],
            dtype=float,
        )
        return methods[int(rng.choice(len(methods), p=weights / weights.sum()))]

    def _sample_amount(self, rng: np.random.Generator, persona: Persona) -> int:
        """Amount in paise. Mostly the persona's bucket, sometimes a neighbour."""
        buckets = list(AMOUNT_RANGES_PAISE)
        home = buckets.index(persona.amount_bucket)
        drift = int(rng.choice([-1, 0, 1], p=[0.18, 0.64, 0.18]))
        bucket = buckets[int(np.clip(home + drift, 0, len(buckets) - 1))]
        low, high = AMOUNT_RANGES_PAISE[bucket]
        # Log-uniform: small baskets dominate inside every band, as in real traffic.
        value = float(np.exp(rng.uniform(math.log(low), math.log(high))))
        return int(round(value / 100.0) * 100)

    # Observable customer history, conditioned on the latent cause. This is the
    # only route to separating AUTH_TIMEOUT from WRONG_CREDENTIALS, which emit
    # the same error triple: someone whose payments normally work just abandoned
    # an OTP screen; someone who fails repeatedly has stale details on file.
    #   cause -> (mean history success rate, mean failures in last 7 days)
    HISTORY_SIGNAL: dict[str, tuple[float, float]] = {
        "AUTH_TIMEOUT": (0.91, 0.5),
        "NETWORK_TRANSIENT": (0.90, 0.6),
        "BANK_DOWNTIME": (0.89, 0.8),
        "INSUFFICIENT_FUNDS": (0.79, 1.9),
        "MERCHANT_ERROR": (0.87, 1.0),
        "WRONG_CREDENTIALS": (0.62, 3.4),
        "PERMANENT_DECLINE": (0.58, 3.9),
    }

    def _apply_history_signal(
        self, rng: np.random.Generator, customer: CustomerLatent, cause: str
    ) -> None:
        """Overwrite the customer's observable history to match their failure mode.

        Deliberately noisy: the spread is wide enough that history alone cannot
        pin the cause down, so it has to be combined with the error fields and
        the bank signal rather than used as a shortcut.
        """
        mean_rate, mean_failures = self.HISTORY_SIGNAL.get(cause, (0.86, 1.2))
        customer.history_success_rate = float(
            np.clip(rng.normal(mean_rate, 0.11), 0.20, 0.995)
        )
        customer.history_failures_7d = int(rng.poisson(max(0.05, mean_failures)))

        # Engagement proxy: how often past nudges to this person worked. Correlates
        # with persona channel strength, which is what makes it worth conditioning on.
        best_channel = max(customer.persona.channel_response.values())
        customer.observed_responses = int(
            rng.binomial(customer.observed_contacts, float(np.clip(best_channel, 0.02, 0.95)))
        )

    def _vpa(self, customer: CustomerLatent, bank: str) -> str:
        """A plausible UPI handle. Cosmetic — carries no signal a policy can use."""
        handle = {
            "SBIN": "sbi", "HDFC": "hdfcbank", "ICIC": "icici", "UTIB": "axis",
            "KKBK": "kotak", "BARB": "barodampay", "PUNB": "pnb", "YESB": "yesbank",
            "IOBA": "iob", "CNRB": "cnrb",
        }.get(bank, "upi")
        first = customer.name.split()[0].lower()
        return f"{first}{customer.customer_id[-4:]}@{handle}"

    # ── Observation ──────────────────────────────────────────────────────

    def observe(self, ep: Episode) -> Observation:
        """Project an episode down to what a policy is allowed to know.

        Everything here is something a real merchant has: the webhook fields, the
        merchant's own recent failure counts for that bank, aggregate customer
        history, whether a token or mandate is on file, and how much of the agent
        bench is still free. Nothing latent crosses this boundary — no true cause,
        no outage schedule, no `self_recover_at`.
        """
        return Observation(
            payment_id=ep.payment_id,
            amount=ep.amount,
            method=ep.method,
            bank=ep.bank,
            error_code=ep.emission.error_code,
            error_source=ep.emission.error_source,
            error_step=ep.emission.error_step,
            error_reason=ep.emission.error_reason,
            error_description=ep.emission.error_description,
            now=ep.now,
            failed_at=ep.failed_at,
            attempts_made=ep.attempts,
            contacts_made=ep.contacts,
            contacts_today=ep.contacts_by_day.get(ep.now.strftime("%Y-%m-%d"), 0),
            escalated=ep.escalated,
            bank_signal=BankSignal(
                bank=ep.bank,
                observed_success_rate_1h=self.world.observed_success_rate(ep.bank, ep.now),
                observed_failure_count_1h=self.world.observed_failures(ep.bank, ep.now),
                concurrent_failure_spike=self.world.failure_spike(ep.bank, ep.now),
            ),
            customer_signal=CustomerSignal(
                success_rate_90d=ep.customer.history_success_rate,
                failure_count_7d=ep.customer.history_failures_7d,
                prior_recovery_responses=ep.customer.observed_responses,
                prior_recovery_contacts=ep.customer.observed_contacts,
            ),
            has_mandate=ep.customer.has_mandate,
            available_methods=list(ep.customer.alt_methods),
            agent_calls_remaining=max(0, self.agent_capacity - self.agent_calls_used),
            agent_capacity=self.agent_capacity,
            last_action=ep.history[-1].action if ep.history else None,
            last_detail=ep.history[-1].detail if ep.history else "",
            minutes_since_last_retry=(
                None if ep.last_retry_at is None
                else (ep.now - ep.last_retry_at).total_seconds() / 60.0
            ),
        )

    # ── Physics: whether a charge actually authorises ─────────────────────

    def _charge_succeeds(
        self,
        ep: Episode,
        t: datetime,
        method: PaymentMethod,
        customer_present: bool,
    ) -> tuple[bool, str]:
        """The single gate every payment attempt passes through.

        Note what this function does not do: it never looks at `ep.true_cause`.
        It asks whether the money is there, whether the details would
        authenticate, whether our own configuration is working and whether the
        bank is up at this instant. The root cause is upstream of all four — it
        is *why* one of them is false, not a term in the calculation.
        """
        switched = method is not ep.method

        if ep.instrument_blocked and not switched:
            return False, "instrument blocked by the issuer"

        if ep.merchant_broken_until is not None and t < ep.merchant_broken_until and not switched:
            return False, "our configuration is still broken for this method"

        # Stored credentials only matter server-side. A customer following a link
        # types their details in fresh, which is why outreach beats retrying when
        # the details on file are stale.
        if not customer_present and not ep.customer.credentials_valid(t):
            return False, "stored credentials are still invalid"

        # A drained account is drained whatever rail you use. Moving to a wallet
        # with its own float is the one genuine escape.
        if not (switched and method is PaymentMethod.WALLET) and not ep.customer.funds_ready(t):
            return False, "insufficient balance at this time"

        p = self.world.success_probability(ep.bank, t)
        rng = ep.rng_pay if customer_present else ep.rng_retry
        if float(rng.random()) < p:
            return True, "authorised"
        return False, f"bank declined (success rate {p:.0%} at this time)"

    @staticmethod
    def _is_quiet_hour(t: datetime) -> bool:
        """22:00–08:00 IST. Messages sent here mostly go unread."""
        return t.hour >= 22 or t.hour < 8

    # ── The clock, and the two things that happen without us ──────────────

    def _advance(self, ep: Episode, until: datetime) -> list[str]:
        """Run the clock forward, resolving passive events in chronological order.

        Two events can fire here. The customer paying unprompted was scheduled at
        reset and is completely action-independent — it fires even after a policy
        has given up, which is what stops `GIVE_UP` from being punished for money
        that was arriving anyway. A pending link click was scheduled when we sent
        the message, so its *timing* is ours but its *outcome* still has to clear
        `_charge_succeeds`.
        """
        notes: list[str] = []
        horizon = min(until, ep.deadline)

        while not ep.paid:
            events: list[tuple[datetime, str]] = []
            recover_at = ep.customer.self_recover_at
            if recover_at is not None and ep.now < recover_at <= horizon:
                events.append((recover_at, "self"))
            if ep.pending_click_at is not None and ep.now < ep.pending_click_at <= horizon:
                events.append((ep.pending_click_at, "click"))
            if not events:
                break
            t, kind = min(events, key=lambda pair: pair[0])
            ep.now = t
            notes.append(
                self._resolve_self_recovery(ep, t) if kind == "self"
                else self._resolve_pending_click(ep, t)
            )

        ep.now = max(ep.now, horizon)
        return [note for note in notes if note]

    def _resolve_self_recovery(self, ep: Episode, t: datetime) -> str:
        """The customer pays on their own. We may or may not deserve the credit."""
        ep.paid = True
        ep.paid_at = t
        ep.terminal = Terminal.RECOVERED
        hours_since_contact = (
            math.inf if ep.last_contact_at is None
            else (t - ep.last_contact_at).total_seconds() / 3600.0
        )
        if hours_since_contact <= AMBIGUITY_WINDOW_HOURS:
            ep.attribution = AttributionTruth.AMBIGUOUS
            return (
                f"customer paid through their own channel {hours_since_contact:.1f}h "
                "after we contacted them — causation unprovable, not counted as a win"
            )
        ep.attribution = AttributionTruth.CUSTOMER_SELF_RECOVERED
        return "customer paid unprompted with no recent contact from us"

    def _resolve_pending_click(self, ep: Episode, t: datetime) -> str:
        """The customer opened our link. The payment still has to actually work."""
        method = ep.pending_click_method or ep.method
        ep.pending_click_at = None
        ep.pending_click_method = None
        ok, why = self._charge_succeeds(ep, t, method, customer_present=True)
        if ok:
            ep.paid = True
            ep.paid_at = t
            ep.terminal = Terminal.RECOVERED
            ep.attribution = AttributionTruth.SYSTEM_RECOVERED
            return f"customer completed payment on {method.value} via our link"
        return f"customer opened our link but the {method.value} payment failed: {why}"

    # ── Stepping a policy's action ────────────────────────────────────────

    ACTION_DURATION_MINUTES: dict[ActionType, int] = {
        ActionType.RETRY: 2,
        ActionType.SEND_LINK: 1,
        ActionType.ESCALATE: 25,
        ActionType.GIVE_UP: 0,
    }

    def step(self, ep: Episode, action: Action) -> StepResult:
        """Apply one action, advance the clock, and report what happened."""
        if ep.terminal is not Terminal.OPEN:
            return self._record(ep, action, 0, "episode already closed", invalid=True)
        if ep.now >= ep.deadline:
            self._close_window(ep)
            return self._record(ep, action, 0, "recovery window has expired", invalid=True)

        # Captured before anything moves the clock. Every rule about *when* we did
        # something has to be judged against this, not against where the clock ends
        # up once the action's duration has been applied.
        decided_at = ep.now

        cost, invalid = 0, False
        if action.type is ActionType.WAIT:
            detail = f"waited {max(0, action.wait_minutes)}m"
        elif action.type is ActionType.RETRY:
            cost, detail, invalid = self._do_retry(ep, action)
        elif action.type is ActionType.SEND_LINK:
            cost, detail, invalid = self._do_send_link(ep, action)
        elif action.type is ActionType.ESCALATE:
            cost, detail, invalid = self._do_escalate(ep, action)
        else:
            ep.terminal = Terminal.ABANDONED
            detail = f"gave up: {action.reason or 'no further action judged worthwhile'}"

        ep.cost_paise += cost
        if invalid:
            ep.invalid_actions += 1

        minutes = (
            max(0, action.wait_minutes) if action.type is ActionType.WAIT
            else self.ACTION_DURATION_MINUTES.get(action.type, 0)
        )
        notes = self._advance(ep, ep.now + timedelta(minutes=minutes))
        if notes:
            detail = "; ".join([detail, *notes])
        if ep.terminal is Terminal.OPEN and ep.now >= ep.deadline:
            self._close_window(ep)
        return self._record(ep, action, cost, detail, invalid=invalid, decided_at=decided_at)

    def _record(
        self,
        ep: Episode,
        action: Action,
        cost: int,
        detail: str,
        invalid: bool = False,
        blocked: list[str] | None = None,
        decided_at: datetime | None = None,
    ) -> StepResult:
        at = decided_at if decided_at is not None else ep.now
        result = StepResult(
            t=ep.now,
            action=action,
            paid=ep.paid,
            paid_amount=ep.amount if ep.paid else 0,
            attribution=ep.attribution,
            cost_paise=cost,
            terminal=ep.terminal,
            detail=detail,
            invalid=invalid,
            compliance_blocked=list(blocked or []),
            decided_at=at,
            bank_health=self.world.observed_success_rate(ep.bank, at),
        )
        ep.history.append(result)
        return result

    def _close_window(self, ep: Episode) -> None:
        ep.now = ep.deadline
        if not ep.paid and ep.terminal is not Terminal.ESCALATED:
            ep.terminal = Terminal.ABANDONED

    # ── Action handlers ───────────────────────────────────────────────────

    def _do_retry(self, ep: Episode, action: Action) -> tuple[int, str, bool]:
        """Re-charge server-side. Cheap in rupees, expensive in blocked cards."""
        if not ep.customer.has_mandate:
            return 0, "retry impossible: no token or mandate on file", True
        method = action.method or ep.method
        if method not in ep.customer.alt_methods:
            return 0, f"retry impossible: no {method.value} instrument on file", True

        ep.attempts += 1
        ep.last_retry_at = ep.now
        ok, why = self._charge_succeeds(ep, ep.now, method, customer_present=False)
        if ok:
            ep.paid = True
            ep.paid_at = ep.now
            ep.terminal = Terminal.RECOVERED
            ep.attribution = AttributionTruth.SYSTEM_RECOVERED
            return 0, f"retry on {method.value} authorised", False

        # Retry storms. Hammer an instrument that keeps declining and the issuer
        # stops asking politely — which destroys the value of every later action
        # too, including the ones that would have worked.
        storm_risk = {
            PaymentMethod.CARD: CARD_BLOCK_PROBABILITY_PER_FAILED_RETRY,
            PaymentMethod.UPI: MANDATE_PAUSE_PROBABILITY_PER_FAILED_RETRY,
        }.get(method, 0.0)
        if storm_risk and not ep.instrument_blocked:
            if float(ep.rng_retry.random()) < storm_risk:
                ep.instrument_blocked = True
                killed = "the issuer has blocked the card" if method is PaymentMethod.CARD \
                    else "the bank has paused the autopay mandate"
                return 0, (
                    f"retry on {method.value} failed ({why}) and {killed} — our own "
                    "retries have now made this instrument unusable"
                ), False
        return 0, f"retry on {method.value} failed: {why}", False

    def _do_send_link(self, ep: Episode, action: Action) -> tuple[int, str, bool]:
        """Message the customer a payment link. Costs money and goodwill."""
        if action.channel is None:
            return 0, "link not sent: no channel specified", True
        channel, method = action.channel, action.method or ep.method
        cost = CHANNEL_COST_PAISE[channel]

        # Click-through, before we touch the counters — the first message must not
        # pay a fatigue penalty for itself.
        rate = ep.customer.channel_rate(channel, action.tone)
        rate *= ep.customer.intent_at(ep.now, ep.failed_at)
        rate *= ep.customer.fatigue_multiplier()
        if self._is_quiet_hour(ep.now):
            rate *= QUIET_HOUR_CLICK_FACTOR
        if method not in ep.customer.alt_methods:
            rate *= 0.25

        ep.contacts += 1
        ep.customer.contacts += 1
        ep.last_contact_at = ep.now
        day = ep.now.strftime("%Y-%m-%d")
        ep.contacts_by_day[day] = ep.contacts_by_day.get(day, 0) + 1

        # Both draws happen every time, clicked or not, so two policies sending
        # the same number of messages stay aligned on this stream.
        opened = float(ep.rng_click.random()) < min(MAX_CLICK_PROBABILITY, rate)
        delay = float(np.clip(
            ep.rng_click.lognormal(mean=math.log(22.0), sigma=1.0), 1.0, 2_880.0
        ))
        sent = f"{channel.value} link sent in {action.tone.value} tone suggesting {method.value}"
        if not opened:
            return cost, f"{sent}; not opened (click probability {rate:.1%})", False
        ep.pending_click_at = ep.now + timedelta(minutes=delay)
        ep.pending_click_method = method
        return cost, f"{sent}; customer opens it in ~{delay:.0f}m", False

    def _do_escalate(self, ep: Episode, action: Action) -> tuple[int, str, bool]:
        """Hand to a human agent. The strongest action available, and the scarcest.

        Useless against a dead instrument or an empty account, rationed by a fixed
        bench, and charged whether or not the phone is answered — which together
        are what make it a real decision instead of a free upgrade.
        """
        if ep.escalated:
            return 0, "escalation impossible: already handed to an agent", True
        if self.agent_calls_used >= self.agent_capacity:
            return 0, (
                f"escalation refused: the agent bench is full for this batch "
                f"({self.agent_calls_used}/{self.agent_capacity} calls used)"
            ), True

        self.agent_calls_used += 1
        ep.escalated = True
        ep.terminal = Terminal.ESCALATED
        cost = ESCALATION_COST_PAISE
        # An agent telephoning someone spends one of the day's contact slots, exactly
        # as a message does — the deployed engine checks escalation against that budget,
        # so the simulator has to charge it too or the policy is measured under a
        # cheaper rulebook than production enforces. It is not counted in `ep.contacts`,
        # which models *message* fatigue: a phone call is an imposition, but it is not
        # the thing that makes the next SMS less likely to be read.
        day = ep.now.strftime("%Y-%m-%d")
        ep.contacts_by_day[day] = ep.contacts_by_day.get(day, 0) + 1

        intent = ep.customer.intent_at(ep.now, ep.failed_at)
        # Both draws happen every time so the stream stays aligned whether or not
        # the agent gets through.
        connect_roll = float(ep.rng_agent.random())
        completion_roll = float(ep.rng_agent.random())

        connect = min(
            AGENT_CONNECT_CEILING, AGENT_CONNECT_FLOOR + AGENT_CONNECT_INTENT_WEIGHT * intent
        )
        if connect_roll >= connect:
            return cost, f"agent could not reach the customer (connect rate {connect:.0%})", False

        completion = min(
            AGENT_COMPLETION_CEILING,
            AGENT_COMPLETION_FLOOR + AGENT_COMPLETION_INTENT_WEIGHT * intent,
        )
        if completion_roll >= completion:
            return cost, (
                "agent reached the customer, who declined to complete the payment now "
                f"(in-call completion rate {completion:.0%})"
            ), False

        # With the customer on the line an agent can try every rail they hold and
        # re-enter details by hand. They still cannot conjure balance.
        alternatives = [m for m in ep.customer.alt_methods if m is not ep.method]
        why = "no instrument available"
        for candidate in [*alternatives, ep.method]:
            ok, why = self._charge_succeeds(ep, ep.now, candidate, customer_present=True)
            if ok:
                ep.paid = True
                ep.paid_at = ep.now
                ep.terminal = Terminal.RECOVERED
                ep.attribution = AttributionTruth.SYSTEM_RECOVERED
                return cost, f"agent completed a {candidate.value} payment with the customer", False
        return cost, f"agent reached the customer but nothing would go through: {why}", False

    # ── Closing the books ─────────────────────────────────────────────────

    def finalize(self, ep: Episode) -> Episode:
        """Run the clock to the end of the recovery window and settle the outcome.

        Called for every episode under every policy, including ones a policy gave
        up on. That is deliberate and it is the load-bearing detail of the whole
        measurement: `do_nothing` takes no actions, so `finalize` alone collects
        exactly the payments that were coming back anyway. Every other policy is
        reported as a delta against that number.
        """
        self._advance(ep, ep.deadline)
        ep.now = ep.deadline
        if not ep.paid:
            ep.attribution = AttributionTruth.NOT_RECOVERED
            if ep.terminal is not Terminal.ESCALATED:
                ep.terminal = Terminal.ABANDONED
        return ep



