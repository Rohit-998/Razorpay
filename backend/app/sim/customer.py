"""Latent customer state — the private half of the causal model.

Nothing in this module is visible to a policy. A policy sees a failed payment
and aggregate history; it must infer the rest.

The variable that earns its keep here is `self_recover_at`: the moment this
customer would have paid with no intervention at all. It is sampled at episode
creation, before any policy runs, and it is never influenced by what a policy
does. Two consequences follow:

  1. The `do_nothing` baseline recovers exactly the customers whose
     `self_recover_at` lands inside the recovery window. That is the honest
     counterfactual every other policy is measured against.
  2. When a customer self-recovers shortly after we contacted them, we can
     detect it and refuse to claim credit. That is `AMBIGUOUS` attribution.

Fast self-recovery is concentrated in exactly the failure modes a careless
system fires on hardest — an abandoned 3DS challenge, a dropped packet. A system
that pings on those and books the resulting payment as its own win reports a
number that is mostly other people's work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from app.models.schemas import PaymentMethod
from app.sim.types import Channel, Tone


@dataclass(frozen=True)
class Persona:
    """A behavioural archetype. Weights are share-of-failures, not share-of-traffic."""

    name: str
    weight: float
    preferred_method: PaymentMethod
    amount_bucket: str

    intent_mean: float
    """Base probability this person still wants the purchase at t=0."""
    intent_halflife_hours: float
    """Hours for remaining intent to halve. Impulse buys decay in hours; a
    business invoice barely decays across days."""

    channel_response: dict[Channel, float]
    """Base click-through per channel, before intent, fatigue and timing."""
    tone_fit: dict[Tone, float]
    """Multiplier on click-through for message register. Getting this wrong is a
    real, measurable cost — which is what makes LLM message generation worth
    having rather than decorative."""

    fatigue_decay: float
    """Click-through multiplier per prior contact. 0.75 means the third message
    lands at 56% of the first."""

    mandate_probability: float
    """Chance this customer has a token or autopay mandate on file, making a
    server-side retry legal at all."""

    self_recover_propensity: float
    """Scales the chance of paying unprompted. Independent of our actions."""


PERSONAS: tuple[Persona, ...] = (
    Persona(
        name="salary_regular",
        weight=0.34,
        preferred_method=PaymentMethod.UPI,
        amount_bucket="small",
        intent_mean=0.72,
        intent_halflife_hours=40.0,
        channel_response={Channel.WHATSAPP: 0.42, Channel.SMS: 0.26, Channel.EMAIL: 0.07},
        tone_fit={Tone.HINGLISH: 1.22, Tone.FRIENDLY: 1.0, Tone.URGENT: 0.88, Tone.INFORMATIONAL: 0.80},
        fatigue_decay=0.80,
        mandate_probability=0.10,
        self_recover_propensity=0.55,
    ),
    Persona(
        name="premium_shopper",
        weight=0.16,
        preferred_method=PaymentMethod.CARD,
        amount_bucket="large",
        intent_mean=0.58,
        intent_halflife_hours=7.0,
        channel_response={Channel.WHATSAPP: 0.20, Channel.SMS: 0.24, Channel.EMAIL: 0.19},
        tone_fit={Tone.HINGLISH: 0.62, Tone.FRIENDLY: 1.0, Tone.URGENT: 1.10, Tone.INFORMATIONAL: 1.05},
        fatigue_decay=0.62,
        mandate_probability=0.30,
        self_recover_propensity=0.62,
    ),
    Persona(
        name="occasional_user",
        weight=0.24,
        preferred_method=PaymentMethod.UPI,
        amount_bucket="micro",
        intent_mean=0.40,
        intent_halflife_hours=12.0,
        channel_response={Channel.WHATSAPP: 0.30, Channel.SMS: 0.14, Channel.EMAIL: 0.03},
        tone_fit={Tone.HINGLISH: 1.30, Tone.FRIENDLY: 1.0, Tone.URGENT: 0.75, Tone.INFORMATIONAL: 0.70},
        fatigue_decay=0.70,
        mandate_probability=0.04,
        self_recover_propensity=0.34,
    ),
    Persona(
        name="business_buyer",
        weight=0.16,
        preferred_method=PaymentMethod.NETBANKING,
        amount_bucket="premium",
        intent_mean=0.88,
        intent_halflife_hours=110.0,
        channel_response={Channel.WHATSAPP: 0.12, Channel.SMS: 0.18, Channel.EMAIL: 0.46},
        tone_fit={Tone.HINGLISH: 0.45, Tone.FRIENDLY: 0.92, Tone.URGENT: 1.02, Tone.INFORMATIONAL: 1.25},
        fatigue_decay=0.86,
        mandate_probability=0.12,
        self_recover_propensity=0.70,
    ),
    Persona(
        name="subscription_holder",
        weight=0.10,
        preferred_method=PaymentMethod.CARD,
        amount_bucket="medium",
        intent_mean=0.90,
        intent_halflife_hours=200.0,
        channel_response={Channel.WHATSAPP: 0.26, Channel.SMS: 0.22, Channel.EMAIL: 0.24},
        tone_fit={Tone.HINGLISH: 0.90, Tone.FRIENDLY: 1.05, Tone.URGENT: 0.95, Tone.INFORMATIONAL: 1.12},
        fatigue_decay=0.82,
        mandate_probability=0.94,
        self_recover_propensity=0.30,
    ),
)

PERSONA_BY_NAME: dict[str, Persona] = {p.name: p for p in PERSONAS}

NAMES: tuple[str, ...] = (
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Gupta", "Vikram Singh",
    "Ananya Reddy", "Rohan Joshi", "Meera Nair", "Arjun Malhotra", "Kavita Desai",
    "Siddharth Iyer", "Pooja Verma", "Divya Menon", "Kunal Agarwal", "Nisha Rao",
    "Aditya Chauhan", "Ritu Saxena", "Manish Tiwari", "Swati Pillai", "Deepak Pandey",
    "Ankita Jain", "Suresh Yadav", "Neha Kapoor", "Karthik Srinivasan", "Pallavi Bhat",
    "Gaurav Mishra", "Shreya Das", "Nikhil Banerjee", "Lakshmi Nambiar", "Faizan Ahmed",
    "Tanvi Kulkarni", "Harish Gowda", "Ishita Chatterjee", "Varun Sethi", "Sanya Bakshi",
)


@dataclass
class CustomerLatent:
    """Private per-customer state for one recovery episode.

    Every field below is hidden from policies. The observable projection is built
    in `environment.py` and contains only aggregates a real system would have.
    """

    customer_id: str
    name: str
    phone: str
    email: str
    persona: Persona

    intent0: float
    """Willingness to complete the purchase at failure time."""
    intent_halflife_hours: float
    funds_available_at: datetime | None
    """When money lands. Before this, a charge on a drained account fails no
    matter how cleverly it is timed. `None` means funds were never the problem."""
    has_mandate: bool
    alt_methods: list[PaymentMethod]
    """Methods this person can actually pay with. Suggesting one they don't have
    is wasted spend."""
    self_recover_at: datetime | None
    """When they would pay unprompted. The counterfactual. Never action-dependent."""
    credential_fix_at: datetime | None
    """For a credentials failure: when they'd notice and correct it. Until then,
    retrying the same details is guaranteed to fail."""

    contacts: int = 0
    """Messages we have sent this episode. Drives fatigue."""
    observed_contacts: int = 0
    observed_responses: int = 0
    history_success_rate: float = 0.9
    history_failures_7d: int = 0

    def intent_at(self, t: datetime, failed_at: datetime) -> float:
        """Remaining purchase intent, decaying exponentially from failure time."""
        hours = max(0.0, (t - failed_at).total_seconds() / 3600.0)
        return float(self.intent0 * math.pow(0.5, hours / self.intent_halflife_hours))

    def fatigue_multiplier(self) -> float:
        """Click-through penalty from messages already sent this episode."""
        return float(math.pow(self.persona.fatigue_decay, self.contacts))

    def channel_rate(self, channel: Channel, tone: Tone) -> float:
        """Base click-through for a channel/tone pair, before intent and fatigue."""
        base = self.persona.channel_response.get(channel, 0.05)
        return float(base * self.persona.tone_fit.get(tone, 1.0))

    def funds_ready(self, t: datetime) -> bool:
        """Whether the account can actually cover the charge at time `t`."""
        return self.funds_available_at is None or t >= self.funds_available_at

    def credentials_valid(self, t: datetime) -> bool:
        """Whether stored payment details would authenticate at time `t`."""
        return self.credential_fix_at is None or t >= self.credential_fix_at


# Root cause → unprompted-payment behaviour:
#   (base probability, median delay in minutes, lognormal spread)
#
# The two fastest rows are the reason honest attribution matters. A dropped
# packet or an abandoned OTP screen leaves an engaged customer who was very
# likely to retry within minutes on their own. Fire an instant retry into that
# window, book the payment, and you have a dashboard full of other people's work.
SELF_RECOVERY_PROFILE: dict[str, tuple[float, float, float]] = {
    "NETWORK_TRANSIENT": (0.74, 6.0, 1.05),
    "AUTH_TIMEOUT": (0.60, 12.0, 1.30),
    "BANK_DOWNTIME": (0.42, 95.0, 1.20),
    "INSUFFICIENT_FUNDS": (0.48, 45.0, 0.95),
    "WRONG_CREDENTIALS": (0.31, 240.0, 1.40),
    "PERMANENT_DECLINE": (0.17, 420.0, 1.50),
    "MERCHANT_ERROR": (0.02, 600.0, 1.50),
}


def sample_persona(rng: np.random.Generator) -> Persona:
    """Draw a persona by failure share."""
    weights = np.array([p.weight for p in PERSONAS], dtype=float)
    return PERSONAS[int(rng.choice(len(PERSONAS), p=weights / weights.sum()))]


def sample_customer(
    rng: np.random.Generator,
    index: int,
    persona: Persona,
    cause: str,
    failed_at: datetime,
    method: PaymentMethod,
    window_hours: int,
) -> CustomerLatent:
    """Build the hidden state for one episode's customer.

    `window_hours` bounds the recovery window; `self_recover_at` is allowed to
    fall outside it, in which case the customer simply never comes back in time
    and no policy — including `do_nothing` — gets credit.
    """
    name = NAMES[int(rng.integers(0, len(NAMES)))]
    slug = name.lower().replace(" ", ".")

    intent0 = float(np.clip(rng.normal(persona.intent_mean, 0.13), 0.05, 0.99))
    halflife = float(max(1.5, rng.normal(persona.intent_halflife_hours,
                                         persona.intent_halflife_hours * 0.25)))

    funds_available_at = _sample_funds_arrival(rng, cause, failed_at)
    credential_fix_at = _sample_credential_fix(rng, cause, failed_at)
    self_recover_at = _sample_self_recovery(
        rng, cause, failed_at, intent0, persona, funds_available_at
    )

    return CustomerLatent(
        customer_id=f"cust_{index:05d}",
        name=name,
        phone=f"+91{int(rng.integers(7000000000, 9999999999))}",
        email=f"{slug}@example.com",
        persona=persona,
        intent0=intent0,
        intent_halflife_hours=halflife,
        funds_available_at=funds_available_at,
        has_mandate=bool(rng.random() < persona.mandate_probability),
        alt_methods=_sample_alt_methods(rng, persona, method),
        self_recover_at=self_recover_at,
        credential_fix_at=credential_fix_at,
        history_success_rate=float(np.clip(rng.normal(0.88, 0.09), 0.35, 0.995)),
        history_failures_7d=int(rng.poisson(0.7)),
        observed_contacts=int(rng.poisson(1.2)),
    )


# All simulation datetimes are naive and represent IST. Keeping one clock avoids
# a class of off-by-5.5-hours bugs, and quiet-hours logic reads directly off
# `.hour`. Conversion to UTC happens only at the persistence boundary.

SALARY_DAYS: tuple[int, ...] = (1, 7, 15)


def _sample_funds_arrival(
    rng: np.random.Generator, cause: str, failed_at: datetime
) -> datetime | None:
    """When money lands in the account. `None` when funds were never the issue."""
    if cause != "INSUFFICIENT_FUNDS":
        return None

    if rng.random() < 0.58:
        # Waiting on a salary credit — the dominant Indian pattern, and the reason
        # a retry scheduled for the right morning beats three retries today.
        candidates = []
        for month_offset in (0, 1):
            month = failed_at.month + month_offset
            year = failed_at.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            for day in SALARY_DAYS:
                try:
                    candidates.append(datetime(year, month, day, 10, 30))
                except ValueError:
                    continue
        future = [c for c in candidates if c > failed_at]
        if future:
            return min(future) + timedelta(minutes=float(rng.integers(0, 420)))

    hours = float(np.clip(rng.lognormal(mean=math.log(20.0), sigma=0.9), 1.0, 200.0))
    return failed_at + timedelta(hours=hours)


def _sample_credential_fix(
    rng: np.random.Generator, cause: str, failed_at: datetime
) -> datetime | None:
    """When the customer corrects bad payment details. `None` when they were fine."""
    if cause != "WRONG_CREDENTIALS":
        return None
    if rng.random() < 0.35:
        # Never noticed. Every retry on stored details is guaranteed to fail, and
        # a policy that keeps trying is burning attempts it could have escalated.
        return failed_at + timedelta(days=999)
    hours = float(np.clip(rng.lognormal(mean=math.log(5.0), sigma=1.1), 0.2, 160.0))
    return failed_at + timedelta(hours=hours)


def _sample_self_recovery(
    rng: np.random.Generator,
    cause: str,
    failed_at: datetime,
    intent0: float,
    persona: Persona,
    funds_available_at: datetime | None,
) -> datetime | None:
    """Sample when this customer would pay unprompted, or `None` if they never would.

    Drawn once, before any policy acts, and never re-drawn. This is the whole
    basis of the counterfactual.
    """
    base_prob, median_minutes, sigma = SELF_RECOVERY_PROFILE.get(
        cause, (0.2, 120.0, 1.3)
    )
    prob = float(np.clip(base_prob * persona.self_recover_propensity * 2.0 * intent0,
                         0.0, 0.92))
    if rng.random() >= prob:
        return None

    delay = float(np.clip(median_minutes * math.exp(sigma * rng.normal()), 1.0, 20160.0))
    # Nobody pays out of a drained account, however willing they are.
    anchor = failed_at if funds_available_at is None else max(failed_at, funds_available_at)
    return anchor + timedelta(minutes=delay)


def _sample_alt_methods(
    rng: np.random.Generator, persona: Persona, method: PaymentMethod
) -> list[PaymentMethod]:
    """Methods this customer can actually pay with, including the one that failed."""
    pool: set[PaymentMethod] = {method, persona.preferred_method}
    for candidate, probability in (
        (PaymentMethod.UPI, 0.86),
        (PaymentMethod.CARD, 0.44),
        (PaymentMethod.NETBANKING, 0.31),
        (PaymentMethod.WALLET, 0.24),
    ):
        if rng.random() < probability:
            pool.add(candidate)
    return sorted(pool, key=lambda m: m.value)



