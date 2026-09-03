"""Named scenarios — the batches policies are evaluated on.

Multiple scenarios exist for one reason: a policy tuned to look good on an average
week can be terrible during an outage, and a policy that never gives up looks fine
until it meets a batch of dead cards. Reporting a single blended number hides
exactly the behaviour worth knowing about, so the eval runs every scenario
separately and reports them side by side.

`stress_dead_instruments` is the one that matters most for the stopping-rule
claim. Most of that batch cannot be recovered by anyone. A policy that keeps
spending on it will show a healthy recovery count and a catastrophic cost per
rupee, and there is no way to dress that up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Base share of each latent cause across a normal week.
BASE_CAUSE_WEIGHTS: dict[str, float] = {
    "BANK_DOWNTIME": 0.20,
    "NETWORK_TRANSIENT": 0.14,
    "AUTH_TIMEOUT": 0.21,
    "INSUFFICIENT_FUNDS": 0.22,
    "WRONG_CREDENTIALS": 0.07,
    "PERMANENT_DECLINE": 0.11,
    "MERCHANT_ERROR": 0.05,
}


@dataclass(frozen=True)
class Scenario:
    """A reproducible batch of failed payments plus the world they failed in."""

    name: str
    description: str
    n_payments: int = 400
    duration_days: int = 7
    window_hours: int = 72
    """How long after failure a recovery still counts."""
    downtime_multiplier: float = 1.0
    cause_weights: dict[str, float] = field(default_factory=lambda: dict(BASE_CAUSE_WEIGHTS))
    start: datetime = datetime(2026, 3, 24, 0, 0)
    """Chosen so the window spans a month end and the 1st-of-month salary credit."""
    merchant_error_windows: int = 2
    """Config breakages are bursty; this many windows get clustered failures."""

    agent_call_share: float = 0.05
    """Share of the batch a human agent can actually call.

    This is the constraint that makes escalation interesting. Without it the
    optimal policy is "phone everyone": a ₹90 call against an average basket of
    several thousand rupees has an absurd expected return, so an unconstrained
    environment rewards a policy with no judgement in it at all. Real merchants
    do not have that option — the recovery bench is a handful of people, and it
    is sized in percent of failed volume, not in payments. At 5% of 400 failed
    payments the policy gets twenty calls and has to decide which twenty.
    """

    def normalised_weights(self) -> dict[str, float]:
        total = sum(self.cause_weights.values())
        return {k: v / total for k, v in self.cause_weights.items()}

    @property
    def agent_capacity(self) -> int:
        """Absolute number of agent calls available for this batch."""
        return max(1, int(round(self.n_payments * self.agent_call_share)))


def _weights(**overrides: float) -> dict[str, float]:
    merged = dict(BASE_CAUSE_WEIGHTS)
    merged.update(overrides)
    return merged


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        name="baseline",
        description=(
            "An ordinary week. Mixed causes in production-like proportions, "
            "routine bank downtime. The headline number comes from here."
        ),
    ),
    "outage_day": Scenario(
        name="outage_day",
        description=(
            "A bad day for the banks. Four times the usual downtime, concentrated "
            "over 48 hours. Rewards policies that infer an outage from correlated "
            "failures and wait for recovery instead of retrying into a wall."
        ),
        n_payments=320,
        duration_days=2,
        window_hours=48,
        downtime_multiplier=4.0,
        cause_weights=_weights(BANK_DOWNTIME=0.42, NETWORK_TRANSIENT=0.18, AUTH_TIMEOUT=0.14),
    ),
    "salary_week": Scenario(
        name="salary_week",
        description=(
            "Month end into the 1st. Balances are empty and refill on payday, so "
            "recovery is almost entirely a timing problem: the same retry that "
            "fails on the 29th succeeds on the 1st."
        ),
        n_payments=380,
        duration_days=8,
        window_hours=96,
        cause_weights=_weights(INSUFFICIENT_FUNDS=0.44, AUTH_TIMEOUT=0.16, BANK_DOWNTIME=0.14),
        start=datetime(2026, 3, 26, 0, 0),
    ),
    "festival_spike": Scenario(
        name="festival_spike",
        description=(
            "Sale weekend. High volume, larger baskets, and authentication failures "
            "from 3DS and UPI-app load. Contact budgets bind hardest here, so cost "
            "per recovered rupee separates policies more than recovery rate does."
        ),
        n_payments=560,
        duration_days=3,
        window_hours=48,
        downtime_multiplier=1.8,
        cause_weights=_weights(AUTH_TIMEOUT=0.34, NETWORK_TRANSIENT=0.20, BANK_DOWNTIME=0.18),
    ),
    "stress_dead_instruments": Scenario(
        name="stress_dead_instruments",
        description=(
            "Adversarial batch: mostly blocked cards, expired mandates and stale "
            "credentials. Most of this money is genuinely unrecoverable. The only "
            "way to score well is to recognise that early and stop spending."
        ),
        n_payments=300,
        duration_days=5,
        cause_weights=_weights(
            PERMANENT_DECLINE=0.38,
            WRONG_CREDENTIALS=0.22,
            MERCHANT_ERROR=0.14,
            AUTH_TIMEOUT=0.10,
            INSUFFICIENT_FUNDS=0.08,
            BANK_DOWNTIME=0.05,
            NETWORK_TRANSIENT=0.03,
        ),
    ),
}

DEFAULT_SCENARIO = "baseline"
