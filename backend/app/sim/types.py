"""Action space, observation space, and step results for the recovery environment.

Deliberate separation of concerns:

  Observation — everything a policy is allowed to see. Derived only from
                observable payment fields and aggregate history. Never contains
                latent state.
  Action      — what a policy may do. A closed set, so cost and compliance can
                be accounted for exactly.
  StepResult  — what happened, including the environment's private verdict on
                attribution (used for scoring, never shown to the policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.models.schemas import PaymentMethod


class ActionType(str, Enum):
    """The five things a recovery policy can do."""

    WAIT = "WAIT"
    """Do nothing for N minutes. Free, but intent decays while you wait."""

    RETRY = "RETRY"
    """Re-attempt the charge server-side. Requires a mandate/token. No customer
    friction, so it is the cheapest real action — when it is legal."""

    SEND_LINK = "SEND_LINK"
    """Send the customer a payment link on a channel. Costs money per message
    and permanently consumes goodwill."""

    ESCALATE = "ESCALATE"
    """Hand to a human agent. Expensive, sometimes works, ends the episode."""

    GIVE_UP = "GIVE_UP"
    """Stop spending on this payment. Ends the episode. Not a failure state —
    knowing when to stop is part of a good policy."""


class Channel(str, Enum):
    """Outreach channels, in increasing order of cost and effectiveness."""

    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"


class Tone(str, Enum):
    """Message register. Fit to the customer's persona matters; see customer.py."""

    FRIENDLY = "friendly"
    URGENT = "urgent"
    INFORMATIONAL = "informational"
    HINGLISH = "hinglish"


class Terminal(str, Enum):
    """Episode end state."""

    OPEN = "OPEN"
    RECOVERED = "RECOVERED"
    ESCALATED = "ESCALATED"
    ABANDONED = "ABANDONED"
    """Policy gave up, or the recovery window expired."""


class AttributionTruth(str, Enum):
    """The environment's private verdict on who actually recovered the money.

    This is the honest-metrics core. A system that counts self-recoveries as its
    own wins is reporting a number that means nothing.
    """

    SYSTEM_RECOVERED = "SYSTEM_RECOVERED"
    """Paid as a direct result of our action: our retry cleared, or the customer
    followed our link."""

    CUSTOMER_SELF_RECOVERED = "CUSTOMER_SELF_RECOVERED"
    """The customer paid on their own, and we had not contacted them recently
    enough to plausibly claim credit."""

    AMBIGUOUS = "AMBIGUOUS"
    """The customer paid through their own channel shortly after we contacted
    them. Causation is genuinely unprovable. Reported separately, never counted
    as a win."""

    NOT_RECOVERED = "NOT_RECOVERED"


@dataclass(frozen=True)
class Action:
    """A single policy action. Frozen so it can be logged without aliasing."""

    type: ActionType
    wait_minutes: int = 0
    method: PaymentMethod | None = None
    """For RETRY: which method to charge. For SEND_LINK: which method to suggest."""
    channel: Channel | None = None
    tone: Tone = Tone.FRIENDLY
    reason: str = ""
    """Free-text justification. Goes straight into the audit trail."""

    def label(self) -> str:
        """Short human-readable form, used in traces and the dashboard."""
        if self.type is ActionType.WAIT:
            return f"WAIT {self.wait_minutes}m"
        if self.type is ActionType.RETRY:
            return f"RETRY {self.method.value if self.method else '?'}"
        if self.type is ActionType.SEND_LINK:
            method = self.method.value if self.method else "any"
            channel = self.channel.value if self.channel else "?"
            return f"LINK {channel}/{method}/{self.tone.value}"
        return self.type.value


@dataclass
class BankSignal:
    """Observable bank health. Aggregate only — a policy cannot see whether a
    specific bank is *actually* down, only the failure rate it can measure."""

    bank: str
    observed_success_rate_1h: float = 0.95
    observed_failure_count_1h: int = 0
    concurrent_failure_spike: bool = False
    """True when this bank's recent failure count is well above its baseline.
    A noisy proxy for downtime, which is what a real system actually has."""


@dataclass
class CustomerSignal:
    """Observable customer history. Aggregates over past *observed* behaviour."""

    success_rate_90d: float = 0.9
    failure_count_7d: int = 0
    prior_recovery_responses: int = 0
    prior_recovery_contacts: int = 0

    @property
    def response_rate(self) -> float:
        """Historical click-through on recovery messages. 0.5 prior when unseen."""
        if self.prior_recovery_contacts == 0:
            return 0.5
        return self.prior_recovery_responses / self.prior_recovery_contacts


@dataclass
class Observation:
    """Everything a policy sees at decision time. No latent state, by construction."""

    payment_id: str
    amount: int
    method: PaymentMethod
    bank: str
    error_code: str
    error_source: str
    error_step: str
    error_reason: str
    error_description: str

    now: datetime
    failed_at: datetime

    attempts_made: int = 0
    contacts_made: int = 0
    escalated: bool = False

    bank_signal: BankSignal = field(default_factory=lambda: BankSignal(bank="unknown"))
    customer_signal: CustomerSignal = field(default_factory=CustomerSignal)

    has_mandate: bool = False
    """Whether a server-side retry is even possible. Observable: we know if we
    hold a token or an autopay mandate."""

    available_methods: list[PaymentMethod] = field(default_factory=list)
    """Methods this customer has previously used successfully — so suggesting an
    alternative is grounded rather than a guess."""

    agent_calls_remaining: int = 0
    """Human agent calls left in the batch's bench for this shift. Shared across
    every payment, so spending one here means not spending it on the next
    payment — which is the whole reason escalation is a decision and not a
    default. A real merchant has a fixed number of agents, and any system that
    escalates without tracking this is writing cheques the ops team cannot cash."""

    agent_capacity: int = 0
    """Size of that bench, so a policy can reason about the fraction left rather
    than an absolute count it has no scale for."""

    last_action: Action | None = None
    last_detail: str = ""

    @property
    def minutes_since_failure(self) -> float:
        return (self.now - self.failed_at).total_seconds() / 60.0

    @property
    def amount_rupees(self) -> float:
        return self.amount / 100.0

    @property
    def amount_bucket(self) -> str:
        r = self.amount_rupees
        if r < 100:
            return "micro"
        if r < 1000:
            return "small"
        if r < 10000:
            return "medium"
        if r < 50000:
            return "large"
        return "premium"


@dataclass
class StepResult:
    """Outcome of one action. `attribution` is scoring-only — policies never see it."""

    t: datetime
    action: Action
    paid: bool
    paid_amount: int
    attribution: AttributionTruth
    cost_paise: int
    terminal: Terminal
    detail: str
    invalid: bool = False
    """Set when a policy attempted something structurally illegal, e.g. a
    server-side retry with no mandate. Counted and reported — a policy that
    proposes impossible actions is not production-ready."""
    compliance_blocked: list[str] = field(default_factory=list)
