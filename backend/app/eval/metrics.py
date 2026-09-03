"""What to measure, and why each number is in the table.

The brief asks for measured money recovered across a batch, with compliant
escalation, stopping rules and an audit trail. Every field below exists to make
one of those checkable, and several exist specifically to make it hard to look
good dishonestly:

  `incremental_lift_rupees` is the headline. It is recovery *minus the same batch
  under `do_nothing`*, so a policy gets credit only for money that would not have
  arrived on its own. A raw recovery rate is not reported anywhere without it.

  `system_recovered_rupees` is the same claim from the other direction, built from
  the environment's private verdict on causation rather than from a subtraction.
  It is never below the subtraction, and the gap between them is not error: it is
  money the policy recovered on Monday that would have arrived unprompted on
  Wednesday. Real, faster, and deliberately excluded from the headline. If the
  subtraction ever exceeds it, rupees appeared with no causal story behind them and
  the report says so instead of publishing the flattering one.

  `ambiguous_rupees` is money the policy cannot prove it caused — a customer who
  paid through their own channel shortly after being messaged. It is never counted
  as a win. A policy with a big ambiguous pile is standing next to recoveries.

  `quiet_hour_contacts`, `invalid_actions` and `episodes_at_step_cap` are the
  ways a policy can look profitable while being unshippable: messaging people at
  3am, proposing actions the gateway would reject, and never deciding to stop.
  None of them show up in a recovery rate, and zero is attainable for all three,
  so any count above zero is a defect rather than a trade.

  `self_inflicted_blocks` is the one cost that is not a veto — a failed retry can
  kill a working card whatever the reason for the failure, so the only policy with
  zero of them is one that never retries. It is reported as a rate against the
  instruments that were still alive to be broken, and judged against the incumbent's
  rate rather than against zero.

  `regret_vs_oracle_rupees` puts the whole thing in proportion. Recovering ₹4 lakh
  means nothing until you know whether ₹5 lakh or ₹40 lakh was available.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.policies.base import MAX_STEPS_PER_EPISODE
from app.sim.environment import Episode, RecoveryEnv
from app.sim.types import ActionType, AttributionTruth


@dataclass(frozen=True)
class CauseBreakdown:
    """One root cause's slice of a batch. Latent, so this is diagnostic only —
    no policy sees it, but it is how you tell a policy that diagnoses from a
    policy that got lucky on the easy causes."""

    cause: str
    payments: int
    at_risk_rupees: float
    recovered_rupees: float
    system_recovered_rupees: float
    spend_rupees: float
    escalations: int

    @property
    def recovery_rate(self) -> float:
        return 0.0 if not self.at_risk_rupees else self.recovered_rupees / self.at_risk_rupees


@dataclass(frozen=True)
class BatchMetrics:
    """One policy's complete record on one (scenario, seed). Raw counts only.

    Nothing here is a comparison. Lift and regret need a second policy's numbers
    and live in `Comparison`, so that a single policy's metrics can never quietly
    contain a baseline it was not actually measured against.
    """

    policy: str
    scenario: str
    seed: int

    payments: int
    at_risk_rupees: float

    recovered_payments: int
    recovered_rupees: float
    system_recovered_payments: int
    system_recovered_rupees: float
    ambiguous_payments: int
    ambiguous_rupees: float
    self_recovered_payments: int
    self_recovered_rupees: float

    spend_rupees: float
    retries: int
    contacts: int
    escalations: int
    agent_capacity: int

    invalid_actions: int
    quiet_hour_contacts: int
    self_inflicted_blocks: int
    live_instrument_payments: int
    max_contacts_to_one_payment: int
    episodes_at_step_cap: int
    median_hours_to_recovery: float

    by_cause: dict[str, CauseBreakdown] = field(default_factory=dict)

    # ── Ratios, all guarded against the empty batch ───────────────────────

    @property
    def recovery_rate(self) -> float:
        """Share of at-risk money that came back. Meaningless alone — a batch of
        transient failures recovers most of this with no system at all."""
        return 0.0 if not self.at_risk_rupees else self.recovered_rupees / self.at_risk_rupees

    @property
    def contacts_per_system_recovery(self) -> float:
        """Messages spent per recovery the policy can actually claim."""
        if not self.system_recovered_payments:
            return math.inf if self.contacts else 0.0
        return self.contacts / self.system_recovered_payments

    @property
    def escalation_utilisation(self) -> float:
        """Share of the agent bench actually used. Low means slots went to waste;
        the environment refuses anything above 1.0."""
        return 0.0 if not self.agent_capacity else self.escalations / self.agent_capacity

    @property
    def unprovable_share(self) -> float:
        """Ambiguous rupees as a share of everything the policy recovered. Rises
        when a policy messages people who were coming back anyway."""
        return 0.0 if not self.recovered_rupees else self.ambiguous_rupees / self.recovered_rupees

    @property
    def self_inflicted_block_rate(self) -> float:
        """Share of still-working instruments this policy killed.

        Against live instruments rather than all payments, because a card that was
        already dead at failure time cannot be broken by us and belongs in neither
        the numerator nor the denominator. `stress_dead_instruments` is half dead
        cards by construction, so including them would make that scenario look
        gentler than `baseline` on identical behaviour."""
        if not self.live_instrument_payments:
            return 0.0
        return self.self_inflicted_blocks / self.live_instrument_payments


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def collect(
    policy_name: str, env: RecoveryEnv, episodes: list[Episode]
) -> BatchMetrics:
    """Read a finished batch and produce its record.

    Everything is derived from the episodes themselves — the step history, the
    environment's attribution verdict, the latent flags. Nothing is passed in by
    the policy, so a policy cannot report its own performance.
    """
    paid = [e for e in episodes if e.paid]
    system = [e for e in paid if e.attribution is AttributionTruth.SYSTEM_RECOVERED]
    ambiguous = [e for e in paid if e.attribution is AttributionTruth.AMBIGUOUS]
    unprompted = [e for e in paid if e.attribution is AttributionTruth.CUSTOMER_SELF_RECOVERED]

    quiet_contacts = sum(
        1
        for e in episodes
        for step in e.history
        if step.action.type is ActionType.SEND_LINK
        and not step.invalid
        # `taken_at`, not `t`: a link takes a minute to go out, so the settled clock
        # would charge a message sent legally at 21:59 as a quiet-hours breach.
        and RecoveryEnv._is_quiet_hour(step.taken_at)
    )
    retries = sum(
        1
        for e in episodes
        for step in e.history
        if step.action.type is ActionType.RETRY and not step.invalid
    )
    hours_to_recovery = [
        (e.paid_at - e.failed_at).total_seconds() / 3600.0
        for e in system
        if e.paid_at is not None
    ]

    by_cause: dict[str, CauseBreakdown] = {}
    grouped: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.true_cause].append(episode)
    for cause, group in sorted(grouped.items()):
        by_cause[cause] = CauseBreakdown(
            cause=cause,
            payments=len(group),
            at_risk_rupees=sum(e.amount_rupees for e in group),
            recovered_rupees=sum(e.amount_rupees for e in group if e.paid),
            system_recovered_rupees=sum(
                e.amount_rupees for e in group
                if e.attribution is AttributionTruth.SYSTEM_RECOVERED
            ),
            spend_rupees=sum(e.cost_paise for e in group) / 100.0,
            escalations=sum(1 for e in group if e.escalated),
        )

    return BatchMetrics(
        policy=policy_name,
        scenario=env.scenario.name,
        seed=env.seed,
        payments=len(episodes),
        at_risk_rupees=sum(e.amount_rupees for e in episodes),
        recovered_payments=len(paid),
        recovered_rupees=sum(e.amount_rupees for e in paid),
        system_recovered_payments=len(system),
        system_recovered_rupees=sum(e.amount_rupees for e in system),
        ambiguous_payments=len(ambiguous),
        ambiguous_rupees=sum(e.amount_rupees for e in ambiguous),
        self_recovered_payments=len(unprompted),
        self_recovered_rupees=sum(e.amount_rupees for e in unprompted),
        spend_rupees=sum(e.cost_paise for e in episodes) / 100.0,
        retries=retries,
        contacts=sum(e.contacts for e in episodes),
        escalations=sum(1 for e in episodes if e.escalated),
        agent_capacity=env.agent_capacity,
        invalid_actions=sum(e.invalid_actions for e in episodes),
        quiet_hour_contacts=quiet_contacts,
        self_inflicted_blocks=sum(
            1 for e in episodes if e.instrument_blocked and not e.instrument_dead_at_failure
        ),
        # The denominator for the line above, and it has to be counted here because
        # `instrument_dead_at_failure` is latent. A raw block count is unreadable
        # without it: 80 blocks is a scandal in a batch of 200 live cards and a
        # rounding error in 27,000.
        live_instrument_payments=sum(1 for e in episodes if not e.instrument_dead_at_failure),
        max_contacts_to_one_payment=max((e.contacts for e in episodes), default=0),
        episodes_at_step_cap=sum(
            1 for e in episodes if len(e.history) >= MAX_STEPS_PER_EPISODE
        ),
        median_hours_to_recovery=_median(hours_to_recovery),
        by_cause=by_cause,
    )


LIFT_TOLERANCE_RUPEES = 1.0
"""Float slack for the identity below. Amounts are integer paise summed and
divided by 100, so real breaches are thousands of rupees, never fractions."""


@dataclass(frozen=True)
class Comparison:
    """One policy's number against the two batches that give it meaning.

    A recovery figure on its own is unreadable. It needs a floor — the same
    payments with no system at all — and a ceiling — the same payments under full
    knowledge. Both have to be *the same payments*: same scenario, same seed, same
    generated batch, so the only difference between the three runs is the policy.
    `__post_init__` refuses anything else, which is the whole reason `BatchMetrics`
    carries no comparison of its own.
    """

    metrics: BatchMetrics
    baseline: BatchMetrics
    """`do_nothing` on the identical batch: the money that arrives with no system."""
    ceiling: BatchMetrics | None = None
    """`oracle` on the identical batch, when it was run. Without it, lift is a
    number with no denominator — ₹4 lakh could be most of what was available or a
    tenth of it."""

    def __post_init__(self) -> None:
        mine = (
            self.metrics.scenario,
            self.metrics.seed,
            self.metrics.payments,
            round(self.metrics.at_risk_rupees, 2),
        )
        for other, label in ((self.baseline, "baseline"), (self.ceiling, "ceiling")):
            if other is None:
                continue
            theirs = (other.scenario, other.seed, other.payments, round(other.at_risk_rupees, 2))
            if theirs != mine:
                raise ValueError(
                    f"{self.metrics.policy} was measured on {mine} but its {label} "
                    f"({other.policy}) was measured on {theirs} — a lift figure across "
                    "two different batches is meaningless"
                )

    @property
    def policy(self) -> str:
        return self.metrics.policy

    @property
    def scenario(self) -> str:
        return self.metrics.scenario

    @property
    def seed(self) -> int:
        return self.metrics.seed

    # ── The headline, and the same claim computed a second way ────────────

    @property
    def incremental_lift_rupees(self) -> float:
        """The headline. Rupees this policy brought in that `do_nothing` did not.

        A subtraction over identical batches, which is the only reason it can be
        called causal: the customers, the outages and every coin flip are shared
        between the two runs, so the difference is the policy and nothing else.
        """
        return self.metrics.recovered_rupees - self.baseline.recovered_rupees

    @property
    def attributed_lift_rupees(self) -> float:
        """The same claim from the environment's own verdict on causation, with no
        baseline involved. Built from `SYSTEM_RECOVERED` episodes only."""
        return self.metrics.system_recovered_rupees

    @property
    def preempted_rupees(self) -> float:
        """The gap between the two, which is a real quantity rather than error.

        A payment the policy recovers on Monday might have arrived unprompted on
        Wednesday. The environment records that as `SYSTEM_RECOVERED` — we did cause
        *that* payment — but it adds nothing to the subtraction, because the money
        shows up in the baseline too. That difference is this number: money the
        policy got to first. Faster, and worth something to a real merchant, but not
        incremental, so it never enters the headline.
        """
        return self.attributed_lift_rupees - self.incremental_lift_rupees

    @property
    def lift_identity_holds(self) -> bool:
        """`incremental_lift <= attributed_lift`, which must be true by construction.

        Unprompted payment times are action-independent and `finalize` collects them
        under every policy, so every rupee of lift has to have a `SYSTEM_RECOVERED`
        episode behind it. If this goes false, money appeared in the subtraction with
        no causal story attached and the two estimates are measuring different things
        — a bug in the environment or the harness, not a good result.
        """
        return self.incremental_lift_rupees <= self.attributed_lift_rupees + LIFT_TOLERANCE_RUPEES

    # ── Was it worth doing ────────────────────────────────────────────────

    @property
    def net_lift_rupees(self) -> float:
        """Lift after what it cost to get. The number a merchant would actually
        sign off on, and the one a message-everybody policy fails."""
        return self.incremental_lift_rupees - self.metrics.spend_rupees

    @property
    def spend_per_incremental_rupee(self) -> float:
        """Paise-level efficiency. `inf` when a policy spent money and recovered
        nothing incremental, which is the honest answer, not a zero."""
        lift = self.incremental_lift_rupees
        if lift <= 0:
            return math.inf if self.metrics.spend_rupees > 0 else 0.0
        return self.metrics.spend_rupees / lift

    @property
    def achievable_lift_rupees(self) -> float | None:
        """What perfect play recovered over `do_nothing` on this same batch."""
        if self.ceiling is None:
            return None
        return self.ceiling.recovered_rupees - self.baseline.recovered_rupees

    @property
    def regret_vs_oracle_rupees(self) -> float | None:
        """Money that was on the table and left there. Never negative — if it comes
        out negative the oracle is not a ceiling, and `test_policies.py` fails."""
        achievable = self.achievable_lift_rupees
        if achievable is None:
            return None
        return achievable - self.incremental_lift_rupees

    @property
    def share_of_achievable_lift(self) -> float | None:
        """Lift as a fraction of the ceiling. This is the figure that survives
        changing the scenario mix, because a batch of easy transient failures
        inflates every recovery rate and this ratio not at all."""
        achievable = self.achievable_lift_rupees
        if achievable is None:
            return None
        return 0.0 if achievable <= 0 else self.incremental_lift_rupees / achievable

    # ── Would it survive a review ─────────────────────────────────────────

    @property
    def blocking_violations(self) -> list[str]:
        """Reasons this policy could not be deployed, whatever its lift.

        Kept separate from the money on purpose. Every item here is invisible in a
        recovery rate, and each one is a thing a payments team would refuse to ship:
        messaging people overnight, proposing actions the gateway rejects, failing
        to terminate, and reporting two figures that contradict each other.

        Instruments we blocked is deliberately *not* here; it is in `harms`. The
        distinction is whether zero is attainable. A policy can send no message
        after 22:00 and propose no impossible action by construction, so any count
        above zero is a defect. But a failed retry blocks a card with probability
        0.06 whatever the reason for the failure, so the only policy with zero
        blocks is one that never retries — and lumping that in here made the column
        useless, printing "not shippable" against everything that acted at all and
        putting the strawman and the proposal under the same word.
        """
        m = self.metrics
        problems: list[str] = []
        if m.quiet_hour_contacts:
            problems.append(
                f"{m.quiet_hour_contacts} messages sent inside quiet hours (22:00–08:00 IST)"
            )
        if m.invalid_actions:
            problems.append(
                f"{m.invalid_actions} actions the gateway refused as structurally impossible"
            )
        if m.episodes_at_step_cap:
            problems.append(
                f"{m.episodes_at_step_cap} payments hit the {MAX_STEPS_PER_EPISODE}-step cap "
                "without the policy deciding to stop"
            )
        if not self.lift_identity_holds:
            problems.append(
                f"lift (₹{self.incremental_lift_rupees:,.0f}) exceeds attributed recovery "
                f"(₹{self.attributed_lift_rupees:,.0f}) — the two measurements disagree"
            )
        return problems

    @property
    def harms(self) -> list[str]:
        """Damage the policy did on the way to its lift. Priced, not vetoed.

        This is the cost side of the trade a recovery system actually makes: some
        customers end up worse off than if nothing had been done. It cannot be
        driven to zero without giving up retries entirely, so it is underwritten
        rather than forbidden — against the incumbent's rate, and against the money
        the retries brought in.
        """
        m = self.metrics
        if not m.self_inflicted_blocks:
            return []
        return [
            f"{m.self_inflicted_blocks} of {m.live_instrument_payments:,} working "
            f"instruments blocked by our own retries "
            f"({m.self_inflicted_block_rate:.2%}) — those customers were left worse "
            "off than if we had done nothing"
        ]

    @property
    def is_shippable(self) -> bool:
        return not self.blocking_violations


def merge_causes(breakdowns: Iterable[CauseBreakdown]) -> dict[str, CauseBreakdown]:
    """Pool per-batch cause slices into one, summing rupees rather than averaging.

    Averaging per-batch recovery rates by cause would weight a seed that produced
    three `MERCHANT_ERROR` payments the same as one that produced forty, and the rare
    causes are exactly the ones where the rates are most unstable.
    """
    pooled: dict[str, list[CauseBreakdown]] = defaultdict(list)
    for breakdown in breakdowns:
        pooled[breakdown.cause].append(breakdown)
    return {
        cause: CauseBreakdown(
            cause=cause,
            payments=sum(b.payments for b in group),
            at_risk_rupees=sum(b.at_risk_rupees for b in group),
            recovered_rupees=sum(b.recovered_rupees for b in group),
            system_recovered_rupees=sum(b.system_recovered_rupees for b in group),
            spend_rupees=sum(b.spend_rupees for b in group),
            escalations=sum(b.escalations for b in group),
        )
        for cause, group in sorted(pooled.items())
    }


def compare(
    metrics: BatchMetrics,
    baseline: BatchMetrics,
    ceiling: BatchMetrics | None = None,
) -> Comparison:
    """Score one policy against the same batch under `do_nothing` and the oracle."""
    return Comparison(metrics=metrics, baseline=baseline, ceiling=ceiling)



