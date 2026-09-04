"""Runs every policy over every scenario at many seeds, and puts error bars on it.

One seed is not a result. The single largest payment in a batch can be worth 6% of
the whole lift figure, and whether it lands turns on one coin flip against a
threshold that moves a few percentage points when a decision is taken an hour
later. A table built from seed 1 would report that as a finding.

Two things make the numbers here trustworthy:

  *Pairing.* Every policy faces the identical batch at each seed — the same
  customers, the same outages, the same draws. So the per-seed difference
  `policy − do_nothing` is a paired observation, and the variance of the difference
  is far smaller than the variance of either side. This is the whole reason a
  handful of seeds is enough to separate policies that differ by a few percent.

  *Bootstrap intervals over the paired differences.* Rupee lift is a sum over a
  heavy-tailed amount distribution, so the interval is resampled rather than
  assumed normal. It is seeded, so the interval is reproducible rather than
  something that moves every time the report is regenerated.

An interval that straddles zero is reported as straddling zero. That is the point
of computing it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from app.eval.metrics import (
    BatchMetrics,
    CauseBreakdown,
    Comparison,
    collect,
    compare,
    merge_causes,
)
from app.policies.base import Policy, run_episode
from app.sim.environment import RecoveryEnv
from app.sim.scenarios import SCENARIOS, Scenario

PolicyBuilder = Callable[[RecoveryEnv], Policy]
"""Built per batch, not once: the oracle needs the environment it will be scored
against, and a learning policy needs somewhere to reset per-batch state."""

BASELINE_POLICY = "do_nothing"
CEILING_POLICY = "oracle"
PROPOSAL_POLICY = "payrevive"
INCUMBENT_POLICY = "rules"
"""Which rung of the ladder is the submission and which is the thing it has to beat.

Named here rather than hardcoded in the report so that "the proposal" is one fact in
one place. `rules` is the incumbent because it is what a competent payments team
writes by hand — fixed backoff, a message, escalate the large ones — and beating a
strawman proves nothing."""

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260904
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A mean with a bootstrap confidence interval around it."""

    mean: float
    low: float
    high: float
    samples: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the effect is distinguishable from no effect at all. A policy
        whose interval contains zero has not been shown to do anything."""
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        sign = "+" if self.mean >= 0 else ""
        return f"{sign}{self.mean:,.0f} [{self.low:,.0f}, {self.high:,.0f}]"


def paired_interval(
    differences: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Percentile bootstrap over per-seed paired differences.

    Resampling the *differences* rather than the two policies separately is what
    keeps the pairing: each draw keeps a seed's two runs together, so the shared
    randomness cancels inside every resample exactly as it does in the point
    estimate. Sampling the two sides independently would throw that away and widen
    the interval by a large factor for no reason.
    """
    values = np.asarray(list(differences), dtype=float)
    if values.size == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    mean = float(values.mean())
    if values.size == 1:
        # One seed cannot bound its own noise, and pretending otherwise with a
        # zero-width interval is the specific dishonesty this module exists to stop.
        return Interval(mean, float("-inf"), float("inf"), 1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return Interval(mean, float(low), float(high), int(values.size))


def run_batch(scenario: Scenario, seed: int, name: str, build: PolicyBuilder) -> BatchMetrics:
    """One policy over one generated batch. The only place a policy meets the world."""
    env = RecoveryEnv(scenario, seed=seed)
    episodes = env.reset()
    policy = build(env)
    policy.begin(len(episodes))
    for episode in episodes:
        run_episode(env, episode, policy)
    return collect(name, env, episodes)


@dataclass(frozen=True)
class PolicyOnScenario:
    """One policy over one scenario at every seed, with the pairing preserved."""

    policy: str
    scenario: str
    comparisons: tuple[Comparison, ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(c.seed for c in self.comparisons)

    @property
    def batches(self) -> tuple[BatchMetrics, ...]:
        return tuple(c.metrics for c in self.comparisons)

    @property
    def lift(self) -> Interval:
        """Incremental rupees over `do_nothing`, one paired observation per seed."""
        return paired_interval([c.incremental_lift_rupees for c in self.comparisons])

    @property
    def net_lift(self) -> Interval:
        """The same after spend. What a merchant actually keeps."""
        return paired_interval([c.net_lift_rupees for c in self.comparisons])

    @property
    def regret(self) -> Interval | None:
        """Money that was available and not collected. None without an oracle run."""
        values = [c.regret_vs_oracle_rupees for c in self.comparisons]
        if any(v is None for v in values):
            return None
        return paired_interval([float(v) for v in values])

    # ── Totals, for the ratios that should not be averaged ────────────────

    @property
    def total_at_risk_rupees(self) -> float:
        return sum(b.at_risk_rupees for b in self.batches)

    @property
    def total_lift_rupees(self) -> float:
        return sum(c.incremental_lift_rupees for c in self.comparisons)

    @property
    def total_spend_rupees(self) -> float:
        return sum(b.spend_rupees for b in self.batches)

    @property
    def total_achievable_rupees(self) -> float | None:
        values = [c.achievable_lift_rupees for c in self.comparisons]
        return None if any(v is None for v in values) else sum(float(v) for v in values)

    @property
    def total_recovered_rupees(self) -> float:
        return sum(b.recovered_rupees for b in self.batches)

    @property
    def total_system_recovered_rupees(self) -> float:
        return sum(b.system_recovered_rupees for b in self.batches)

    @property
    def total_ambiguous_rupees(self) -> float:
        return sum(b.ambiguous_rupees for b in self.batches)

    @property
    def total_preempted_rupees(self) -> float:
        return sum(c.preempted_rupees for c in self.comparisons)

    @property
    def total_contacts(self) -> int:
        return sum(b.contacts for b in self.batches)

    @property
    def total_retries(self) -> int:
        return sum(b.retries for b in self.batches)

    @property
    def total_actions(self) -> int:
        """Actions the compliance engine was asked to approve, across every batch. The
        denominator for the refusals it would have issued."""
        return sum(b.actions_needing_approval for b in self.batches)

    @property
    def total_escalations(self) -> int:
        return sum(b.escalations for b in self.batches)

    @property
    def total_agent_capacity(self) -> int:
        return sum(b.agent_capacity for b in self.batches)

    @property
    def median_hours_to_recovery(self) -> float:
        """Median of the per-batch medians. Good enough for a speed column, and it
        avoids holding every episode in memory just to re-median them."""
        values = sorted(b.median_hours_to_recovery for b in self.batches if b.median_hours_to_recovery)
        if not values:
            return 0.0
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0

    def merged_by_cause(self) -> dict[str, CauseBreakdown]:
        return merge_causes(b for batch in self.batches for b in batch.by_cause.values())

    @property
    def share_of_achievable_lift(self) -> float | None:
        """Pooled, not the mean of per-seed shares.

        A ratio of means is the right estimator for "how much of the available money
        did this collect across everything we ran". Averaging per-seed percentages
        would weight a quiet seed the same as a heavy one and let a policy raise its
        headline by doing well on small batches.
        """
        achievable = self.total_achievable_rupees
        if achievable is None:
            return None
        return 0.0 if achievable <= 0 else self.total_lift_rupees / achievable

    @property
    def seeds_beating_baseline(self) -> int:
        """How many seeds the policy actually won on. A positive mean built from one
        lucky seed out of ten is not a result, and this is where that shows."""
        return sum(1 for c in self.comparisons if c.incremental_lift_rupees > 0)

    # ── Shippability, pooled across seeds ─────────────────────────────────

    HARD_LIMITS: ClassVar[tuple[str, ...]] = (
        "quiet_hour_contacts",
        "invalid_actions",
        "engine_refused_actions",
        "episodes_at_step_cap",
    )
    """The concerns where zero is attainable, and therefore required.

    `self_inflicted_blocks` is measured alongside these but is not one of them: a
    failed retry kills a working card with fixed probability, so the only policy
    that blocks nothing is one that retries nothing. Gating on it printed the same
    verdict against every policy that acted and told a reader nothing."""

    @property
    def totals_of_concern(self) -> dict[str, int]:
        return {
            "quiet_hour_contacts": sum(b.quiet_hour_contacts for b in self.batches),
            "self_inflicted_blocks": sum(b.self_inflicted_blocks for b in self.batches),
            "invalid_actions": sum(b.invalid_actions for b in self.batches),
            "engine_refused_actions": sum(
                b.engine_refused_actions for b in self.batches
            ),
            "episodes_at_step_cap": sum(b.episodes_at_step_cap for b in self.batches),
        }

    @property
    def total_live_instrument_payments(self) -> int:
        """Payments whose card or mandate was still working when they failed — the
        only ones this policy could possibly have broken."""
        return sum(b.live_instrument_payments for b in self.batches)

    @property
    def self_inflicted_block_rate(self) -> float:
        live = self.total_live_instrument_payments
        return 0.0 if not live else self.totals_of_concern["self_inflicted_blocks"] / live

    @property
    def is_shippable(self) -> bool:
        totals = self.totals_of_concern
        return not any(totals[key] for key in self.HARD_LIMITS) and all(
            c.lift_identity_holds for c in self.comparisons
        )


@dataclass(frozen=True)
class EvalRun:
    """Every policy, every scenario, every seed — the whole table in one object."""

    policies: tuple[str, ...]
    scenarios: tuple[str, ...]
    seeds: tuple[int, ...]
    results: dict[tuple[str, str], PolicyOnScenario] = field(default_factory=dict)

    def get(self, policy: str, scenario: str) -> PolicyOnScenario:
        return self.results[(policy, scenario)]

    def for_policy(self, policy: str) -> list[PolicyOnScenario]:
        return [self.results[(policy, s)] for s in self.scenarios]

    def pooled_lift(self, policy: str) -> Interval:
        """Lift across every (scenario, seed) at once.

        Each pair is still a paired difference against `do_nothing` on that exact
        batch, so pooling is legitimate. It answers a different question from any
        single scenario's number: not "how does this do on festival traffic" but
        "what does this do over a mix of weeks, including the bad ones".
        """
        return paired_interval([
            c.incremental_lift_rupees
            for scenario in self.scenarios
            for c in self.results[(policy, scenario)].comparisons
        ])

    def pooled_share_of_achievable(self, policy: str) -> float | None:
        lift, achievable = 0.0, 0.0
        for scenario in self.scenarios:
            result = self.results[(policy, scenario)]
            available = result.total_achievable_rupees
            if available is None:
                return None
            lift += result.total_lift_rupees
            achievable += available
        return 0.0 if achievable <= 0 else lift / achievable

    def head_to_head(self, policy: str, rival: str) -> Interval:
        """Paired difference against another policy rather than against nothing.

        The comparison that matters once a policy clearly beats doing nothing: the
        incumbent already recovers money, so the only interesting question is whether
        the new thing recovers more of it on the same batches.
        """
        return paired_interval([
            mine.incremental_lift_rupees - theirs.incremental_lift_rupees
            for scenario in self.scenarios
            for mine, theirs in zip(
                self.results[(policy, scenario)].comparisons,
                self.results[(rival, scenario)].comparisons,
                strict=True,
            )
        ])


def evaluate(
    builders: dict[str, PolicyBuilder],
    scenarios: Sequence[str] | None = None,
    seeds: Sequence[int] = tuple(range(1, 21)),
    progress: Callable[[str], None] | None = None,
) -> EvalRun:
    """Run the whole grid and assemble the comparisons.

    The baseline and the ceiling are run first for each (scenario, seed) so that
    every other policy is compared against the same batch it was actually measured
    on. `Comparison` refuses a mismatch, which turns a subtle reporting error — lift
    quoted against a different week's baseline — into an exception.
    """
    if BASELINE_POLICY not in builders:
        raise ValueError(
            f"no '{BASELINE_POLICY}' in the policy set: every figure in this harness is "
            "a difference against doing nothing, so the baseline is not optional"
        )
    names = tuple(builders)
    scenario_names = tuple(scenarios if scenarios is not None else SCENARIOS)
    seed_list = tuple(seeds)

    gathered: dict[tuple[str, str], list[Comparison]] = {
        (policy, scenario): [] for policy in names for scenario in scenario_names
    }

    for scenario in scenario_names:
        spec = SCENARIOS[scenario]
        for seed in seed_list:
            if progress is not None:
                progress(f"{scenario} seed {seed}")
            batches = {
                policy: run_batch(spec, seed, policy, build)
                for policy, build in builders.items()
            }
            baseline = batches[BASELINE_POLICY]
            ceiling = batches.get(CEILING_POLICY)
            for policy, metrics in batches.items():
                gathered[(policy, scenario)].append(compare(metrics, baseline, ceiling))

    return EvalRun(
        policies=names,
        scenarios=scenario_names,
        seeds=seed_list,
        results={
            key: PolicyOnScenario(policy=key[0], scenario=key[1], comparisons=tuple(values))
            for key, values in gathered.items()
        },
    )

