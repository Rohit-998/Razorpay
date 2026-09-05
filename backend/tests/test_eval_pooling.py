"""Pooling across scenarios, which is what the headline number actually is.

The figure the submission leads with is not one scenario's result — it is every batch at
once, five scenarios by twenty seeds, so that a policy cannot post a good headline by
being excellent on quiet weeks. `pooled_lift` has always done that. `net_lift`, `regret`
and the count of batches won existed only per scenario, and the report threw the pooled
versions away after printing them in a markdown table.

That was invisible in the API and total in the product: the analytics page reads the
pooled response, so the one number the brief asks for rendered as an em dash with an
empty section under it.

These tests run a real grid rather than fabricating metrics — ten batches, about two
seconds — because the invariants worth asserting are relationships between the pooled
figure and the per-scenario ones, and a hand-built `BatchMetrics` would let both sides
be wrong together.
"""

from __future__ import annotations

import pytest

from app.eval.harness import evaluate
from app.policies import LADDER

SCENARIOS = ("baseline", "outage_day")
SEEDS = (1, 2, 3)


@pytest.fixture(scope="module")
def run():
    """One grid for the whole module. Two scenarios, because pooling over one scenario is
    not pooling, and the bug being locked down is specifically about crossing them."""
    return evaluate(LADDER, scenarios=SCENARIOS, seeds=SEEDS)


def test_the_pooled_interval_is_drawn_from_every_batch_not_every_seed(run) -> None:
    """A pooled interval over 3 observations instead of 6 would be twice as wide as it
    should be and would still look like a confidence interval."""
    pooled = run.pooled_lift("payrevive")
    assert pooled.samples == len(SCENARIOS) * len(SEEDS)
    assert run.get("payrevive", "baseline").lift.samples == len(SEEDS)


def test_the_pooled_net_lift_covers_the_same_batches_as_the_pooled_lift(run) -> None:
    """The two are rendered under one heading — gross above, net in the subtitle — so they
    have to be the same observations. If one pooled and the other did not, the difference
    between them would stop being spend."""
    assert run.pooled_net_lift("payrevive").samples == run.pooled_lift("payrevive").samples


def test_spend_is_what_separates_the_two(run) -> None:
    """Net is the smaller number, and doing nothing spends nothing.

    A net figure that equalled its gross would mean the cost of the actions had quietly
    stopped being counted, which is the direction that flatters the submission.
    """
    assert run.pooled_net_lift("payrevive").mean < run.pooled_lift("payrevive").mean
    assert run.pooled_net_lift("do_nothing").mean == run.pooled_lift("do_nothing").mean


def test_the_pooled_win_count_is_the_sum_of_the_per_scenario_ones(run) -> None:
    """The count and its denominator, cross-checked against the per-scenario figures the
    report already stored — so a pooled count cannot drift from the rows beneath it."""
    won, ran = run.pooled_seeds_beating_baseline("payrevive")
    per_scenario = [run.get("payrevive", s).seeds_beating_baseline for s in SCENARIOS]

    assert won == sum(per_scenario)
    assert ran == len(SCENARIOS) * len(SEEDS)
    assert won == ran, "the proposal beats doing nothing on every batch it faces"


def test_the_baseline_never_beats_itself(run) -> None:
    """`do_nothing` is the thing lift is measured against, so its own lift is zero on every
    batch — and zero is not a win. A count of 6/6 here would mean the comparison had been
    wired against something else."""
    assert run.pooled_seeds_beating_baseline("do_nothing") == (0, len(SCENARIOS) * len(SEEDS))


def test_regret_is_withheld_when_any_batch_ran_without_a_ceiling() -> None:
    """Regret needs the oracle on the identical batch. Computed over only the batches that
    had one, it would be a smaller denominator than the lift printed beside it — and the
    page would show a gap-to-ceiling for a ceiling that was never run.

    `None` is what the analytics page renders as absent, which is the honest output.
    """
    without_ceiling = evaluate(
        {name: LADDER[name] for name in ("do_nothing", "payrevive")},
        scenarios=("baseline",),
        seeds=(1, 2),
    )
    assert without_ceiling.pooled_regret("payrevive") is None
    assert without_ceiling.pooled_share_of_achievable("payrevive") is None
    # The lift itself survives: it only needs the baseline, which did run.
    assert without_ceiling.pooled_lift("payrevive").mean > 0


def test_the_gap_to_the_ceiling_closes_as_the_policy_improves(run) -> None:
    """Regret is money that was available and not collected, so it has to rank the ladder
    in reverse. This is the assertion that would catch a regret figure computed against
    the wrong policy's batches."""
    regrets = {
        name: run.pooled_regret(name).mean
        for name in ("do_nothing", "naive_retry", "rules", "payrevive", "oracle")
    }
    assert regrets["do_nothing"] > regrets["naive_retry"] > regrets["rules"]
    assert regrets["rules"] > regrets["payrevive"] > regrets["oracle"]
    assert regrets["oracle"] == pytest.approx(0.0), "the ceiling leaves nothing on the table"
