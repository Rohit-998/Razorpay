"""Properties the policy ladder must hold, or the comparison table is decoration.

`test_sim.py` guards the environment. This file guards the four policies that are
scored against it, and in particular the two claims that make the headline number
mean anything:

  * the oracle really is an upper bound — a ceiling that any hand-written policy
    can beat is worse than no ceiling at all, because it makes every other figure
    in the report look impossibly good
  * the scarce agent bench really binds — without it the optimal policy is "phone
    everybody", which is not a policy and not deployable

Both were bugs first and tests second. The oracle lost to `rules` on the stress
batch until it learned to escalate, and escalation was a free upgrade until the
bench became finite.
"""

from __future__ import annotations

import pytest

from app.policies import DoNothingPolicy, NaiveRetryPolicy, OraclePolicy, RulesPolicy, run_episode
from app.policies.base import MAX_STEPS_PER_EPISODE
from app.policies.rules import MAX_CONTACTS
from app.sim.environment import ESCALATION_COST_PAISE, RecoveryEnv
from app.sim.scenarios import SCENARIOS
from app.sim.types import Action, ActionType, AttributionTruth

SCENARIO_NAMES = list(SCENARIOS)
SEED = 1

BUILDERS = {
    "do_nothing": lambda env: DoNothingPolicy(),
    "naive_retry": lambda env: NaiveRetryPolicy(),
    "rules": lambda env: RulesPolicy(),
    "oracle": lambda env: OraclePolicy(env),
}
CONTENDERS = ["do_nothing", "naive_retry", "rules"]


class EscalateEverything(DoNothingPolicy):
    """The strategy an unconstrained environment would reward. Kept as a probe.

    If this ever comes close to `rules`, the economics are broken: it means the
    agent bench is cheap enough that judgement is worthless, and there is nothing
    left for a learned policy to learn.
    """

    name = "escalate_all"

    def act(self, obs):
        if not obs.escalated and obs.agent_calls_remaining > 0:
            return Action(ActionType.ESCALATE, reason="probe: escalate unconditionally")
        return Action(ActionType.GIVE_UP, reason="probe: nothing left to escalate with")


class BlindEscalator(DoNothingPolicy):
    """Probe: asks for an agent without checking whether one is free.

    A policy that ignores a capacity signal is not deployable, and the environment
    has to say so rather than quietly absorbing the request.
    """

    name = "blind_escalator"

    def act(self, obs):
        if not obs.escalated:
            return Action(ActionType.ESCALATE, reason="probe: ignore the bench entirely")
        return Action(ActionType.GIVE_UP, reason="probe done")


def _run(scenario: str, build, seed: int = SEED):
    """Run one policy over a whole batch and hand back the closed episodes."""
    env = RecoveryEnv(SCENARIOS[scenario], seed=seed)
    episodes = env.reset()
    policy = build(env)
    policy.begin(len(episodes))
    for episode in episodes:
        run_episode(env, episode, policy)
    return env, episodes


def _recovered_rupees(episodes) -> float:
    return sum(e.amount_rupees for e in episodes if e.paid)


def _lift(scenario: str, build, baseline: float | None = None) -> float:
    """Incremental rupees over `do_nothing` on the identical batch."""
    if baseline is None:
        baseline = _recovered_rupees(_run(scenario, BUILDERS["do_nothing"])[1])
    return _recovered_rupees(_run(scenario, build)[1]) - baseline


# ── The ceiling has to be a ceiling ──────────────────────────────────────


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_oracle_is_an_upper_bound_on_every_other_policy(scenario: str) -> None:
    """No policy may out-recover full knowledge of the latent state.

    This failed when the oracle had no escalation branch: `rules` escalated on
    high-value payments, the oracle never did, and the "ceiling" came in ₹4 lakh
    below a hand-written rule on the stress batch. Any future action added to the
    action space has to be added here too, or this test will say so.
    """
    baseline = _recovered_rupees(_run(scenario, BUILDERS["do_nothing"])[1])
    ceiling = _lift(scenario, BUILDERS["oracle"], baseline)
    for name in CONTENDERS:
        lift = _lift(scenario, BUILDERS[name], baseline)
        assert lift <= ceiling, (
            f"{name} recovered +₹{lift:,.0f} on {scenario} against an oracle ceiling of "
            f"+₹{ceiling:,.0f} — the upper bound is not an upper bound"
        )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_oracle_never_claims_an_ambiguous_recovery(scenario: str) -> None:
    """Perfect play never messages someone who was already coming back.

    So it never produces a recovery it cannot attribute. Any policy showing a
    healthy recovery count alongside a pile of AMBIGUOUS verdicts is standing next
    to other people's money, and this is the contrast that proves the point.
    """
    _, episodes = _run(scenario, BUILDERS["oracle"])
    ambiguous = [e for e in episodes if e.attribution is AttributionTruth.AMBIGUOUS]
    assert not ambiguous, f"oracle produced {len(ambiguous)} unattributable recoveries"


# ── The scarce agent bench ───────────────────────────────────────────────


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
@pytest.mark.parametrize("name", [*BUILDERS, "escalate_all"])
def test_no_policy_can_exceed_the_agent_bench(scenario: str, name: str) -> None:
    """A policy cannot escalate more payments than the merchant has agents for."""
    build = BUILDERS.get(name, lambda env: EscalateEverything())
    env, episodes = _run(scenario, build)
    escalated = sum(1 for e in episodes if e.escalated)
    assert escalated <= env.agent_capacity
    assert env.agent_calls_used == escalated


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_escalating_everything_is_a_bad_strategy(scenario: str) -> None:
    """The naive use of the strongest action must lose to judgement about it.

    `escalate_all` spends the whole bench on whichever payments happen to fail
    first. If that ever rivals `rules`, escalation has stopped being a decision.
    """
    baseline = _recovered_rupees(_run(scenario, BUILDERS["do_nothing"])[1])
    naive = _lift(scenario, lambda env: EscalateEverything(), baseline)
    considered = _lift(scenario, BUILDERS["rules"], baseline)
    assert naive < considered, (
        f"on {scenario} escalating blindly returned +₹{naive:,.0f} against "
        f"+₹{considered:,.0f} for a policy that chooses — the bench is too cheap"
    )


def test_an_agent_call_is_charged_whether_or_not_anyone_answers() -> None:
    """The cost is the attempt, not the outcome. A policy cannot escalate for free
    into an unanswered phone and pay nothing for it."""
    env, episodes = _run("baseline", lambda env: EscalateEverything())
    accepted = sum(1 for e in episodes if e.escalated)
    connected = sum(
        1 for e in episodes
        for step in e.history
        if step.action.type is ActionType.ESCALATE and "could not reach" not in step.detail
    )
    total = sum(e.cost_paise for e in episodes)
    assert accepted > 0 and connected < accepted, "expected some calls to go unanswered"
    assert total == accepted * ESCALATION_COST_PAISE


def test_escalating_into_a_full_bench_is_refused_and_costs_nothing() -> None:
    """The refusal is visible to the policy as an invalid action, not swallowed."""
    env, episodes = _run("baseline", lambda env: BlindEscalator())
    refusals = [
        step for e in episodes for step in e.history
        if step.action.type is ActionType.ESCALATE and "bench is full" in step.detail
    ]
    assert refusals, "the bench never filled, so this scenario proves nothing"
    assert all(step.invalid and step.cost_paise == 0 for step in refusals)
    assert sum(1 for e in episodes if e.escalated) == env.agent_capacity


# ── Stated budgets have to be the real budgets ────────────────────────────


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_rules_respects_the_contact_budget_it_documents(scenario: str) -> None:
    """Two messages per payment, as advertised.

    It once sent more than four, because one branch fell through to `send` with no
    budget check and the exhaustion test required *both* budgets spent — which
    never happened for a customer with no mandate. The lift that bought was real
    money in the report and entirely an accident.
    """
    _, episodes = _run(scenario, BUILDERS["rules"])
    worst = max(e.contacts for e in episodes)
    assert worst <= MAX_CONTACTS, f"rules sent {worst} messages to one payment"


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_incumbent_actually_reaches_its_message_step(scenario: str) -> None:
    """`naive_retry` fires three retries and then one SMS. It has to get to the SMS.

    Reading the environment's accepted-attempt counter instead of its own meant a
    mandate-less payment looped on a rejected retry until the step cap, so the
    incumbent never messaged anyone and its lift was understated by half. A real
    retry loop counts the attempt it made, not the attempt the gateway allowed.
    """
    _, episodes = _run(scenario, BUILDERS["naive_retry"])
    contacted = sum(1 for e in episodes if e.contacts > 0)
    assert contacted > 0.25 * len(episodes), (
        f"only {contacted} of {len(episodes)} payments ever got the incumbent's message"
    )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
@pytest.mark.parametrize("name", list(BUILDERS))
def test_no_policy_runs_out_of_steps(scenario: str, name: str) -> None:
    """Hitting the step cap means a policy failed to terminate, which is a bug in
    the policy. The harness counts it; this test refuses to let one ship."""
    _, episodes = _run(scenario, BUILDERS[name])
    stuck = [e for e in episodes if len(e.history) >= MAX_STEPS_PER_EPISODE]
    assert not stuck, f"{name} hit the {MAX_STEPS_PER_EPISODE}-step cap on {len(stuck)} payments"


@pytest.mark.parametrize("name", list(BUILDERS))
def test_every_policy_is_reproducible(name: str) -> None:
    """Same scenario, same seed, same policy — identical money, cost and verdicts."""
    first = [
        (e.payment_id, e.paid, e.cost_paise, e.attribution, e.contacts, e.escalated)
        for e in _run("baseline", BUILDERS[name])[1]
    ]
    second = [
        (e.payment_id, e.paid, e.cost_paise, e.attribution, e.contacts, e.escalated)
        for e in _run("baseline", BUILDERS[name])[1]
    ]
    assert first == second


