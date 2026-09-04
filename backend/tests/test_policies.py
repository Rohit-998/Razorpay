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

from app.eval.metrics import engine_refusals
from app.policies import (
    DoNothingPolicy,
    NaiveRetryPolicy,
    OraclePolicy,
    PayRevivePolicy,
    RulesPolicy,
    run_episode,
)
from app.policies.base import MAX_STEPS_PER_EPISODE
from app.policies.payrevive import (
    MAX_CONTACTS_PER_DAY,
    MAX_RETRIES_PER_PAYMENT,
    MIN_ACTION_VALUE_PAISE,
    MIN_RETRY_INTERVAL_MINUTES,
    UPI_TRANSACTION_CEILING_PAISE,
)
from app.policies.rules import MAX_CONTACTS
from app.sim.environment import ESCALATION_COST_PAISE, RecoveryEnv
from app.sim.scenarios import SCENARIOS
from app.sim.types import Action, ActionType, AttributionTruth

SCENARIO_NAMES = list(SCENARIOS)
SEED = 1
PAIRED_SEEDS = (1, 2, 3)
"""Seeds for head-to-head comparisons. More than one, because a batch is a draw and
the difference between two policies on a single batch is mostly noise."""

BUILDERS = {
    "do_nothing": lambda env: DoNothingPolicy(),
    "naive_retry": lambda env: NaiveRetryPolicy(),
    "rules": lambda env: RulesPolicy(),
    "payrevive": lambda env: PayRevivePolicy(env),
    "oracle": lambda env: OraclePolicy(env),
}
CONTENDERS = ["do_nothing", "naive_retry", "rules", "payrevive"]


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


def _lift(
    scenario: str, build, baseline: float | None = None, seed: int = SEED
) -> float:
    """Incremental rupees over `do_nothing` on the identical batch."""
    if baseline is None:
        baseline = _recovered_rupees(_run(scenario, BUILDERS["do_nothing"], seed)[1])
    return _recovered_rupees(_run(scenario, build, seed)[1]) - baseline


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


# ── `payrevive`, the policy this project proposes ─────────────────────────
#
# The claim being made about it is not "it recovers more money" on its own. It is
# "it recovers more money *and* would survive a payments team's review", and the
# second half is the part a lift figure cannot show. These tests are that half.


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_payrevive_beats_the_hand_written_rules(scenario: str) -> None:
    """The whole reason to prefer a fitted policy over a good rule.

    Asserted per scenario rather than pooled, because a policy that wins on average
    by collapsing on the stress batch has not beaten the rule — it has found a
    different set of blind spots.

    Asserted on the mean paired difference over several seeds rather than one, because
    one batch is one draw. The single-seed version of this test failed on `baseline`
    at seed 1 while `payrevive` was ahead on the other seven, which says nothing about
    the policy and everything about asserting on a sample of one — `app.eval` reports
    bootstrap intervals over paired differences for exactly this reason, and a test
    that claims less rigour than the report is not guarding the report's claim.
    """
    margins = []
    for seed in PAIRED_SEEDS:
        baseline = _recovered_rupees(_run(scenario, BUILDERS["do_nothing"], seed)[1])
        rules = _lift(scenario, BUILDERS["rules"], baseline, seed)
        payrevive = _lift(scenario, BUILDERS["payrevive"], baseline, seed)
        margins.append(payrevive - rules)
    mean = sum(margins) / len(margins)
    assert mean > 0, (
        f"payrevive beat `rules` by ₹{mean:,.0f} a batch on {scenario} across "
        f"{len(PAIRED_SEEDS)} seeds ({', '.join(f'{m:+,.0f}' for m in margins)}) — "
        f"the fitted model is not earning its complexity"
    )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_payrevive_never_opens_with_the_terminal_action(scenario: str) -> None:
    """An agent call closes the episode, so it may only be opened with when the
    episode has nothing else in it.

    This was a real loss. Escalation was priced as a peer of a retry, so the batch's
    largest failures were phoned at minute zero — and since only 60% of calls connect,
    a miss ended a payment that `rules` went on to recover with one WhatsApp link. The
    fix is not to escalate less but to escalate later: `payrevive` still ranks the whole
    batch to decide *who* gets one of the scarce slots, and now spends the slot only
    once nothing cheaper clears the bar.

    The assertion used to be a blanket ban, and a blanket ban was wrong. Three payments
    in the grid open with a call and are right to: ₹1.18–1.81 L failures on instruments
    the fitted model prices at under ₹5 of expected recovery across every legal route it
    holds. Foreclosing routes worth nothing costs nothing, and a human is the only thing
    left that might work. So the invariant is the gate itself rather than its usual
    consequence — escalation is never first *while a cheaper route is worth taking* —
    which is what the original bug violated and a count of first-actions only detected
    by accident.

    It reaches into `_priced` deliberately. The property is about what the policy knew
    when it decided, and that is not recoverable from the finished episode: by the time
    the history exists, the observation that justified step 0 is gone. Re-deriving it
    means re-observing the same batch before anything has happened to it.
    """
    _, episodes = _run(scenario, BUILDERS["payrevive"])
    opened_with = [
        e.payment_id
        for e in episodes
        if e.history and e.history[0].action.type is ActionType.ESCALATE
    ]
    escalated = sum(1 for e in episodes if e.escalated)
    assert escalated > 0, "no payment was escalated, so this proves nothing"

    # Same scenario, same seed, so the same batch — observed at reset, which is the
    # state every step 0 above was decided from.
    env = RecoveryEnv(SCENARIOS[scenario], seed=SEED)
    fresh = {e.payment_id: e for e in env.reset()}
    policy = PayRevivePolicy(env)
    policy.begin(len(fresh))
    foreclosed = []
    for payment_id in opened_with:
        obs = env.observe(fresh[payment_id])
        cheaper = [
            c for c in policy._priced(obs) if c.action.type is not ActionType.ESCALATE
        ]
        best = max(cheaper, key=lambda c: c.value, default=None)
        if best is not None and best.value > MIN_ACTION_VALUE_PAISE:
            foreclosed.append(
                f"{payment_id} (₹{fresh[payment_id].amount_rupees:,.0f}) had "
                f"{best.action.type.value} worth ₹{best.value / 100:,.0f}"
            )
    assert not foreclosed, (
        f"payrevive spent an agent slot on {len(foreclosed)} of the {len(opened_with)} "
        f"payments it opened with a call on {scenario} while something cheaper was "
        f"worth trying, ending the episode to do it: {foreclosed[:3]}"
    )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_payrevive_holds_an_overnight_failure_instead_of_abandoning_it(
    scenario: str,
) -> None:
    """A payment that fails at 23:00 is worth waiting for, not writing off.

    Quiet hours mean no message may go out, and for a customer with no mandate that
    leaves nothing legal to do at all — so the valuation of *waiting* has to be able
    to see that a message becomes legal at 08:00. It could not: the future was priced
    with the present's contact ban applied, every route came back worth nothing, and
    the policy gave up overnight on payments `rules` collected by simply sleeping.

    The assertion is economic rather than mechanical, because dropping an overnight
    payment is sometimes right — a ₹50 order is not worth a night's patience. What
    cannot happen is dropping a large one untried.
    """
    _, episodes = _run(scenario, BUILDERS["payrevive"])
    overnight = [e for e in episodes if e.failed_at.hour >= 22 or e.failed_at.hour < 8]
    assert overnight, f"no payment failed inside quiet hours on {scenario}"
    dropped = [
        e.amount_rupees
        for e in overnight
        if len(e.history) == 1 and e.history[0].action.type is ActionType.GIVE_UP
    ]
    assert max(dropped, default=0.0) < 10_000, (
        f"payrevive abandoned a ₹{max(dropped):,.0f} overnight failure on {scenario} "
        f"without trying anything, out of {len(dropped)} it dropped"
    )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_payrevive_proposes_nothing_the_gateway_would_refuse(scenario: str) -> None:
    """`naive_retry` proposes 83,000 impossible actions across the grid — retries with
    no mandate on file, rails the customer does not hold. Each one is a real API call
    a merchant would be rate-limited for. This policy checks first."""
    _, episodes = _run(scenario, BUILDERS["payrevive"])
    refused = [s for e in episodes for s in e.history if s.invalid]
    assert not refused, (
        f"payrevive proposed {len(refused)} invalid actions on {scenario}, "
        f"first: {refused[0].detail}"
    )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_payrevive_never_messages_inside_quiet_hours(scenario: str) -> None:
    """22:00–08:00 IST, judged on when the message was sent.

    A hard rule, not a priced one: the fitted model says an overnight message is read
    at 0.22× the daytime rate, which on a large enough payment still clears the bar.
    A compliance limit a good enough expected value can buy its way past is not a
    limit, so the arithmetic never gets to see this decision.
    """
    _, episodes = _run(scenario, BUILDERS["payrevive"])
    sent_at_night = [
        step.taken_at
        for e in episodes
        for step in e.history
        if step.action.type is ActionType.SEND_LINK
        and not step.invalid
        and (step.taken_at.hour >= 22 or step.taken_at.hour < 8)
    ]
    assert not sent_at_night, (
        f"payrevive sent {len(sent_at_night)} messages in quiet hours on {scenario}, "
        f"first at {sent_at_night[0]:%H:%M}"
    )


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_payrevive_honours_the_limits_its_own_compliance_engine_enforces(
    scenario: str,
) -> None:
    """Replay every action through the deployed engine and require it to approve.

    Not a re-implementation of the rules — the actual `compliance.evaluate` the API
    calls on live traffic, fed the counters the engine derives from Redis. If the
    policy is measured while taking actions production would refuse at the door, the
    report is quoting money that cannot be collected, which is worse than quoting no
    number at all.

    It has caught four real breaches, none of which any aggregate would have shown:
    18% of retries inside the gateway's minimum interval, sixteen messages over the
    daily contact cap from a budget that was cumulative rather than daily, 235 agent
    calls to customers already messaged twice that day, and UPI collect requests above
    the ₹1 lakh the network will carry.

    The replay itself lives in `app.eval.metrics.engine_refusals`, because the report
    publishes the same count for every policy on the ladder and a test asserting on a
    privately re-derived number would eventually disagree with the table it is meant to
    be guarding.
    """
    _, episodes = _run(scenario, BUILDERS["payrevive"])
    refused = engine_refusals(episodes)
    assert not refused, (
        f"the deployed compliance engine refuses {len(refused)} of payrevive's "
        f"actions on {scenario}, first: {refused[0]}"
    )


def test_payrevives_limits_match_the_deployed_compliance_engine() -> None:
    """The two copies of these numbers must not drift.

    `payrevive` cannot import `app.config` — the simulator has to run without cloud
    settings loaded — so the limits are declared twice. This is the test that keeps
    the duplicate honest, and it is the reason the duplicate is acceptable.
    """
    from app.config import get_settings

    settings = get_settings()
    assert MAX_RETRIES_PER_PAYMENT == settings.max_retries_per_payment
    assert MAX_CONTACTS_PER_DAY == settings.max_contacts_per_day
    assert MIN_RETRY_INTERVAL_MINUTES == settings.min_retry_interval_minutes
    assert UPI_TRANSACTION_CEILING_PAISE == settings.upi_transaction_ceiling_paise


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_payrevive_explains_every_action_it_takes(scenario: str) -> None:
    """The audit trail the brief asks for, enforced rather than hoped for.

    Every action carries the number that justified it — the fitted rate, the risk it
    was weighed against, and the expected rupees. A decision log that says `RETRY`
    and nothing else cannot be argued with after the fact.
    """
    _, episodes = _run(scenario, BUILDERS["payrevive"])
    unexplained = [
        step.action.type.value
        for e in episodes
        for step in e.history
        if not step.action.reason.strip()
    ]
    assert not unexplained, f"{len(unexplained)} unexplained actions: {unexplained[:5]}"


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_payrevive_spends_the_agent_bench_on_the_batchs_biggest_payments(
    scenario: str,
) -> None:
    """The structural edge over any per-payment rule.

    `rules` gives a slot to the first payment over ₹25,000 that asks, so an ordinary
    morning can spend the week's calls before the large failures arrive. This ranks
    the batch first. The test is that escalated payments are materially larger than
    the ones passed over — not merely different.
    """
    env, episodes = _run(scenario, BUILDERS["payrevive"])
    called = [e.amount_rupees for e in episodes if e.escalated]
    skipped = [e.amount_rupees for e in episodes if not e.escalated]
    assert called, f"payrevive used none of the {env.agent_capacity} agent calls"
    assert sum(called) / len(called) > 2 * (sum(skipped) / len(skipped)), (
        f"escalated payments average ₹{sum(called) / len(called):,.0f} against "
        f"₹{sum(skipped) / len(skipped):,.0f} passed over — the bench is not being "
        "aimed at the money"
    )


def test_payrevive_reads_no_latent_state() -> None:
    """The claim that makes its lift figure admissible.

    `payrevive` is handed the whole `RecoveryEnv` so it can rank the batch, which puts
    every hidden field one attribute access away: the true cause, the customer object
    with its funds-arrival and credential-fix times, the outage schedule, the RNG
    streams. Nothing but discipline stops it reading them, and discipline is not a
    thing a report can cite.

    So each latent field is replaced with a descriptor that raises if the *caller* is
    `app.policies.payrevive`, and permits the environment's own `observe()` — which
    has to read them, that being its job. Then a full batch runs. The assertion is
    structural rather than statistical: not "its numbers look plausible" but "this
    line of code never executed".
    """
    import sys

    from app.sim.environment import Episode

    latent = (
        "true_cause",
        "customer",
        "merchant_broken_until",
        "instrument_dead_at_failure",
        "instrument_blocked",
        "window_hours",
        "rng_retry",
        "rng_click",
        "rng_pay",
        "rng_agent",
    )
    for field in latent:
        assert field in Episode.__dataclass_fields__, f"`{field}` is not an Episode field"

    class Tripwire:
        """Readable by anything except the policy under test."""

        def __init__(self, field: str) -> None:
            self.field = field
            self.store = f"_tripwired_{field}"

        def __get__(self, obj, kind=None):
            if obj is None:
                return self
            caller = sys._getframe(1).f_globals.get("__name__", "")
            if caller == "app.policies.payrevive":
                raise AssertionError(
                    f"payrevive read the latent field `{self.field}` — its lift is "
                    "not measured on observables only"
                )
            return getattr(obj, self.store)

        def __set__(self, obj, value) -> None:
            object.__setattr__(obj, self.store, value)

    env = RecoveryEnv(SCENARIOS["baseline"], seed=SEED)
    episodes = env.reset()
    originals = {f: getattr(episodes[0], f) for f in latent}
    del originals

    try:
        for field in latent:
            values = [getattr(e, field) for e in episodes]
            setattr(Episode, field, Tripwire(field))
            for episode, value in zip(episodes, values):
                setattr(episode, field, value)
        policy = PayRevivePolicy(env)
        policy.begin(len(episodes))
        for episode in episodes:
            run_episode(env, episode, policy)
    finally:
        for field in latent:
            if isinstance(Episode.__dict__.get(field), Tripwire):
                delattr(Episode, field)

    assert any(e.paid for e in episodes), "the tripwired batch recovered nothing at all"


