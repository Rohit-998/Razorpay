"""Properties the simulator must hold, or the measured numbers mean nothing.

These are not smoke tests. Each one guards a specific claim the project makes:

  * determinism and common random numbers → policy comparisons are trustworthy
  * the counterfactual subset invariant  → `do_nothing` is a valid baseline
  * no outcome depends on the label      → the environment cannot be gamed
  * permanent means permanent            → the stopping rule has something to find
  * the emission model does not leak     → the classifier's accuracy is real
"""

from __future__ import annotations

import pytest

from app.models.schemas import PaymentMethod
from app.sim import emission as em
from app.sim.environment import RecoveryEnv
from app.sim.scenarios import SCENARIOS
from app.sim.types import Action, ActionType, AttributionTruth, Channel, Terminal, Tone

SCENARIO_NAMES = list(SCENARIOS)


def _run(scenario_name: str, seed: int, policy) -> list:
    env = RecoveryEnv(SCENARIOS[scenario_name], seed=seed)
    episodes = env.reset()
    for episode in episodes:
        policy(env, episode)
        env.finalize(episode)
    return episodes


def _do_nothing(env: RecoveryEnv, episode) -> None:
    """The baseline. Takes no action at all."""


def _hammer(env: RecoveryEnv, episode) -> None:
    """Three retries and nothing else. The naive production loop."""
    for _ in range(3):
        env.step(episode, Action(ActionType.RETRY, method=episode.method))
        env.step(episode, Action(ActionType.WAIT, wait_minutes=30))


def _mixed(env: RecoveryEnv, episode) -> None:
    """Exercises every branch of the action space."""
    env.step(episode, Action(ActionType.WAIT, wait_minutes=45))
    env.step(episode, Action(ActionType.RETRY, method=episode.method))
    env.step(
        episode,
        Action(
            ActionType.SEND_LINK,
            channel=Channel.WHATSAPP,
            tone=Tone.HINGLISH,
            method=episode.method,
        ),
    )
    env.step(episode, Action(ActionType.ESCALATE))
    env.step(episode, Action(ActionType.GIVE_UP, reason="exhausted"))


POLICIES = {"do_nothing": _do_nothing, "hammer": _hammer, "mixed": _mixed}


# ── Reproducibility ──────────────────────────────────────────────────────


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_batch_generation_is_deterministic(scenario: str) -> None:
    """Same (scenario, seed) → byte-identical batch. Without this nothing else
    in the evaluation is comparable."""
    first = RecoveryEnv(SCENARIOS[scenario], seed=11).reset()
    second = RecoveryEnv(SCENARIOS[scenario], seed=11).reset()
    assert [
        (e.payment_id, e.amount, e.method, e.bank, e.failed_at, e.true_cause) for e in first
    ] == [
        (e.payment_id, e.amount, e.method, e.bank, e.failed_at, e.true_cause) for e in second
    ]


@pytest.mark.parametrize("name", list(POLICIES))
def test_policy_outcomes_are_deterministic(name: str) -> None:
    """Re-running a policy on one seed reproduces every outcome exactly."""
    policy = POLICIES[name]
    first = _run("baseline", 5, policy)
    second = _run("baseline", 5, policy)
    assert [(e.paid, e.paid_at, e.cost_paise, e.attribution, e.terminal) for e in first] == [
        (e.paid, e.paid_at, e.cost_paise, e.attribution, e.terminal) for e in second
    ]


def test_different_seeds_give_different_batches() -> None:
    """Guards against a seeding bug that would silently collapse every seed onto
    one batch and make confidence intervals meaningless."""
    a = RecoveryEnv(SCENARIOS["baseline"], seed=1).reset()
    b = RecoveryEnv(SCENARIOS["baseline"], seed=2).reset()
    assert [e.payment_id for e in a] != [e.payment_id for e in b]


# ── The counterfactual ───────────────────────────────────────────────────


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
@pytest.mark.parametrize("name", ["hammer", "mixed"])
def test_no_policy_can_lose_a_baseline_recovery(scenario: str, name: str) -> None:
    """Every payment `do_nothing` recovers must still be recovered under any
    policy, on the same seed.

    This is the invariant that makes `policy − do_nothing` a defensible reading
    of "money recovered". If an action could suppress an unprompted payment, the
    baseline would not be a lower bound and the lift figure would be arithmetic
    on two unrelated quantities.
    """
    baseline = {e.payment_id for e in _run(scenario, 4, _do_nothing) if e.paid}
    treated = {e.payment_id for e in _run(scenario, 4, POLICIES[name]) if e.paid}
    assert baseline <= treated, f"{len(baseline - treated)} baseline recoveries were lost"


def test_do_nothing_recovers_exactly_the_in_window_self_recoveries() -> None:
    """The baseline is not a guess — it is the set of customers whose
    action-independent `self_recover_at` lands inside the recovery window."""
    for episode in _run("baseline", 9, _do_nothing):
        due = episode.customer.self_recover_at
        expected = due is not None and due <= episode.deadline
        assert episode.paid is expected
        assert episode.cost_paise == 0


def test_giving_up_does_not_forfeit_money_that_was_arriving_anyway() -> None:
    """A policy that stops spending still books the customer's own payment, as
    `CUSTOMER_SELF_RECOVERED`. Otherwise the stopping rule would be penalised for
    revenue it never had any influence over."""

    def give_up_immediately(env: RecoveryEnv, episode) -> None:
        env.step(episode, Action(ActionType.GIVE_UP, reason="not worth it"))

    episodes = _run("baseline", 6, give_up_immediately)
    recovered = [e for e in episodes if e.paid]
    assert recovered, "expected some customers to pay after we gave up"
    assert all(e.attribution is AttributionTruth.CUSTOMER_SELF_RECOVERED for e in recovered)
    assert {e.payment_id for e in _run("baseline", 6, _do_nothing) if e.paid} == {
        e.payment_id for e in recovered
    }


# ── Causal integrity: an outcome is never a function of a label ───────────


@pytest.mark.parametrize("name", list(POLICIES))
def test_outcomes_ignore_the_root_cause_label(name: str) -> None:
    """Scramble every `true_cause` to a deliberately wrong value and nothing about
    the outcomes changes.

    This is the test that separates a causal simulator from a scripted one. The
    label is an upstream latent that *caused* the physical state — funds absent,
    card dead, bank down — and the physical state is what resolves an action. If
    any handler secretly consulted the label, this test fails.
    """
    policy = POLICIES[name]

    honest = _run("baseline", 8, policy)

    env = RecoveryEnv(SCENARIOS["baseline"], seed=8)
    scrambled = env.reset()
    for episode in scrambled:
        episode.true_cause = "TOTALLY_WRONG_LABEL"
        policy(env, episode)
        env.finalize(episode)

    assert [(e.paid, e.paid_at, e.cost_paise, e.attribution, e.terminal) for e in honest] == [
        (e.paid, e.paid_at, e.cost_paise, e.attribution, e.terminal) for e in scrambled
    ]


def test_a_dead_instrument_never_authorises_on_a_retry() -> None:
    """`PERMANENT_DECLINE` means the instrument is gone. No number of retries can
    recover it, which is the whole reason a stopping rule is worth having."""
    env = RecoveryEnv(SCENARIOS["stress_dead_instruments"], seed=2)
    checked = 0
    for episode in env.reset():
        if episode.true_cause != "PERMANENT_DECLINE":
            continue
        checked += 1
        for _ in range(4):
            env.step(episode, Action(ActionType.RETRY, method=episode.method))
            env.step(episode, Action(ActionType.WAIT, wait_minutes=60))
        assert episode.attribution is not AttributionTruth.SYSTEM_RECOVERED
    assert checked > 20, "expected a substantial dead-instrument population"


def test_a_dead_card_can_still_be_escaped_by_switching_method() -> None:
    """The dead instrument is the instrument, not the customer. A different rail
    is a genuine recovery route — so `GIVE_UP` is not automatically correct."""
    env = RecoveryEnv(SCENARIOS["stress_dead_instruments"], seed=2)
    recovered = 0
    for episode in env.reset():
        if episode.true_cause != "PERMANENT_DECLINE":
            continue
        alternatives = [m for m in episode.customer.alt_methods if m is not episode.method]
        if not alternatives:
            continue
        env.step(
            episode,
            Action(
                ActionType.SEND_LINK,
                channel=Channel.WHATSAPP,
                method=alternatives[0],
                tone=Tone.FRIENDLY,
            ),
        )
        env.finalize(episode)
        recovered += episode.attribution is AttributionTruth.SYSTEM_RECOVERED
    assert recovered > 0, "switching method should sometimes work on a dead card"


# ── Honest attribution ───────────────────────────────────────────────────


def test_a_recent_contact_makes_a_self_recovery_ambiguous_not_a_win() -> None:
    """Contact someone, have them pay through their own channel an hour later, and
    the honest verdict is `AMBIGUOUS`. Counting that as a win is how recovery
    dashboards end up reporting other people's work."""

    def blanket_sms(env: RecoveryEnv, episode) -> None:
        env.step(episode, Action(ActionType.SEND_LINK, channel=Channel.SMS, method=episode.method))

    episodes = _run("baseline", 12, blanket_sms)
    ambiguous = [e for e in episodes if e.attribution is AttributionTruth.AMBIGUOUS]
    assert ambiguous, "blanket outreach should produce unprovable cases"
    for episode in ambiguous:
        assert episode.paid and episode.last_contact_at is not None
        hours = (episode.paid_at - episode.last_contact_at).total_seconds() / 3600.0
        assert 0 <= hours <= 6.0


def test_every_recovery_carries_exactly_one_verdict() -> None:
    """No episode may be paid without an attribution, or unpaid with one."""
    for name in POLICIES:
        for episode in _run("baseline", 13, POLICIES[name]):
            if episode.paid:
                assert episode.attribution is not AttributionTruth.NOT_RECOVERED
                assert episode.terminal is Terminal.RECOVERED
                assert episode.paid_at is not None and episode.paid_at <= episode.deadline
            else:
                assert episode.attribution is AttributionTruth.NOT_RECOVERED
                assert episode.terminal in (Terminal.ABANDONED, Terminal.ESCALATED)


# ── Spend accounting ─────────────────────────────────────────────────────


def test_cost_equals_the_sum_of_what_was_actually_spent() -> None:
    """Cost per recovered rupee is only meaningful if the numerator is exact."""
    for episode in _run("festival_spike", 3, _mixed):
        assert episode.cost_paise == sum(step.cost_paise for step in episode.history)


def test_actions_after_the_episode_closes_cost_nothing() -> None:
    """A policy cannot keep spending on a payment that is already settled."""
    env = RecoveryEnv(SCENARIOS["baseline"], seed=14)
    for episode in env.reset():
        env.step(episode, Action(ActionType.GIVE_UP, reason="stop"))
        before = episode.cost_paise
        for _ in range(3):
            result = env.step(
                episode, Action(ActionType.SEND_LINK, channel=Channel.WHATSAPP)
            )
            assert result.invalid and result.cost_paise == 0
        assert episode.cost_paise == before


def test_a_retry_without_a_mandate_is_rejected_as_impossible() -> None:
    """Structurally illegal actions are counted, not silently absorbed. A policy
    that proposes them is not production-ready and the report should say so."""
    env = RecoveryEnv(SCENARIOS["baseline"], seed=15)
    rejected = 0
    for episode in env.reset():
        if episode.customer.has_mandate:
            continue
        result = env.step(episode, Action(ActionType.RETRY, method=episode.method))
        assert result.invalid and result.cost_paise == 0 and episode.attempts == 0
        # The clock still moves, so a customer may pay unprompted during the two
        # minutes the rejected attempt took. That must never be booked as ours.
        assert result.attribution is not AttributionTruth.SYSTEM_RECOVERED
        rejected += 1
    assert rejected > 50, "most customers should not have a mandate on file"


# ── The constraints have to bite ──────────────────────────────────────────


def test_repeated_contact_decays_and_quiet_hours_suppress_clicks() -> None:
    """Fatigue and quiet hours are optimisation pressure, not decoration. If
    neither reduced click-through, a contact budget would be free to ignore."""
    env = RecoveryEnv(SCENARIOS["baseline"], seed=16)
    episodes = env.reset()

    daytime = [e for e in episodes if 9 <= e.failed_at.hour <= 17]
    overnight = [e for e in episodes if e.failed_at.hour >= 22 or e.failed_at.hour < 7]
    assert daytime and overnight

    def click_rate(batch, contacts: int) -> float:
        opened = 0
        for episode in batch:
            episode.customer.contacts = contacts
            rate = episode.customer.channel_rate(Channel.WHATSAPP, Tone.FRIENDLY)
            rate *= episode.customer.intent_at(episode.failed_at, episode.failed_at)
            rate *= episode.customer.fatigue_multiplier()
            if RecoveryEnv._is_quiet_hour(episode.failed_at):
                rate *= 0.28
            opened += rate
        return opened / len(batch)

    assert click_rate(daytime, 0) > click_rate(overnight, 0) * 1.8
    assert click_rate(daytime, 3) < click_rate(daytime, 0) * 0.8


def test_a_message_suggesting_an_unavailable_method_is_worth_less() -> None:
    """Suggesting a rail the customer does not hold is wasted spend, and the
    environment has to charge for it or `available_methods` carries no weight."""
    env = RecoveryEnv(SCENARIOS["baseline"], seed=17)
    grounded = ungrounded = 0
    for episode in env.reset():
        missing = [m for m in PaymentMethod if m not in episode.customer.alt_methods]
        if not missing:
            continue
        _, detail, _ = env._do_send_link(
            episode, Action(ActionType.SEND_LINK, channel=Channel.WHATSAPP, method=missing[0])
        )
        ungrounded += "not opened" in detail
        grounded += 1
    assert grounded > 20 and ungrounded / grounded > 0.6


# ── The emission model, which decides whether the ML is real ──────────────


def test_every_emission_distribution_is_a_distribution() -> None:
    for cause, profile in em.EMISSIONS.items():
        for field_name in ("reason", "step", "source"):
            total = sum(getattr(profile, field_name).values())
            assert abs(total - 1.0) < 1e-9, f"{cause}.{field_name} sums to {total}"


def test_no_error_field_gives_away_the_cause() -> None:
    """Every observable value must appear under at least two causes. A value unique
    to one cause is a leaked label, and a classifier trained on it learns the
    lookup table its author wrote rather than anything about payments."""
    for field_name in ("reason", "step", "source"):
        owners: dict[str, set[str]] = {}
        for cause, profile in em.EMISSIONS.items():
            for value in getattr(profile, field_name):
                owners.setdefault(value, set()).add(cause)
        for value, causes in owners.items():
            assert len(causes) > 1, f"{field_name}={value!r} only ever occurs for {causes}"


def test_the_error_fields_alone_cap_well_below_certainty() -> None:
    """The Bayes-optimal score obtainable from the error triple with no bank,
    customer or timing context. Reported in the eval as a floor to clear and a
    ceiling that proves the auxiliary features are doing the work."""
    bound = em.label_ambiguity()
    assert 0.60 <= bound["bayes_optimal_accuracy_error_fields_only"] <= 0.75
    assert bound["distinct_signatures"] > 100


def test_the_pairs_that_are_meant_to_be_confusable_actually_are() -> None:
    """Downtime vs a network blip, and an abandoned OTP vs stale card details.
    Both pairs must overlap heavily on the error fields, forcing the classifier to
    use correlated failures and customer history instead."""
    for left, right in (
        ("BANK_DOWNTIME", "NETWORK_TRANSIENT"),
        ("AUTH_TIMEOUT", "WRONG_CREDENTIALS"),
    ):
        a, b = em.EMISSIONS[left], em.EMISSIONS[right]
        shared = sum(
            min(a.reason.get(r, 0.0), b.reason.get(r, 0.0)) for r in em.ERROR_REASONS
        )
        assert shared > 0.35, f"{left} and {right} share only {shared:.0%} of their mass"
