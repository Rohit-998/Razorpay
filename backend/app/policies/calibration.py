"""A recoverability model fitted from action logs, not hand-tuned.

`payrevive` needs one number for every route it might take: the probability that
this action, on this payment, at this hour, actually collects the money. Writing
those numbers by hand is the thing `rules` already does, and it is what makes
`rules` beatable — a fixed threshold cannot notice that this week's outages clear
in twenty minutes rather than ninety.

So they are fitted. Three properties make the fit honest:

  *Only observable conditioning.* Every cell is keyed on things a live system has
  at decision time — the error reason off the webhook, hours since failure, how
  many attempts and messages this payment has already had. No cell is keyed on the
  root cause, on `funds_available_at`, or on anything else latent. The model cannot
  encode knowledge the policy is not allowed to use.

  *Only observed outcomes.* Each trial is an action that was actually taken and an
  outcome that was actually visible: a retry that authorised or did not, a link our
  own tracking saw opened, an agent call that ended in a payment. Nothing is fitted
  against the environment's private attribution verdict.

  *Out of sample.* Calibration runs on `CALIBRATION_SEEDS`, which is disjoint from
  the seeds `app.eval` scores on. A model fitted on the same batches it is then
  measured over would report its own training error as a result.

The exploration log comes from a deliberately randomised `ProbePolicy` plus the
incumbent `rules`, which is the production-shaped version of this: you fit the next
model on the logs your current system already produced, and you add exploration
where the current system never goes. `naive_retry` never sends a link after 2am and
`rules` never escalates below ₹25,000, so without the probe those cells would be
empty and the new policy would inherit the incumbent's blind spots as if they were
facts about the world.

Regenerate with:

    cd backend && python -m app.policies.calibration
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.models.schemas import PaymentMethod
from app.policies.base import BasePolicy
from app.sim.environment import Episode
from app.sim.types import Action, ActionType, Channel, Observation, Tone

MODEL_PATH = Path(__file__).with_name("calibration.json")

CALIBRATION_SEEDS = tuple(range(1001, 1031))
"""Deliberately disjoint from the evaluation seeds (1–20 by default). If these
overlapped, every lift figure in the report would be partly in-sample."""

MIN_TRIALS = 40
"""Below this a cell is too thin to trust, and the lookup backs off to a coarser
one rather than reporting a rate built from nine observations."""

LINK_OPENED_MARKER = "customer opens it"
"""How the log records that our link was opened. Standing in for the click
tracking a real merchant has on its own payment links — the one part of a link's
outcome that is genuinely observable rather than inferred."""

ELAPSED_KNOTS_HOURS: tuple[float, ...] = (1.0, 6.0, 24.0)
"""Hours since failure, bucketed. Coarse on purpose: the shape that matters is
"minutes", "same shift", "next day", "stale", and finer buckets only thin the
cells out."""


def elapsed_bucket(hours: float) -> int:
    """Which elapsed-time bucket an age in hours falls into."""
    for index, knot in enumerate(ELAPSED_KNOTS_HOURS):
        if hours < knot:
            return index
    return len(ELAPSED_KNOTS_HOURS)


def tries_bucket(count: int) -> int:
    """Attempts or contacts already spent, capped. The third bucket is "several",
    where the marginal rate has flattened out and the distinction stops paying."""
    return min(count, 2)


HEALTH_KNOTS: tuple[float, ...] = (0.25, 0.60)
"""The bank's observed success rate over the last hour, bucketed into dark,
degraded and healthy.

This is the dimension the error reason cannot supply. A `payment_failed` on a bank
clearing 90% of charges is a problem with this payment; the same webhook on a bank
clearing 5% is a problem with the bank, and the two call for opposite actions — act
now, or wait. Without it a fitted model reports one number for both and a policy
built on it cannot tell an outage from a decline."""


def health_bucket(rate: float) -> int:
    """Which health bucket an observed hourly success rate falls into."""
    for index, knot in enumerate(HEALTH_KNOTS):
        if rate < knot:
            return index
    return len(HEALTH_KNOTS)


HEALTH_GAP_KNOTS_HOURS: tuple[float, ...] = (1.0, 6.0)
"""How far ahead a health forecast reaches, bucketed.

Knowing a bank is dark is only half of what a decision to wait needs; the other half
is whether it will still be dark later. `rules` answers that with a constant — wait 45
minutes — and the constant is right about as often as outages last 45 minutes. These
buckets match the distances this policy can actually wait, since it waits to the next
elapsed knot: under an hour, the rest of the shift, or overnight."""


def gap_bucket(hours: float) -> int:
    """Which forecast-horizon bucket a wait of this length falls into."""
    for index, knot in enumerate(HEALTH_GAP_KNOTS_HOURS):
        if hours < knot:
            return index
    return len(HEALTH_GAP_KNOTS_HOURS)


@dataclass(frozen=True)
class Model:
    """Fitted success rates, with the trial counts that produced them.

    Keys are `route|reason|elapsed_bucket|tries_bucket`, and `rate` drops the
    trailing fields one at a time until a cell has enough trials to be worth
    reading. That backoff is the whole reason thin cells are safe to keep: a rate
    fitted from nine observations is never used, but the coarser rate above it
    still carries the signal those nine observations belong to.
    """

    tallies: dict[str, tuple[int, int]]
    channel_lift: dict[str, float]
    tone_lift: dict[str, float]
    health_factor: dict[str, float]
    health_transition: dict[str, float]
    quiet_lift: float
    attempt_decay: float
    fitted_on: dict[str, object]

    def rate(self, *parts: object, default: float = 0.0) -> float:
        """Success rate for the most specific cell that has enough trials."""
        fields = [str(p) for p in parts]
        while fields:
            hits, trials = self.tallies.get("|".join(fields), (0, 0))
            if trials >= MIN_TRIALS:
                return hits / trials
            fields.pop()
        return default

    def trials(self, *parts: object) -> int:
        return self.tallies.get("|".join(str(p) for p in parts), (0, 0))[1]

    def health(self, reason: str, bucket: int) -> float:
        """Multiplier on a retry's fitted rate for the bank's current health.

        Reason-specific where the log can resolve it, standardised-pooled where it
        cannot, and 1.0 where neither is fitted — which is the right default, because
        an unfitted health dimension should leave the base rate alone rather than
        invent a discount for it.
        """
        specific = self.health_factor.get(f"{reason}|{bucket}")
        if specific is not None:
            return specific
        return self.health_factor.get(f"*|{bucket}", 1.0)

    def forecast(self, bucket: int, gap_hours: float) -> dict[int, float]:
        """Where the bank's health will be `gap_hours` from now, as a distribution.

        This is the half of "should I wait" that a fitted rate cannot supply on its
        own. A dark bank discounts a retry now; whether waiting fixes anything depends
        on whether it is still dark then, and that is a question about how long this
        merchant's outages last — fitted here from how observed health actually moved
        in the logs, rather than asserted as a fixed wait.

        Falls back to certainty about the present when a cell is unfitted, which is
        the conservative reading: no evidence of recovery means no credit for it.
        """
        gap = gap_bucket(gap_hours)
        weights = {
            b: self.health_transition.get(f"{bucket}|{gap}|{b}", 0.0)
            for b in range(len(HEALTH_KNOTS) + 1)
        }
        total = sum(weights.values())
        if total <= 0.0:
            return {bucket: 1.0}
        return {b: w / total for b, w in weights.items() if w > 0.0}


def load(path: Path = MODEL_PATH) -> Model:
    """Read the committed fit. Raises if it is missing, rather than silently
    falling back to invented numbers — a policy quietly running on defaults would
    still produce a plausible report."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Model(
        tallies={k: (v[0], v[1]) for k, v in raw["tallies"].items()},
        channel_lift=raw["channel_lift"],
        tone_lift=raw["tone_lift"],
        health_factor=raw["health_factor"],
        health_transition=raw["health_transition"],
        quiet_lift=raw["quiet_lift"],
        attempt_decay=raw["attempt_decay"],
        fitted_on=raw["fitted_on"],
    )


class ProbePolicy(BasePolicy):
    """Randomised exploration, for logging only. Never scored.

    Its job is coverage, not money: it takes legal actions across the whole space
    so the fit has trials in cells the incumbent never visits. It deliberately
    keeps every action *structurally* legal — retries only with a mandate, on rails
    the customer holds — because an invalid action teaches nothing about whether a
    route works, and letting them into the log would bias every rate downwards.
    """

    name = "probe"
    description = "Randomised exploration policy used to generate calibration logs."

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Observation) -> Action:
        roll = float(self.rng.random())
        rails = list(obs.available_methods) or [obs.method]

        if obs.has_mandate and roll < 0.45:
            return Action(
                ActionType.RETRY,
                method=rails[int(self.rng.integers(0, len(rails)))],
                reason="probe: sampling a retry outcome",
            )
        if roll < 0.80:
            return Action(
                ActionType.SEND_LINK,
                channel=list(Channel)[int(self.rng.integers(0, len(Channel)))],
                method=rails[int(self.rng.integers(0, len(rails)))],
                tone=list(Tone)[int(self.rng.integers(0, len(Tone)))],
                reason="probe: sampling a link outcome",
            )
        if roll < 0.88 and not obs.escalated and obs.agent_calls_remaining > 0:
            return Action(ActionType.ESCALATE, reason="probe: sampling an agent outcome")
        if roll < 0.96:
            return Action(
                ActionType.WAIT,
                wait_minutes=int(self.rng.integers(15, 600)),
                reason="probe: advancing the clock to reach later buckets",
            )
        return Action(ActionType.GIVE_UP, reason="probe: ending this episode")


def _bump(store: dict[str, list[int]], key: str, hit: bool) -> None:
    cell = store.setdefault(key, [0, 0])
    cell[0] += int(hit)
    cell[1] += 1


def _bump_pair(store: dict[str, int], key: str) -> None:
    store[key] = store.get(key, 0) + 1


def _walk(
    ep: Episode,
    store: dict[str, list[int]],
    slices: dict[str, list[int]],
    transitions: dict[str, int],
    randomised: bool,
) -> None:
    """Turn one finished episode's history into trials.

    Counters are rebuilt as the history is replayed rather than read off the final
    episode, because every rate has to be conditioned on what was known *at that
    step* — the third retry's success rate is a different quantity from the first's,
    and reading the final attempt count would collapse them.

    `randomised` says whether this episode came from the probe. The keyed cells in
    `store` take every episode: they are conditional rates on observables, and
    conditioning is what makes them safe to read whatever chose the action. The
    marginals in `slices` take probe episodes only. That distinction is the whole
    validity of `channel_lift` and the rest — `rules` picks its channel from the
    amount and holds off during outages, so a marginal pooled over its episodes
    would report the incumbent's own policy back as a fact about customers.
    """
    reason = ep.emission.error_reason
    attempts = contacts = 0
    marks: list[tuple[float, int]] = []
    for step in ep.history:
        if step.invalid:
            continue
        hours = max(0.0, (step.taken_at - ep.failed_at).total_seconds() / 3600.0)
        age = elapsed_bucket(hours)
        kind = step.action.type

        if randomised:
            # Every step, whatever it was, is a reading of the bank's health at a known
            # time. Waits count as much as retries here: what is being fitted is how
            # the gateway moved, not what we did about it.
            marks.append((hours, health_bucket(step.bank_health)))

        if kind is ActionType.RETRY:
            switched = step.action.method is not ep.method
            route = "retry_switched" if switched else "retry_same"
            routes = [route]
            if switched and step.action.method is not None:
                # Rail-specific too, because "switch" is not one action. A drained
                # account is escaped by a wallet and not by another card, and the
                # policy's actual decision is which rail — a pooled switched rate
                # cannot answer that. Thin rail cells back off to the pooled one.
                routes.append(f"retry_to_{step.action.method.value}")
            for key in routes:
                _bump(store, f"{key}|{reason}|{age}|{tries_bucket(attempts)}", step.paid)
                _bump(store, f"{key}|{reason}|{age}", step.paid)
                _bump(store, f"{key}|{reason}", step.paid)
                _bump(store, key, step.paid)
            if randomised:
                # Attempt count pooled over every reason, which the keyed cells above
                # cannot give: they stop distinguishing attempts at `tries_bucket`'s
                # cap, so a policy reading them has no way to know that the eighth
                # retry is worth less than the third. This is the slice that tells it.
                _bump(slices, f"attempt|{tries_bucket(attempts)}", step.paid)
                # Health keeps the reason attached, because the reason mix is itself a
                # function of bank health — an outage hour is full of `payment_failed`
                # and `timeout`. Pooling health across reasons would measure that shift
                # in composition and report it as an effect of the outage.
                _bump(
                    slices,
                    f"health|{health_bucket(step.bank_health)}|{reason}",
                    step.paid,
                )
            attempts += 1

        elif kind is ActionType.SEND_LINK:
            opened = LINK_OPENED_MARKER in step.detail
            _bump(store, f"link_open|{reason}|{age}|{tries_bucket(contacts)}", opened)
            _bump(store, f"link_open|{reason}|{age}", opened)
            _bump(store, f"link_open|{reason}", opened)
            _bump(store, "link_open", opened)
            if randomised:
                if step.action.channel is not None:
                    _bump(slices, f"channel|{step.action.channel.value}", opened)
                _bump(slices, f"tone|{step.action.tone.value}", opened)
                _bump(
                    slices,
                    f"quiet|{step.taken_at.hour >= 22 or step.taken_at.hour < 8}",
                    opened,
                )
            contacts += 1

        elif kind is ActionType.ESCALATE:
            _bump(store, f"agent_paid|{reason}|{age}", step.paid)
            _bump(store, f"agent_paid|{reason}", step.paid)
            _bump(store, "agent_paid", step.paid)

    # Health transitions, from every ordered pair of readings in the episode rather
    # than consecutive ones only. The probe's waits are uniform over 15–600 minutes,
    # so consecutive pairs alone would leave the longer horizons thin; all pairs fill
    # them from the same readings. Within a gap bucket this still estimates the
    # transition the log actually exhibits, which is the quantity a policy deciding
    # how long to wait needs.
    for index, (at, was) in enumerate(marks):
        for later_at, became in marks[index + 1:]:
            _bump_pair(transitions, f"{was}|{gap_bucket(later_at - at)}|{became}")


def _lift(slices: dict[str, list[int]], prefix: str, overall: float) -> dict[str, float]:
    """A slice's open rate relative to the overall one.

    A marginal, not a coefficient — channel and tone are chosen by the probe
    uniformly at random and independently of the payment, which is exactly what
    makes the marginal unconfounded here. It would not be if these came from the
    incumbent's logs, where channel is chosen from the amount.
    """
    out: dict[str, float] = {}
    for key, (hits, trials) in slices.items():
        head, _, value = key.partition("|")
        if head != prefix or trials < MIN_TRIALS or overall <= 0.0:
            continue
        out[value] = (hits / trials) / overall
    return out


def _attempt_decay(slices: dict[str, list[int]]) -> float:
    """How much worse each successive retry is, as a ratio.

    The keyed cells stop resolving attempts at `tries_bucket`'s cap, so a policy
    looking at them cannot tell the eighth retry from the third — which is how a
    policy ends up retrying until it runs out of steps. This measures the ratio
    between consecutive attempt buckets so the extrapolation past the cap is fitted
    rather than invented. Clamped well below 1: the tail is unambiguously declining
    in the logs, and a decay near 1 would mean the extrapolation had failed rather
    than that retries do not decay.
    """
    rates = []
    for k in range(3):
        hits, trials = slices.get(f"attempt|{k}", [0, 0])
        rates.append(hits / trials if trials >= MIN_TRIALS else None)
    ratios = [
        b / a for a, b in zip(rates, rates[1:])
        if a is not None and b is not None and a > 0.0
    ]
    if not ratios:
        return 0.5
    return min(0.9, max(0.1, sum(ratios) / len(ratios)))


def _health_factors(slices: dict[str, list[int]]) -> tuple[dict[str, float], float]:
    """How much a retry is worth on a dark bank versus a healthy one, per reason.

    This is deliberately not one number per health bucket. Fitting it that way first
    produced 0.93x / 1.06x / 1.07x — a signal so flat it would have been fair to call
    bank health uninformative and drop it. It is not flat; it is an interaction, and
    pooling over reasons is what destroyed it:

        gateway_technical_error    19% dark -> 37% degraded -> 56% healthy
        network_error              23%      -> 45%          -> 73%
        timeout                    34%      -> 60%          -> 81%
        insufficient_funds          2%      ->  5%          ->  6%
        authentication_failed      40%      -> 44%          -> 38%

    Which is the physics, and it is worth stating plainly because it is the reason
    this dimension pays. A timeout is the gateway's failure, so whether retrying it
    works is a question about the gateway, and the answer swings by a factor of two.
    A drained account is the customer's, and the bank's queue length has nothing to
    say about it. One scalar per bucket has to average those, and the average is
    dominated by whichever reason is most common rather than by whichever is most
    health-sensitive.

    So each factor is a *within-reason* ratio: this reason's retry rate in this health
    bucket over this reason's retry rate pooled across buckets. By construction it
    averages to 1.0 over the probe's own health mix, which is what makes it safe to
    multiply into a rate from `tallies` rather than replacing it. Reasons too thin to
    resolve fall back to `*|{bucket}`, a directly standardised pooled figure that
    holds the reason mix fixed so it cannot pick up the fact that outage hours are
    full of timeouts in the first place.

    One residual approximation, stated rather than hidden: the base rates in
    `tallies` pool both loggers, and `rules` holds off during outages, so its retries
    skew healthy. Multiplying a probe-centred factor into a base that is already
    mildly health-skewed leaves that skew in place. It is the same construction
    `channel_lift` and `tone_lift` already use, it biases the level rather than the
    ordering, and closing it properly means health-keyed cells and roughly three
    times the log to fill them.

    Returns the factors and the standardised overall rate the `*` entries are
    relative to, which goes into the fit's provenance so a ratio can be read as a rate.
    """
    cells: dict[tuple[int, str], tuple[int, int]] = {}
    for key, (hits, trials) in slices.items():
        head, _, rest = key.partition("|")
        bucket, _, reason = rest.partition("|")
        if head != "health" or not reason:
            continue
        cells[(int(bucket), reason)] = (hits, trials)

    buckets = sorted({b for b, _ in cells})
    if not buckets:
        return {}, 0.0

    factors: dict[str, float] = {}
    pooled: dict[str, tuple[int, int]] = {}
    for reason in {r for _, r in cells}:
        hits = sum(cells.get((b, reason), (0, 0))[0] for b in buckets)
        trials = sum(cells.get((b, reason), (0, 0))[1] for b in buckets)
        pooled[reason] = (hits, trials)
        if trials < MIN_TRIALS or hits == 0:
            continue
        base = hits / trials
        for bucket in buckets:
            cell_hits, cell_trials = cells.get((bucket, reason), (0, 0))
            if cell_trials >= MIN_TRIALS:
                factors[f"{reason}|{bucket}"] = (cell_hits / cell_trials) / base

    # The pooled backoff. A reason only carries weight where every bucket has enough
    # of it to state a rate, because standardising over a reason one bucket never saw
    # means filling that cell with an assumption, and the assumption would then be
    # doing the work.
    common = [
        r
        for r in pooled
        if all(cells.get((b, r), (0, 0))[1] >= MIN_TRIALS for b in buckets)
    ]
    if not common:
        return factors, 0.0
    weights = {r: pooled[r][1] for r in common}
    total = sum(weights.values())
    overall = sum(weights[r] * (pooled[r][0] / pooled[r][1]) for r in common) / total
    if overall <= 0.0:
        return factors, 0.0
    for bucket in buckets:
        standardised = (
            sum(
                weights[r] * (cells[(bucket, r)][0] / cells[(bucket, r)][1])
                for r in common
            )
            / total
        )
        factors[f"*|{bucket}"] = standardised / overall
    return factors, overall


def _transition_table(counts: dict[str, int]) -> dict[str, float]:
    """Normalise the transition counts into probabilities per (from, gap).

    A row is only kept if it has enough observations to be a forecast rather than an
    anecdote. A dropped row makes `Model.forecast` fall back to assuming nothing
    changes, which is the reading that gives waiting no credit it has not earned.
    """
    rows: dict[str, int] = {}
    for key, count in counts.items():
        was, gap, _ = key.split("|")
        rows[f"{was}|{gap}"] = rows.get(f"{was}|{gap}", 0) + count
    return {
        key: count / rows[key.rsplit("|", 1)[0]]
        for key, count in sorted(counts.items())
        if rows[key.rsplit("|", 1)[0]] >= MIN_TRIALS
    }


def fit(log: list[tuple[str, Episode]], fitted_on: dict[str, object]) -> Model:
    """Tally every logged action into the model.

    The log is `(logger, episode)` pairs rather than bare episodes because the fit
    treats them differently, and has to: conditional cells take everything, marginals
    take only what the probe randomised.
    """
    store: dict[str, list[int]] = {}
    slices: dict[str, list[int]] = {}
    transitions: dict[str, int] = {}
    for logger, ep in log:
        _walk(ep, store, slices, transitions, randomised=logger == "probe")
    opens, sends = store.get("link_open", [0, 0])
    overall = opens / sends if sends else 0.0
    quiet = _lift(slices, "quiet", overall)
    health_factor, retry_overall = _health_factors(slices)

    return Model(
        tallies={k: (v[0], v[1]) for k, v in sorted(store.items())},
        channel_lift=_lift(slices, "channel", overall),
        tone_lift=_lift(slices, "tone", overall),
        health_factor=health_factor,
        health_transition=_transition_table(transitions),
        quiet_lift=quiet.get("True", 1.0),
        attempt_decay=_attempt_decay(slices),
        fitted_on={
            **fitted_on,
            "link_open_rate_overall": round(overall, 4),
            "retry_rate_standardised_probe_only": round(retry_overall, 4),
            "health_readings_probe_only": sum(transitions.values()),
        },
    )


def collect(
    seeds: tuple[int, ...] = CALIBRATION_SEEDS,
) -> tuple[list[tuple[str, Episode]], dict[str, object]]:
    """Run the exploration log, tagging each episode with the policy that produced it.

    The tag is carried alongside the episode rather than stored on it: which policy
    logged a payment is a fact about this fit, not a property of the payment, and
    writing it onto `Episode` would put a calibration concern inside the simulator.

    Imports are local so importing this module for `load()` alone does not drag in
    the whole simulator.
    """
    from app.policies.base import run_episode
    from app.policies.rules import RulesPolicy
    from app.sim.environment import RecoveryEnv
    from app.sim.scenarios import SCENARIOS

    log: list[tuple[str, Episode]] = []
    for name, scenario in sorted(SCENARIOS.items()):
        for seed in seeds:
            for label in ("probe", "rules"):
                env = RecoveryEnv(scenario, seed=seed)
                batch = env.reset()
                policy = ProbePolicy(seed=seed) if label == "probe" else RulesPolicy()
                policy.begin(len(batch))
                for episode in batch:
                    run_episode(env, episode, policy)
                log.extend((label, episode) for episode in batch)
    return log, {
        "seeds": list(seeds),
        "scenarios": sorted(SCENARIOS),
        "loggers": ["probe", "rules"],
        "episodes": len(log),
        "probe_episodes": sum(1 for label, _ in log if label == "probe"),
    }


def main() -> int:
    import sys

    print(f"logging {len(CALIBRATION_SEEDS)} seeds x 5 scenarios x 2 policies", file=sys.stderr)
    log, provenance = collect()
    model = fit(log, provenance)
    MODEL_PATH.write_text(
        json.dumps(
            {
                "tallies": {k: list(v) for k, v in model.tallies.items()},
                "channel_lift": {k: round(v, 4) for k, v in sorted(model.channel_lift.items())},
                "tone_lift": {k: round(v, 4) for k, v in sorted(model.tone_lift.items())},
                "health_factor": {
                    k: round(v, 4) for k, v in sorted(model.health_factor.items())
                },
                "health_transition": {
                    k: round(v, 4) for k, v in sorted(model.health_transition.items())
                },
                "quiet_lift": round(model.quiet_lift, 4),
                "attempt_decay": round(model.attempt_decay, 4),
                "fitted_on": model.fitted_on,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    usable = sum(1 for v in model.tallies.values() if v[1] >= MIN_TRIALS)
    print(
        f"{provenance['episodes']:,} episodes -> {len(model.tallies):,} cells "
        f"({usable:,} with >={MIN_TRIALS} trials)",
        file=sys.stderr,
    )
    for route in ("retry_same", "retry_switched", "link_open", "agent_paid"):
        hits, trials = model.tallies.get(route, (0, 0))
        if trials:
            print(f"  {route:<15} {hits / trials:6.1%}  over {trials:,} trials", file=sys.stderr)
    print(f"  quiet-hour read lift {model.quiet_lift:.2f}x", file=sys.stderr)
    print(f"  each further retry is worth {model.attempt_decay:.2f}x the last", file=sys.stderr)
    labels = ("dark", "degraded", "healthy")
    spread = sorted(
        (
            (
                model.health(reason, 0) / model.health(reason, len(HEALTH_KNOTS)),
                reason,
            )
            for reason in {k.split("|")[0] for k in model.health_factor} - {"*"}
        )
    )
    print("  retry on a dark bank vs a healthy one, most health-sensitive first:", file=sys.stderr)
    for ratio, reason in spread[:4]:
        rates = "  ".join(
            f"{label} {model.health(reason, i):.2f}x" for i, label in enumerate(labels)
        )
        print(f"    {reason:<24}{rates}", file=sys.stderr)
    pooled = "  ".join(
        f"{label} {model.health_factor.get(f'*|{i}', 1.0):.2f}x"
        for i, label in enumerate(labels)
    )
    print(f"    {'(pooled backoff)':<24}{pooled}", file=sys.stderr)
    healthy = len(HEALTH_KNOTS)
    print("  chance a dark bank has recovered after waiting:", file=sys.stderr)
    for gap, span in enumerate(("under 1h", "1-6h", "over 6h")):
        forecast = model.forecast(0, HEALTH_GAP_KNOTS_HOURS[gap] - 0.5 if gap < len(HEALTH_GAP_KNOTS_HOURS) else 12.0)
        print(
            f"    {span:<12}{forecast.get(healthy, 0.0):.0%} healthy, "
            f"{forecast.get(0, 0.0):.0%} still dark",
            file=sys.stderr,
        )
    print(f"wrote {MODEL_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
