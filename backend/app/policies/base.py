"""The policy interface, and the loop that runs one against the environment.

A policy is anything that turns an `Observation` into an `Action`. That is a
deliberately narrow contract: it receives only what a real system would have on
a live payment, and it can only do things the environment knows how to price.

Two rules make the comparison fair:

  Every policy sees the same `Observation` type. Nothing in it is derived from
  latent state, so no policy can accidentally cheat. `OraclePolicy` is the sole
  exception and says so loudly — it exists to bound how much value is on the
  table, and its score is a ceiling, never a claim.

  Every policy is run through the same loop and finalised the same way. Cost,
  contacts and the clock are the environment's business, not the policy's, so a
  policy cannot quietly award itself a cheaper message or an extra hour.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.sim.environment import Episode, RecoveryEnv
from app.sim.types import Action, ActionType, Observation, Terminal

MAX_STEPS_PER_EPISODE = 24
"""Guard against a policy that never terminates. Hitting it is a bug in the
policy, and the harness reports how often it happens."""


@runtime_checkable
class Policy(Protocol):
    """Turns observations into actions. Stateless across episodes by default."""

    name: str
    description: str

    def begin(self, episode_count: int) -> None:
        """Called once before a batch. Somewhere to reset per-batch state."""

    def act(self, obs: Observation) -> Action:
        """Choose the next action for this payment."""
        ...

    def observe_outcome(self, obs: Observation, action: Action, paid: bool) -> None:
        """Optional feedback hook. Only the learning policies implement it."""


class BasePolicy:
    """Convenience base with no-op hooks, so simple policies stay short."""

    name = "base"
    description = ""

    def begin(self, episode_count: int) -> None:  # noqa: D102
        return None

    def observe_outcome(self, obs: Observation, action: Action, paid: bool) -> None:  # noqa: D102
        return None

    def act(self, obs: Observation) -> Action:  # noqa: D102
        raise NotImplementedError


def run_episode(env: RecoveryEnv, episode: Episode, policy: Policy) -> Episode:
    """Drive one episode to a terminal state, then close the recovery window.

    `finalize` runs unconditionally, including for a policy that gave up on the
    first step. That is what keeps the baseline honest: unprompted payments are
    collected either way, so stopping early costs a policy only the money it
    could have caused, never the money that was already on its way.
    """
    for _ in range(MAX_STEPS_PER_EPISODE):
        if episode.terminal is not Terminal.OPEN or episode.now >= episode.deadline:
            break
        obs = env.observe(episode)
        action = policy.act(obs)
        result = env.step(episode, action)
        policy.observe_outcome(obs, action, result.paid)
        if action.type in (ActionType.GIVE_UP, ActionType.ESCALATE):
            break
    return env.finalize(episode)
