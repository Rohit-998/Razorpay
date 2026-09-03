"""The two reference points every other policy is judged between.

`DoNothingPolicy` is the counterfactual. It is not a strawman — it recovers a
substantial share of the money, because a lot of failed payments come back on
their own. Any system that cannot beat it is worse than switching itself off,
and reporting a recovery rate without it is how a dashboard ends up taking
credit for other people's work.

`NaiveRetryPolicy` is what most production retry loops actually do: fire a fixed
number of attempts on a fixed schedule, then message everyone who is left. It is
here because it is the honest incumbent. Beating a strawman proves nothing;
beating this is the minimum bar for the work to be worth deploying.
"""

from __future__ import annotations

from app.policies.base import BasePolicy
from app.sim.types import Action, ActionType, Channel, Observation, Tone


class DoNothingPolicy(BasePolicy):
    """Take no recovery action. Measures what returns unaided."""

    name = "do_nothing"
    description = (
        "The counterfactual baseline. Spends nothing and contacts nobody, so "
        "every recovery it books is a customer who came back on their own."
    )

    def act(self, obs: Observation) -> Action:
        return Action(
            ActionType.GIVE_UP,
            reason="baseline: measuring what comes back with no intervention",
        )


class NaiveRetryPolicy(BasePolicy):
    """Three retries on a fixed schedule, then one message. The incumbent.

    Note what it does not do. It never asks whether the bank is up, whether the
    instrument is already dead, or whether this customer was about to pay anyway.
    It retries because retrying is what it does, which is why it destroys value
    on an adversarial batch while still looking respectable on an average week.
    """

    name = "naive_retry"
    description = (
        "Fixed schedule: retry at +2m, +30m and +2h, then a single SMS. No "
        "diagnosis, no stopping rule — the loop most gateways ship with."
    )

    RETRY_SCHEDULE_MINUTES = (2, 30, 120)

    def begin(self, episode_count: int) -> None:
        self._fired: dict[str, int] = {}

    def act(self, obs: Observation) -> Action:
        elapsed = obs.minutes_since_failure
        # Counts attempts *this policy made*, not attempts the gateway accepted.
        # A real retry loop has no idea its charge was structurally impossible; it
        # ticks its counter and moves on. Reading the environment's accepted-attempt
        # count instead would turn a rejected retry into an infinite loop.
        fired = self._fired.get(obs.payment_id, 0)

        if fired < len(self.RETRY_SCHEDULE_MINUTES):
            due_at = self.RETRY_SCHEDULE_MINUTES[fired]
            if elapsed < due_at:
                return Action(
                    ActionType.WAIT,
                    wait_minutes=int(due_at - elapsed) + 1,
                    reason=f"waiting for scheduled attempt {fired + 1}",
                )
            self._fired[obs.payment_id] = fired + 1
            return Action(
                ActionType.RETRY,
                method=obs.method,
                reason=f"scheduled retry {fired + 1} of {len(self.RETRY_SCHEDULE_MINUTES)}",
            )

        if obs.contacts_made == 0:
            return Action(
                ActionType.SEND_LINK,
                channel=Channel.SMS,
                method=obs.method,
                tone=Tone.URGENT,
                reason="retries exhausted, sending a payment link",
            )

        return Action(ActionType.GIVE_UP, reason="schedule exhausted")
