"""Recovery policies, and the loop that scores them.

Every policy in this package answers the same question — *what should we do about
this failed payment right now?* — from the same observation, and is charged the
same prices by the same environment. That is the only way the comparison in
`app/eval/` means anything.

The set is deliberately ordered from "does nothing" to "cheats":

    do_nothing    the counterfactual; recovers whatever came back unaided
    naive_retry   the incumbent: three retries on a timer, then a blast
    rules         a strong hand-written expert policy — the bar to beat
    payrevive     this project's proposal: expected value over a fitted model
    oracle        full latent knowledge; the ceiling, not a result
"""

from app.policies.base import BasePolicy, Policy, run_episode
from app.policies.baselines import DoNothingPolicy, NaiveRetryPolicy
from app.policies.oracle import OraclePolicy
from app.policies.payrevive import PayRevivePolicy
from app.policies.rules import RulesPolicy

LADDER = {
    "do_nothing": lambda env: DoNothingPolicy(),
    "naive_retry": lambda env: NaiveRetryPolicy(),
    "rules": lambda env: RulesPolicy(),
    "payrevive": lambda env: PayRevivePolicy(env),
    "oracle": lambda env: OraclePolicy(env),
}
"""The ladder, in reporting order. Built per batch rather than once, because the
oracle is scored against a specific environment and a learning policy needs a place
to reset per-batch state. `do_nothing` first and `oracle` last is not cosmetic — the
harness reads the floor and the ceiling out of this dict by name."""

__all__ = [
    "LADDER",
    "BasePolicy",
    "DoNothingPolicy",
    "NaiveRetryPolicy",
    "OraclePolicy",
    "PayRevivePolicy",
    "Policy",
    "RulesPolicy",
    "run_episode",
]
