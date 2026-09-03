"""Recovery policies, and the loop that scores them.

Every policy in this package answers the same question — *what should we do about
this failed payment right now?* — from the same observation, and is charged the
same prices by the same environment. That is the only way the comparison in
`app/eval/` means anything.

The set is deliberately ordered from "does nothing" to "cheats":

    do_nothing    the counterfactual; recovers whatever came back unaided
    naive_retry   the incumbent: three retries on a timer, then a blast
    rules         a strong hand-written expert policy — the bar to beat
    oracle        full latent knowledge; the ceiling, not a result
"""

from app.policies.base import BasePolicy, Policy, run_episode
from app.policies.baselines import DoNothingPolicy, NaiveRetryPolicy
from app.policies.oracle import OraclePolicy
from app.policies.rules import RulesPolicy

__all__ = [
    "BasePolicy",
    "DoNothingPolicy",
    "NaiveRetryPolicy",
    "OraclePolicy",
    "Policy",
    "RulesPolicy",
    "run_episode",
]
