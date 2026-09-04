"""Labelled training data for the root-cause classifier, from the simulator.

Two things were wrong with where the training set used to come from.

The labels were the model's own output. `/model/train` read them from
`recovery_sessions.root_cause`, and `worker.process_failed_payment` overwrites that column
with `classification.root_cause` on every run. `batch.py` seeds it with the generator's true
cause, so the first training run was honest and every run after a pipeline pass was the
classifier being fitted to its own predictions. Accuracy under that arrangement measures
self-consistency, and rises as the model gets more confidently wrong.

The features were built by a different code path than the one that serves them. Any
divergence between the training row and the serving row is invisible in the metrics and
shows up only as production behaving unlike the report. So this module does not build
feature vectors: it hands the simulator's observation to the *production*
`FeatureExtractor` behind a store that answers from the simulated world instead of Redis.
The row that trains the model and the row that the worker classifies come off the same
function.

What crosses the boundary is `RecoveryEnv.observe()` and nothing else — the same projection
the policies see, which by construction holds no latent state. `true_cause` is read off the
episode as the label and is never a feature. In particular `bank_is_in_downtime` is fed from
the observable failure spike, not from `world.downtime_at()`: downtime-caused failures are
generated inside real outage windows, so the latent flag is very nearly the BANK_DOWNTIME
label itself, and a model trained on it would report an accuracy no deployment could
reproduce. The spike is what a merchant computing over its own recent traffic actually has.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from app.cache.feature_store import FeatureExtractor
from app.models.schemas import (
    BankHealthStatus,
    ErrorSource,
    FailedPayment,
    FeatureVector,
    PaymentMethod,
)
from app.ml.classifier import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from app.sim.environment import Episode, RecoveryEnv
from app.sim.scenarios import SCENARIOS
from app.sim.types import Observation


class _SimStore:
    """A feature store that answers from a simulated batch instead of Redis.

    It implements the three reads `FeatureExtractor` performs, so the extractor cannot tell
    the difference and no second copy of the feature-building code has to exist. Every value
    returned comes off an `Observation`, which is the simulator's guarantee that no latent
    state is in it.
    """

    def __init__(self, env: RecoveryEnv, observations: dict[str, Observation]) -> None:
        self._env = env
        self._obs = observations
        self._current: Observation | None = None
        self._method_times: dict[str, list[datetime]] = {}
        for ep in env.episodes:
            self._method_times.setdefault(ep.method.value, []).append(ep.failed_at)
        for times in self._method_times.values():
            times.sort()

    def use(self, observation: Observation) -> None:
        """Point the store at the payment about to be extracted.

        The extractor looks features up by bank code and customer id, and the simulator's
        signals are per-episode — two payments on HDFC an hour apart see different bank
        health. Rather than fake a global key space, the caller names the row first.
        """
        self._current = observation

    async def get_bank_health(self, bank: str) -> BankHealthStatus:
        obs = self._require()
        signal = obs.bank_signal
        return BankHealthStatus(
            bank_code=bank,
            is_healthy=signal.observed_success_rate_1h > 0.5 and not signal.concurrent_failure_spike,
            success_rate_1h=signal.observed_success_rate_1h,
            failure_count_1h=signal.observed_failure_count_1h,
            # The observable proxy, not `world.downtime_at()`. See the module docstring:
            # the latent flag is close enough to the BANK_DOWNTIME label to be leakage.
            is_in_downtime=signal.concurrent_failure_spike,
            downtime_severity="high" if signal.concurrent_failure_spike else None,
        )

    async def get_method_health(self, method: str) -> float:
        """Recent success rate for a payment method, from the merchant's own failure stream.

        Derived the same way `World.observed_success_rate` derives the bank figure, and for
        the same reason: successes outside recovery are not simulated, so the only honest
        estimate is the one a real dashboard shows — this hour's failures against the volume
        this method usually produces. Holding it at the production default of 0.95 instead
        would leave a constant column in the matrix, which XGBoost never splits on, and the
        feature would be listed in the vector while doing nothing.
        """
        obs = self._require()
        times = self._method_times.get(method)
        if not times:
            return 0.95
        hours = max(1.0, self._env.scenario.duration_days * 24.0)
        baseline = max(0.5, len(times) / hours)
        window_start = obs.now - timedelta(minutes=60)
        observed = sum(1 for t in times if window_start <= t <= obs.now)
        if observed <= baseline:
            return 0.95
        excess = min(1.0, (observed - baseline) / (baseline * 6.0))
        return round(float(max(0.02, 0.95 * (1.0 - excess))), 4)

    async def get_customer_features(self, customer_id: str) -> dict:
        obs = self._require()
        signal = obs.customer_signal
        return {
            "customer_success_rate_30d": signal.success_rate_90d,
            "customer_failure_count_7d": signal.failure_count_7d,
            "customer_recovery_response": signal.response_rate,
        }

    def _require(self) -> Observation:
        if self._current is None:
            raise RuntimeError("use() must name the payment before extracting it")
        return self._current


@dataclass(frozen=True)
class Row:
    """One training example: what was observable, what actually caused it, where it came from."""

    features: FeatureVector
    cause: str
    scenario: str
    seed: int


def _as_payment(ep: Episode) -> FailedPayment:
    """The episode as the webhook would have delivered it.

    Only fields a `payment.failed` payload carries. `true_cause` is not among them, which is
    the point of going through this type rather than reading the episode directly.
    """
    return FailedPayment(
        payment_id=ep.payment_id,
        order_id=ep.order_id,
        amount=ep.amount,
        currency="INR",
        method=ep.method,
        bank=ep.bank,
        vpa=ep.vpa,
        wallet=ep.wallet,
        error_code=ep.emission.error_code,
        error_source=ErrorSource(ep.emission.error_source),
        error_step=ep.emission.error_step,
        error_reason=ep.emission.error_reason,
        error_description=ep.emission.error_description,
        customer_contact=ep.customer.phone,
        customer_email=ep.customer.email,
        created_at=ep.failed_at,
    )


async def build_rows(scenario_names: list[str], seeds: list[int]) -> list[Row]:
    """Labelled rows from every `(scenario, seed)` pair, built by the production extractor.

    Deterministic: `RecoveryEnv.reset()` is a pure function of the pair, so the same
    arguments give the same dataset on any machine. No environment `step()` is ever called —
    a classifier sees a payment once, at the moment it fails, before any action is taken.
    """
    rows: list[Row] = []
    for name in scenario_names:
        for seed in seeds:
            env = RecoveryEnv(SCENARIOS[name], seed=seed)
            episodes = env.reset()
            observations = {ep.payment_id: env.observe(ep) for ep in episodes}
            store = _SimStore(env, observations)
            extractor = FeatureExtractor(store)  # type: ignore[arg-type]
            for ep in episodes:
                store.use(observations[ep.payment_id])
                features = await extractor.extract(_as_payment(ep))
                rows.append(Row(features=features, cause=ep.true_cause,
                                scenario=name, seed=seed))
    return rows


def fit_encoder() -> OrdinalEncoder:
    """An encoder over the declared vocabulary, not over whatever the sample happened to contain.

    Fitting on observed data would give a different code space every time the dataset
    changes, and a value absent from one training run would be `-1` at serve time even
    though the model has a column for it.
    """
    encoder = OrdinalEncoder(
        categories=[list(v) for v in CATEGORICAL_FEATURES.values()],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit([[cats[0] for cats in CATEGORICAL_FEATURES.values()]])
    return encoder


ERROR_FIELD_COLUMNS: tuple[str, ...] = ("error_source", "error_step", "error_reason")
"""The three fields the Bayes bound in `sim.emission.label_ambiguity` is computed over.

Named here so the ablation that scores a model on these alone uses exactly the columns the
bound assumes, rather than a hand-picked subset that would make the comparison meaningless.
"""


def to_matrix(rows: list[Row], encoder: OrdinalEncoder) -> tuple[np.ndarray, np.ndarray]:
    """Encode rows into `(X, y)` in `ALL_FEATURE_NAMES` order.

    The column order matters more than it looks: `classifier._prepare_features` builds a
    single serving row in this same order, and SHAP labels its outputs by position. A
    mismatch here would be silent — the model would still predict, just from the wrong
    columns.
    """
    dicts = [r.features.to_dict() for r in rows]
    cat = encoder.transform([[d[c] for c in CATEGORICAL_FEATURES] for d in dicts])
    num = np.array([
        [d[c] for c in NUMERICAL_FEATURES] + [int(d[c]) for c in BOOLEAN_FEATURES]
        for d in dicts
    ], dtype=float)
    X = np.hstack([cat, num])
    y = np.array([r.cause for r in rows])
    return X, y
