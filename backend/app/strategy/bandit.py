"""Thompson Sampling Contextual Bandit for Recovery Strategy Selection."""

import numpy as np
from app.cache.redis_client import redis_client
from app.models.schemas import RootCause, RecoveryStrategy, FeatureVector, RecoveryDecision
from datetime import datetime
import structlog

logger = structlog.get_logger()

# Fallback rules for cold start (before bandit has enough data)
FALLBACK_RULES: dict[str, str] = {
    "BANK_DOWNTIME": "DELAYED_RETRY",
    "NETWORK_TRANSIENT": "IMMEDIATE_RETRY",
    "AUTH_TIMEOUT": "LINK_SAME_METHOD",
    "INSUFFICIENT_FUNDS": "SCHEDULED_RETRY",
    "WRONG_CREDENTIALS": "LINK_SAME_METHOD",
    "PERMANENT_DECLINE": "LINK_ALT_METHOD",
    "MERCHANT_ERROR": "ESCALATE",
}

# Default delays per strategy (minutes)
STRATEGY_DELAYS: dict[str, int] = {
    "IMMEDIATE_RETRY": 0,
    "DELAYED_RETRY": 30,
    "LINK_SAME_METHOD": 5,
    "LINK_ALT_METHOD": 5,
    "SCHEDULED_RETRY": 1440,  # 24 hours (next day / salary day)
    "ESCALATE": 0,
}

# Minimum observations per context before bandit takes over from rules
MIN_OBSERVATIONS = 15


class RecoveryBandit:
    """
    Contextual Multi-Armed Bandit using Thompson Sampling.
    
    Arms: 6 recovery strategies
    Context: root_cause + payment_method + amount_bucket
    Reward: Binary (1 = recovered, 0 = not)
    
    Posterior parameters stored in Redis:
      bandit:{context_key}:{strategy} → Hash {alpha, beta}
    
    Uses Beta(alpha, beta) posterior — conjugate prior for Bernoulli rewards.
    Thompson Sampling: sample from each arm's posterior, pick the highest.
    """

    async def select_strategy(
        self,
        root_cause: str,
        features: FeatureVector,
        available_strategies: list[str] | None = None,
    ) -> RecoveryDecision:
        """Select the best recovery strategy for this context."""

        context_key = self._make_context_key(root_cause, features)

        if available_strategies is None:
            available_strategies = [s.value for s in RecoveryStrategy]

        # Check if we have enough observations for this context
        total_obs = await self._get_total_observations(context_key)

        if total_obs < MIN_OBSERVATIONS:
            # Cold start — use rule-based fallback
            strategy = FALLBACK_RULES.get(root_cause, "ESCALATE")
            logger.info(
                "bandit.cold_start",
                context=context_key,
                observations=total_obs,
                strategy=strategy,
            )
            return RecoveryDecision(
                strategy=RecoveryStrategy(strategy),
                reasoning=f"Cold start ({total_obs}/{MIN_OBSERVATIONS} observations). Using rule-based fallback for {root_cause}.",
                confidence=0.5,
                delay_minutes=STRATEGY_DELAYS.get(strategy, 0),
                decided_by="rule",
            )

        # Thompson Sampling: sample from each arm's posterior
        best_strategy = None
        best_sample = -1
        samples = {}

        for strategy in available_strategies:
            params = await self._get_posterior(context_key, strategy)
            # Sample from Beta distribution
            sample = np.random.beta(params["alpha"], params["beta"])
            samples[strategy] = round(sample, 4)

            if sample > best_sample:
                best_sample = sample
                best_strategy = strategy

        logger.info(
            "bandit.selected",
            context=context_key,
            strategy=best_strategy,
            samples=samples,
            observations=total_obs,
        )

        return RecoveryDecision(
            strategy=RecoveryStrategy(best_strategy),
            reasoning=f"Thompson Sampling selected {best_strategy} (sample={best_sample:.3f}) from context {context_key} ({total_obs} observations).",
            confidence=best_sample,
            delay_minutes=STRATEGY_DELAYS.get(best_strategy, 0),
            decided_by="bandit",
        )

    async def update(
        self,
        root_cause: str,
        features: FeatureVector,
        strategy: str,
        reward: float,
    ):
        """Update posterior after observing recovery outcome."""
        context_key = self._make_context_key(root_cause, features)
        redis_key = f"bandit:{context_key}:{strategy}"

        params = await self._get_posterior(context_key, strategy)

        # Update Beta posterior
        if reward > 0:
            params["alpha"] += 1
        else:
            params["beta"] += 1

        await redis_client.hset(redis_key, mapping={
            "alpha": str(params["alpha"]),
            "beta": str(params["beta"]),
        })

        logger.info(
            "bandit.updated",
            context=context_key,
            strategy=strategy,
            reward=reward,
            alpha=params["alpha"],
            beta=params["beta"],
        )

    async def get_learning_data(self) -> dict:
        """Get bandit learning state for dashboard visualization."""
        data = {}
        async for key in redis_client.scan_iter("bandit:*"):
            parts = key.split(":")
            if len(parts) == 3:
                _, context, strategy = parts
                params = await redis_client.hgetall(key)
                if context not in data:
                    data[context] = {}
                alpha = float(params.get("alpha", "1"))
                beta_val = float(params.get("beta", "1"))
                total = alpha + beta_val - 2  # subtract priors
                data[context][strategy] = {
                    "alpha": alpha,
                    "beta": beta_val,
                    "mean": round(alpha / (alpha + beta_val), 4),
                    "trials": int(total),
                }
        return data

    def _make_context_key(self, root_cause: str, features: FeatureVector) -> str:
        """Create context key from root cause + method + amount bucket."""
        return f"{root_cause}_{features.payment_method}_{features.amount_bucket}"

    async def _get_posterior(self, context_key: str, strategy: str) -> dict:
        """Get Beta posterior parameters from Redis."""
        redis_key = f"bandit:{context_key}:{strategy}"
        data = await redis_client.hgetall(redis_key)
        return {
            "alpha": float(data.get("alpha", "1")),
            "beta": float(data.get("beta", "1")),
        }

    async def _get_total_observations(self, context_key: str) -> int:
        """Get total observations across all strategies for a context."""
        total = 0
        for strategy in RecoveryStrategy:
            params = await self._get_posterior(context_key, strategy.value)
            total += (params["alpha"] + params["beta"] - 2)  # subtract priors
        return int(total)


# Singleton
bandit = RecoveryBandit()
