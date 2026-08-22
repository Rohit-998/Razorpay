"""Circuit Breaker — Per-bank state machine preventing retry storms."""

import time
from app.cache.redis_client import redis_client
from app.models.schemas import CircuitState, CircuitBreakerState
import structlog

logger = structlog.get_logger()


class CircuitBreaker:
    """
    Per-bank circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED
    
    State stored in Redis:
      circuit:{bank_code} → Hash with state, failure_count, timestamps
    
    Transitions:
      CLOSED → OPEN:      failure_count > THRESHOLD in WINDOW
      OPEN → HALF_OPEN:   After RECOVERY_TIMEOUT seconds
      HALF_OPEN → CLOSED: Test retry succeeds
      HALF_OPEN → OPEN:   Test retry fails
    """

    FAILURE_THRESHOLD = 10    # failures to trip
    FAILURE_WINDOW = 300      # 5 minutes
    RECOVERY_TIMEOUT = 300    # 5 minutes before half-open test

    async def can_retry(self, bank_code: str) -> tuple[bool, str]:
        """Check if retries are allowed for this bank."""
        if not bank_code:
            return True, "No bank code, allowing retry"

        state = await self._get_state(bank_code)

        if state.state == CircuitState.CLOSED:
            return True, "Circuit closed — retries allowed"

        elif state.state == CircuitState.OPEN:
            elapsed = time.time() - (state.last_failure or 0)
            if elapsed > self.RECOVERY_TIMEOUT:
                await self._transition(bank_code, CircuitState.HALF_OPEN)
                return True, "Circuit half-open — testing one retry"
            remaining = int(self.RECOVERY_TIMEOUT - elapsed)
            return False, f"Circuit OPEN for {bank_code} — retry in {remaining}s"

        elif state.state == CircuitState.HALF_OPEN:
            return False, "Circuit half-open — test retry in progress"

        return True, "Unknown state, allowing retry"

    async def record_success(self, bank_code: str):
        """Record a successful retry — may close the circuit."""
        if not bank_code:
            return
        
        state = await self._get_state(bank_code)
        if state.state == CircuitState.HALF_OPEN:
            await self._transition(bank_code, CircuitState.CLOSED)
            logger.info("circuit_breaker.closed", bank=bank_code)

    async def record_failure(self, bank_code: str):
        """Record a failed retry — may open the circuit."""
        if not bank_code:
            return

        state = await self._get_state(bank_code)
        now = time.time()

        if state.state == CircuitState.HALF_OPEN:
            await self._transition(bank_code, CircuitState.OPEN)
            logger.warning("circuit_breaker.reopened", bank=bank_code)
            return

        # Increment failure count
        new_count = state.failure_count + 1
        await redis_client.hset(f"circuit:{bank_code}", mapping={
            "failure_count": str(new_count),
            "last_failure": str(now),
        })

        # Check threshold
        if new_count >= self.FAILURE_THRESHOLD:
            # Check if failures are within the window
            window_start = now - self.FAILURE_WINDOW
            if state.last_failure and state.last_failure > window_start:
                await self._transition(bank_code, CircuitState.OPEN)
                logger.warning(
                    "circuit_breaker.opened",
                    bank=bank_code,
                    failure_count=new_count,
                )

    async def _get_state(self, bank_code: str) -> CircuitBreakerState:
        """Get current circuit breaker state from Redis."""
        data = await redis_client.hgetall(f"circuit:{bank_code}")
        if not data:
            return CircuitBreakerState(bank_code=bank_code)

        return CircuitBreakerState(
            bank_code=bank_code,
            state=CircuitState(data.get("state", "CLOSED")),
            failure_count=int(data.get("failure_count", "0")),
            last_failure=float(data.get("last_failure", "0")) or None,
            last_check=float(data.get("last_check", "0")) or None,
        )

    async def _transition(self, bank_code: str, new_state: CircuitState):
        """Transition circuit breaker to a new state."""
        now = time.time()
        mapping = {
            "state": new_state.value,
            "last_check": str(now),
        }
        if new_state == CircuitState.CLOSED:
            mapping["failure_count"] = "0"

        await redis_client.hset(f"circuit:{bank_code}", mapping=mapping)
        await redis_client.expire(f"circuit:{bank_code}", 3600)  # 1h TTL

    async def get_all_states(self) -> list[CircuitBreakerState]:
        """Get all circuit breaker states (for dashboard)."""
        states = []
        async for key in redis_client.scan_iter("circuit:*"):
            bank_code = key.split(":")[1]
            state = await self._get_state(bank_code)
            states.append(state)
        return states


# Singleton
circuit_breaker = CircuitBreaker()
