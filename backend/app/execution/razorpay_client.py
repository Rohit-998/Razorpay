"""Razorpay API Client for executing recovery actions."""

import asyncio
import time
import httpx
from app.config import get_settings
import structlog
from typing import Optional

logger = structlog.get_logger()

MIN_INTERVAL_SECONDS = 0.4
"""The smallest gap between two writes to the Razorpay API.

Not a guess at their published limit — a floor that keeps a batch from arriving as a burst. The
batch worker processes payments in a tight loop, so forty link creations went out inside four
seconds and Razorpay answered `429` to every one after the first few. Nothing crashed: the
client returned `None`, the executor logged `Failed to create payment link`, and the run
reported success. The cost was invisible and total — `PAYMENT_LINK_SENT` is the only audit event
that can produce a `SYSTEM_RECOVERED` verdict, so the throttle silently made the one provable
recovery path unreachable and left the dashboard's headline at ₹0.
"""

MAX_ATTEMPTS = 4
"""Attempts per link, including the first. Only `429` and 5xx are retried — a `400` is a
malformed request and sending it again just spends another slot on the same rejection."""

_RETRYABLE = (429, 500, 502, 503, 504)


class RazorpayClient:
    """Client for interacting with Razorpay API."""

    def __init__(self):
        self.settings = get_settings()
        self.auth = (self.settings.razorpay_key_id, self.settings.razorpay_key_secret)
        self.base_url = "https://api.razorpay.com/v1"
        self._gate = asyncio.Lock()
        self._last_write = 0.0

    async def _space_out(self) -> None:
        """Hold the caller until `MIN_INTERVAL_SECONDS` has passed since the last write.

        Under a lock, because the ARQ worker can have several jobs in flight and a per-task
        sleep would let them all wake into the same millisecond.
        """
        async with self._gate:
            wait = MIN_INTERVAL_SECONDS - (time.monotonic() - self._last_write)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_write = time.monotonic()

    async def create_payment_link(
        self,
        amount: int,
        currency: str,
        reference_id: str,
        description: str,
        customer_contact: str,
        customer_email: str,
        expire_by: Optional[int] = None,
    ) -> tuple[Optional[dict], Optional[str]]:
        """Create a Payment Link for recovery. `(None, reason)` means no link exists.

        Unconfigured credentials used to return `{"id": "plink_mock123", ...}` — a constant.
        That is a worse failure than returning nothing, and quietly. The link id is what
        attribution is built on: `payment_link.paid` names the link that was paid, which is
        how a recovery becomes `SYSTEM_RECOVERED` rather than `AMBIGUOUS`. A fake id gets
        written to the audit trail as though it were real, can never be paid, and is shared by
        every payment in the batch — so the one field that carries causation would have been
        both counterfeit and non-unique.

        Returning `None` is already handled: the executor logs an execution error and reports
        the action as not taken, so nothing is credited and no contact is recorded. What it did
        not have was the *reason*, so a rate limit and a rejected phone number both reached the
        exception ledger as the same four words. The reason now travels with the refusal,
        because "Failed to create payment link" is a line a reviewer is meant to be able to act
        on.
        """
        if not self.settings.razorpay_key_id:
            reason = (
                "RAZORPAY_KEY_ID is unset, so no payment link can be created. Refusing rather "
                "than returning a placeholder id, which attribution would treat as a real link."
            )
            logger.error("razorpay.not_configured", reference_id=reference_id, detail=reason)
            return None, reason

        payload = {
            "amount": amount,
            "currency": currency,
            "accept_partial": False,
            "reference_id": reference_id,
            "description": description,
            "customer": {
                "contact": customer_contact,
                "email": customer_email
            },
            "notify": {
                "sms": True,
                "email": True
            },
            "reminder_enable": True
        }

        if expire_by:
            payload["expire_by"] = expire_by

        last_reason = "no attempt was made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            await self._space_out()
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/payment_links",
                        json=payload,
                        auth=self.auth,
                        timeout=10.0,
                    )
            except httpx.HTTPError as e:
                last_reason = f"could not reach the Razorpay API: {e}"
                logger.warning(
                    "razorpay.payment_link_unreachable",
                    reference_id=reference_id,
                    attempt=attempt,
                    error=str(e),
                )
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
                continue

            if response.status_code < 400:
                data = response.json()
                logger.info(
                    "razorpay.payment_link_created", link_id=data["id"], attempt=attempt
                )
                return data, None

            detail = self._describe(response)
            if response.status_code not in _RETRYABLE:
                logger.error(
                    "razorpay.payment_link_rejected",
                    reference_id=reference_id,
                    status=response.status_code,
                    detail=detail,
                )
                return None, f"Razorpay rejected the link ({response.status_code}): {detail}"

            last_reason = f"Razorpay answered {response.status_code}: {detail}"
            if attempt == MAX_ATTEMPTS:
                break
            # `Retry-After` when they name a delay, exponential backoff when they do not.
            named = response.headers.get("Retry-After")
            pause = float(named) if named and named.isdigit() else min(2 ** (attempt - 1), 8)
            logger.warning(
                "razorpay.payment_link_throttled",
                reference_id=reference_id,
                status=response.status_code,
                attempt=attempt,
                retry_in_seconds=pause,
            )
            await asyncio.sleep(pause)

        logger.error(
            "razorpay.payment_link_failed",
            reference_id=reference_id,
            attempts=MAX_ATTEMPTS,
            reason=last_reason,
        )
        return None, f"{last_reason} (gave up after {MAX_ATTEMPTS} attempts)"

    @staticmethod
    def _describe(response: httpx.Response) -> str:
        """Razorpay's own sentence about the refusal, when the body carries one."""
        try:
            error = (response.json() or {}).get("error") or {}
        except ValueError:
            return response.text[:200] or response.reason_phrase
        return error.get("description") or error.get("code") or response.reason_phrase

    async def fetch_downtimes(self) -> Optional[list[dict]]:
        """Current payment downtimes, normalised to `{bank, method, severity, status}`.

        `None` and `[]` mean different things and the caller depends on the difference:
        `None` is "we could not ask", `[]` is "we asked and nothing is down". Collapsing them
        would let a failed poll clear every outage flag in the feature store and tell the
        policy the world is healthy at precisely the moment it cannot know.

        Razorpay reports downtime per instrument, so `instrument.bank` is present for
        netbanking and card outages and absent for a UPI handle problem. The bank code is what
        the feature store is keyed on, so entries without one are dropped rather than filed
        under a placeholder.
        """
        if not self.settings.razorpay_key_id:
            logger.warning("razorpay.downtime_unconfigured")
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/payments/downtimes", auth=self.auth, timeout=10.0
                )
                response.raise_for_status()
                items = response.json().get("items") or []
        except (httpx.HTTPError, ValueError) as e:
            logger.error("razorpay.downtime_fetch_failed", error=str(e))
            return None

        entries = []
        for item in items:
            # `resolved` outages stay in the feed with a status; only the live ones should
            # move the policy, and the caller clears the flag for anything no longer here.
            if item.get("status") == "resolved":
                continue
            bank = (item.get("instrument") or {}).get("bank")
            if not bank:
                continue
            entries.append({
                "bank": bank,
                "method": item.get("method"),
                "severity": item.get("severity"),
                "status": item.get("status"),
            })
        logger.info("razorpay.downtimes_fetched", count=len(entries))
        return entries


# Singleton
razorpay_client = RazorpayClient()
