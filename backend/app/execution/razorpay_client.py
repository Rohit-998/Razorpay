"""Razorpay API Client for executing recovery actions."""

import httpx
from app.config import get_settings
import structlog
from typing import Optional

logger = structlog.get_logger()


class RazorpayClient:
    """Client for interacting with Razorpay API."""

    def __init__(self):
        self.settings = get_settings()
        self.auth = (self.settings.razorpay_key_id, self.settings.razorpay_key_secret)
        self.base_url = "https://api.razorpay.com/v1"

    async def create_payment_link(
        self,
        amount: int,
        currency: str,
        reference_id: str,
        description: str,
        customer_contact: str,
        customer_email: str,
        expire_by: Optional[int] = None,
    ) -> Optional[dict]:
        """Create a Payment Link for recovery. `None` means no link exists.

        Unconfigured credentials used to return `{"id": "plink_mock123", ...}` — a constant.
        That is a worse failure than returning nothing, and quietly. The link id is what
        attribution is built on: `payment_link.paid` names the link that was paid, which is
        how a recovery becomes `SYSTEM_RECOVERED` rather than `AMBIGUOUS`. A fake id gets
        written to the audit trail as though it were real, can never be paid, and is shared by
        every payment in the batch — so the one field that carries causation would have been
        both counterfeit and non-unique.

        Returning `None` is already handled: the executor logs an execution error and reports
        the action as not taken, so nothing is credited and no contact is recorded.
        """
        if not self.settings.razorpay_key_id:
            logger.error(
                "razorpay.not_configured",
                reference_id=reference_id,
                detail=(
                    "RAZORPAY_KEY_ID is unset, so no payment link can be created. Refusing "
                    "rather than returning a placeholder id, which attribution would treat "
                    "as a real link."
                ),
            )
            return None

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

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/payment_links",
                    json=payload,
                    auth=self.auth,
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                logger.info("razorpay.payment_link_created", link_id=data["id"])
                return data
        except httpx.HTTPError as e:
            logger.error("razorpay.payment_link_failed", error=str(e), reference_id=reference_id)
            return None

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
