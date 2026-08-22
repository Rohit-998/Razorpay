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
        """Create a Payment Link for recovery."""
        if not self.settings.razorpay_key_id:
            logger.warning("razorpay.mock_create_link", reference_id=reference_id)
            return {"id": "plink_mock123", "short_url": "https://rzp.io/i/mock"}

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


# Singleton
razorpay_client = RazorpayClient()
