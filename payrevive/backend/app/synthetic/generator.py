"""Synthetic Payment Failure Generator — realistic Indian payment patterns."""

import random
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4
from app.models.schemas import FailedPayment, PaymentMethod, ErrorSource
import structlog

logger = structlog.get_logger()

# =============================================================
# INDIAN PAYMENT LANDSCAPE PATTERNS
# =============================================================

BANKS = {
    "SBIN": 0.22,  # SBI
    "HDFC": 0.18,
    "ICIC": 0.15,  # ICICI
    "UTIB": 0.12,  # Axis
    "KKBK": 0.08,  # Kotak
    "BARB": 0.07,  # Bank of Baroda
    "PUNB": 0.06,  # PNB
    "YESB": 0.05,  # Yes Bank
    "IOBA": 0.04,  # IOB
    "CNRB": 0.03,  # Canara
}

UPI_APPS = {
    "gpay": 0.35,
    "phonepe": 0.30,
    "paytm": 0.20,
    "bhim": 0.10,
    "others": 0.05,
}

AMOUNT_RANGES = {
    "micro": (1000, 9900, 0.15),        # ₹10-99
    "small": (10000, 99900, 0.30),      # ₹100-999
    "medium": (100000, 999900, 0.35),   # ₹1,000-9,999
    "large": (1000000, 5000000, 0.15),  # ₹10,000-50,000
    "premium": (5000100, 20000000, 0.05),  # ₹50,001-2,00,000
}

# Root cause → (error_code, error_source, error_step, error_reason, method_bias)
FAILURE_PATTERNS = {
    "BANK_DOWNTIME": {
        "weight": 0.25,
        "error_code": "GATEWAY_ERROR",
        "error_source": "gateway",
        "error_step": "payment_initiation",
        "error_reasons": ["gateway_technical_error", "timeout"],
        "recoverable": True,
        "method_bias": {"upi": 0.4, "card": 0.3, "netbanking": 0.25, "wallet": 0.05},
    },
    "NETWORK_TRANSIENT": {
        "weight": 0.15,
        "error_code": "GATEWAY_ERROR",
        "error_source": "gateway",
        "error_step": "payment_initiation",
        "error_reasons": ["network_error", "timeout"],
        "recoverable": True,
        "method_bias": {"upi": 0.5, "card": 0.25, "netbanking": 0.15, "wallet": 0.10},
    },
    "AUTH_TIMEOUT": {
        "weight": 0.20,
        "error_code": "BAD_REQUEST_ERROR",
        "error_source": "customer",
        "error_step": "payment_authentication",
        "error_reasons": ["authentication_failed", "timeout", "payment_cancelled"],
        "recoverable": True,
        "method_bias": {"card": 0.45, "netbanking": 0.30, "upi": 0.20, "wallet": 0.05},
    },
    "INSUFFICIENT_FUNDS": {
        "weight": 0.20,
        "error_code": "BAD_REQUEST_ERROR",
        "error_source": "customer",
        "error_step": "payment_processing",
        "error_reasons": ["insufficient_funds"],
        "recoverable": True,
        "method_bias": {"card": 0.35, "upi": 0.35, "netbanking": 0.20, "wallet": 0.10},
    },
    "WRONG_CREDENTIALS": {
        "weight": 0.05,
        "error_code": "BAD_REQUEST_ERROR",
        "error_source": "customer",
        "error_step": "payment_authentication",
        "error_reasons": ["authentication_failed"],
        "recoverable": False,
        "method_bias": {"card": 0.40, "netbanking": 0.35, "upi": 0.20, "wallet": 0.05},
    },
    "PERMANENT_DECLINE": {
        "weight": 0.10,
        "error_code": "BAD_REQUEST_ERROR",
        "error_source": "gateway",
        "error_step": "payment_processing",
        "error_reasons": ["card_blocked", "invalid_card", "mandate_expired"],
        "recoverable": False,
        "method_bias": {"card": 0.60, "upi": 0.15, "netbanking": 0.15, "wallet": 0.10},
    },
    "MERCHANT_ERROR": {
        "weight": 0.05,
        "error_code": "BAD_REQUEST_ERROR",
        "error_source": "business",
        "error_step": "payment_initiation",
        "error_reasons": ["bank_not_enabled", "other"],
        "recoverable": False,
        "method_bias": {"netbanking": 0.40, "card": 0.30, "upi": 0.20, "wallet": 0.10},
    },
}

CUSTOMER_PERSONAS = {
    "salary_regular": {
        "weight": 0.40,
        "preferred_method": "upi",
        "avg_amount_bucket": "small",
        "response_rate": 0.7,
        "failure_bias": {"INSUFFICIENT_FUNDS": 2.0, "AUTH_TIMEOUT": 0.5},
    },
    "premium_shopper": {
        "weight": 0.20,
        "preferred_method": "card",
        "avg_amount_bucket": "large",
        "response_rate": 0.5,
        "failure_bias": {"AUTH_TIMEOUT": 2.0, "PERMANENT_DECLINE": 1.5},
    },
    "occasional_user": {
        "weight": 0.25,
        "preferred_method": "upi",
        "avg_amount_bucket": "micro",
        "response_rate": 0.3,
        "failure_bias": {"WRONG_CREDENTIALS": 2.0, "NETWORK_TRANSIENT": 1.5},
    },
    "business_buyer": {
        "weight": 0.15,
        "preferred_method": "netbanking",
        "avg_amount_bucket": "premium",
        "response_rate": 0.8,
        "failure_bias": {"BANK_DOWNTIME": 1.5, "MERCHANT_ERROR": 2.0},
    },
}

INDIAN_NAMES = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Gupta",
    "Vikram Singh", "Ananya Reddy", "Rohan Joshi", "Meera Nair",
    "Arjun Malhotra", "Kavita Desai", "Siddharth Iyer", "Pooja Verma",
    "Rajesh Khanna", "Divya Menon", "Kunal Agarwal", "Nisha Rao",
    "Aditya Chauhan", "Ritu Saxena", "Manish Tiwari", "Swati Pillai",
    "Deepak Pandey", "Ankita Jain", "Suresh Yadav", "Neha Kapoor",
    "Karthik Srinivasan", "Pallavi Bhat", "Gaurav Mishra", "Shreya Das",
    "Nikhil Banerjee", "Lakshmi Nambiar",
]


class SyntheticDataGenerator:
    """Generate realistic Indian payment failure data for demo/training."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def generate_customers(self, n: int = 50) -> list[dict]:
        """Generate synthetic customer profiles."""
        customers = []
        personas = list(CUSTOMER_PERSONAS.keys())
        persona_weights = [CUSTOMER_PERSONAS[p]["weight"] for p in personas]

        for i in range(n):
            persona = random.choices(personas, weights=persona_weights)[0]
            persona_data = CUSTOMER_PERSONAS[persona]
            name = random.choice(INDIAN_NAMES)

            customers.append({
                "customer_id": f"cust_{uuid4().hex[:8]}",
                "name": name,
                "phone": f"+91{random.randint(7000000000, 9999999999)}",
                "email": f"{name.lower().replace(' ', '.')}@example.com",
                "persona": persona,
                "preferred_method": persona_data["preferred_method"],
            })

        return customers

    def generate_failures(
        self,
        n: int = 150,
        customers: list[dict] | None = None,
        duration_days: int = 7,
    ) -> tuple[list[FailedPayment], list[str]]:
        """
        Generate n synthetic failed payments with realistic patterns.
        
        Returns: (list of FailedPayment, list of root_cause labels)
        """
        if customers is None:
            customers = self.generate_customers()

        base_time = datetime.utcnow() - timedelta(days=duration_days)
        payments = []
        labels = []

        # Generate bank downtime episodes (correlated failures)
        bank_episodes = self._generate_bank_episodes(duration_days)

        for i in range(n):
            # Pick root cause
            root_cause = self._pick_root_cause(i, n, base_time, duration_days)

            # Pick customer
            customer = random.choice(customers)
            persona = CUSTOMER_PERSONAS.get(customer["persona"], CUSTOMER_PERSONAS["salary_regular"])

            # Pick time (with realistic patterns)
            payment_time = self._pick_time(base_time, duration_days, root_cause)

            # Pick method (biased by root cause and persona)
            method = self._pick_method(root_cause, persona)

            # Pick amount (biased by persona)
            amount = self._pick_amount(persona)

            # Pick bank
            bank = self._pick_bank()

            # Get failure details
            pattern = FAILURE_PATTERNS[root_cause]
            error_reason = random.choice(pattern["error_reasons"])

            # Adjust for bank downtime episodes
            if root_cause == "BANK_DOWNTIME":
                # Place it during a downtime episode
                for ep in bank_episodes:
                    if ep["bank"] == bank:
                        offset = random.uniform(0, ep["duration_min"] * 60)
                        payment_time = ep["start"] + timedelta(seconds=offset)
                        break

            payment = FailedPayment(
                payment_id=f"pay_{uuid4().hex[:12]}",
                order_id=f"order_{uuid4().hex[:12]}",
                amount=amount,
                currency="INR",
                method=PaymentMethod(method),
                bank=bank,
                wallet="paytm" if method == "wallet" else None,
                vpa=f"{customer['name'].lower().replace(' ', '')}@ok{bank.lower()}" if method == "upi" else None,
                error_code=pattern["error_code"],
                error_source=ErrorSource(pattern["error_source"]),
                error_step=pattern["error_step"],
                error_reason=error_reason,
                error_description=self._get_error_description(root_cause, error_reason),
                customer_contact=customer["phone"],
                customer_email=customer["email"],
                is_recurring=random.random() < 0.1,
                created_at=payment_time,
            )

            payments.append(payment)
            labels.append(root_cause)

        # Sort by time (important for temporal split)
        paired = list(zip(payments, labels))
        paired.sort(key=lambda x: x[0].created_at)
        payments, labels = zip(*paired) if paired else ([], [])

        logger.info(
            "synthetic_data.generated",
            count=len(payments),
            duration_days=duration_days,
            root_cause_distribution={
                rc: labels.count(rc) for rc in set(labels)
            },
        )

        return list(payments), list(labels)

    def _pick_root_cause(self, idx: int, total: int, base_time: datetime, days: int) -> str:
        """Pick root cause with weighted distribution."""
        causes = list(FAILURE_PATTERNS.keys())
        weights = [FAILURE_PATTERNS[c]["weight"] for c in causes]
        return random.choices(causes, weights=weights)[0]

    def _pick_time(self, base_time: datetime, duration_days: int, root_cause: str) -> datetime:
        """Pick payment time with realistic patterns."""
        offset_seconds = random.uniform(0, duration_days * 86400)
        dt = base_time + timedelta(seconds=offset_seconds)

        # Bank downtime bias: more likely during maintenance windows (12-6 AM)
        if root_cause == "BANK_DOWNTIME" and random.random() < 0.4:
            dt = dt.replace(hour=random.randint(0, 5))

        # Insufficient funds bias: more likely end of month
        if root_cause == "INSUFFICIENT_FUNDS" and random.random() < 0.5:
            dt = dt.replace(day=min(random.randint(25, 30), 28))

        return dt

    def _pick_method(self, root_cause: str, persona: dict) -> str:
        """Pick payment method biased by root cause and persona."""
        bias = FAILURE_PATTERNS[root_cause]["method_bias"]
        methods = list(bias.keys())
        weights = list(bias.values())

        # Slightly adjust for persona preference
        preferred = persona.get("preferred_method", "upi")
        for i, m in enumerate(methods):
            if m == preferred:
                weights[i] *= 1.3

        return random.choices(methods, weights=weights)[0]

    def _pick_amount(self, persona: dict) -> int:
        """Pick amount biased by persona."""
        bucket_weights = {
            "micro": 0.15, "small": 0.30, "medium": 0.35,
            "large": 0.15, "premium": 0.05,
        }

        # Adjust for persona
        preferred_bucket = persona.get("avg_amount_bucket", "small")
        for k in bucket_weights:
            if k == preferred_bucket:
                bucket_weights[k] *= 2.0

        buckets = list(bucket_weights.keys())
        weights = list(bucket_weights.values())
        bucket = random.choices(buckets, weights=weights)[0]

        low, high, _ = AMOUNT_RANGES[bucket]
        return random.randint(low, high)

    def _pick_bank(self) -> str:
        """Pick bank with realistic market share distribution."""
        banks = list(BANKS.keys())
        weights = list(BANKS.values())
        return random.choices(banks, weights=weights)[0]

    def _generate_bank_episodes(self, duration_days: int) -> list[dict]:
        """Generate correlated bank downtime episodes."""
        episodes = []
        base = datetime.utcnow() - timedelta(days=duration_days)

        for bank in BANKS:
            # Each bank has 1-3 downtime episodes over the period
            n_episodes = random.randint(1, 3)
            for _ in range(n_episodes):
                start_offset = random.uniform(0, duration_days * 86400)
                start = base + timedelta(seconds=start_offset)
                duration = random.randint(15, 120)  # 15-120 minutes

                episodes.append({
                    "bank": bank,
                    "start": start,
                    "duration_min": duration,
                    "severity": random.choice(["high", "medium", "low"]),
                })

        return episodes

    def _get_error_description(self, root_cause: str, reason: str) -> str:
        """Generate human-readable error description."""
        descriptions = {
            "BANK_DOWNTIME": "Payment failed due to temporary bank system unavailability. Please try again after some time.",
            "NETWORK_TRANSIENT": "Payment could not be processed due to a network connectivity issue. Please retry.",
            "AUTH_TIMEOUT": "Payment authentication timed out. The customer did not complete OTP/3DS verification in time.",
            "INSUFFICIENT_FUNDS": "Payment declined due to insufficient balance in the customer's account.",
            "WRONG_CREDENTIALS": "Payment authentication failed. The customer entered incorrect credentials.",
            "PERMANENT_DECLINE": "Payment was permanently declined by the issuing bank.",
            "MERCHANT_ERROR": "Payment could not be initiated due to a merchant configuration issue.",
        }
        return descriptions.get(root_cause, f"Payment failed: {reason}")


# Singleton
generator = SyntheticDataGenerator()
