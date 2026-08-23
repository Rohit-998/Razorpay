"""Pydantic models for payment data, recovery decisions, and audit events."""

from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


# =============================================================
# ENUMS
# =============================================================

class PaymentMethod(str, Enum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class ErrorSource(str, Enum):
    CUSTOMER = "customer"
    GATEWAY = "gateway"
    BUSINESS = "business"
    RAZORPAY = "razorpay"


class RootCause(str, Enum):
    BANK_DOWNTIME = "BANK_DOWNTIME"
    NETWORK_TRANSIENT = "NETWORK_TRANSIENT"
    AUTH_TIMEOUT = "AUTH_TIMEOUT"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    WRONG_CREDENTIALS = "WRONG_CREDENTIALS"
    PERMANENT_DECLINE = "PERMANENT_DECLINE"
    MERCHANT_ERROR = "MERCHANT_ERROR"


class RecoveryStrategy(str, Enum):
    IMMEDIATE_RETRY = "IMMEDIATE_RETRY"
    DELAYED_RETRY = "DELAYED_RETRY"
    LINK_SAME_METHOD = "LINK_SAME_METHOD"
    LINK_ALT_METHOD = "LINK_ALT_METHOD"
    SCHEDULED_RETRY = "SCHEDULED_RETRY"
    ESCALATE = "ESCALATE"


class SessionStatus(str, Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"
    EXCEPTION = "EXCEPTION"


class AttributionType(str, Enum):
    SYSTEM_RECOVERED = "SYSTEM_RECOVERED"
    CUSTOMER_SELF_RECOVERED = "CUSTOMER_SELF_RECOVERED"
    AMBIGUOUS = "AMBIGUOUS"
    NO_SESSION = "NO_SESSION"


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# =============================================================
# PAYMENT MODELS
# =============================================================

class FailedPayment(BaseModel):
    """Normalized payment failure from Razorpay webhook."""
    payment_id: str
    order_id: Optional[str] = None
    amount: int  # in paise
    currency: str = "INR"
    method: PaymentMethod
    bank: Optional[str] = None
    wallet: Optional[str] = None
    vpa: Optional[str] = None
    error_code: str
    error_source: ErrorSource
    error_step: str
    error_reason: str
    error_description: str = ""
    customer_contact: Optional[str] = None
    customer_email: Optional[str] = None
    is_recurring: bool = False
    created_at: datetime
    raw_webhook: Optional[dict] = None

    @property
    def amount_rupees(self) -> float:
        return self.amount / 100

    @property
    def amount_bucket(self) -> str:
        rupees = self.amount_rupees
        if rupees < 100:
            return "micro"
        elif rupees < 1000:
            return "small"
        elif rupees < 10000:
            return "medium"
        elif rupees < 50000:
            return "large"
        return "premium"


# =============================================================
# FEATURE MODELS
# =============================================================

class FeatureVector(BaseModel):
    """17 features across 4 categories for ML classification."""
    # Payment Context (5)
    error_source: str
    error_step: str
    error_reason: str
    payment_method: str
    amount_bucket: str

    # Temporal (5)
    hour_of_day: int
    day_of_week: int
    is_month_end: bool
    is_salary_window: bool
    is_maintenance_window: bool

    # Bank Health (4)
    bank_success_rate_1h: float = 0.95
    bank_failure_count_1h: int = 0
    bank_is_in_downtime: bool = False
    method_success_rate_1h: float = 0.95

    # Customer History (3)
    customer_success_rate_30d: float = 0.9
    customer_failure_count_7d: int = 0
    customer_recovery_response: float = 0.5

    def to_dict(self) -> dict:
        return self.model_dump()


# =============================================================
# CLASSIFICATION MODELS
# =============================================================

class ShapExplanation(BaseModel):
    """Single SHAP feature contribution."""
    feature: str
    value: float | str | bool
    shap_value: float
    direction: str  # "→ CLASS_NAME"


class ClassificationResult(BaseModel):
    """Output of the XGBoost root cause classifier."""
    root_cause: RootCause
    confidence: float
    all_probabilities: dict[str, float]
    shap_explanations: list[ShapExplanation]
    explanation_summary: str
    inference_time_ms: float


# =============================================================
# STRATEGY / RECOVERY MODELS
# =============================================================

class RecoveryDecision(BaseModel):
    """Decision output from strategy selector (bandit/LLM/rules)."""
    strategy: RecoveryStrategy
    reasoning: str
    confidence: float
    delay_minutes: int = 0
    preferred_method: Optional[PaymentMethod] = None
    message_template: str = "default"
    decided_by: str  # "bandit", "llm", "rule"
    decided_at: datetime = Field(default_factory=datetime.utcnow)


class LLMRecoveryDecision(BaseModel):
    """Structured output from Gemini Flash LLM reasoning."""
    reasoning: str
    strategy: str
    delay_minutes: int = 0
    preferred_method: Optional[str] = None
    message_tone: str = "friendly"
    confidence: float = 0.5
    risk_factors: list[str] = []


class RecoveryOutcome(BaseModel):
    """Result of a recovery execution attempt."""
    status: str  # "RECOVERED", "FAILED", "SCHEDULED", "AWAITING_CUSTOMER", "ESCALATED"
    action_taken: str
    link_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    error: Optional[str] = None


class RecoverySession(BaseModel):
    """Full recovery session state."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    payment_id: str
    status: SessionStatus = SessionStatus.OPEN
    root_cause: Optional[RootCause] = None
    root_cause_confidence: Optional[float] = None
    strategy: Optional[RecoveryStrategy] = None
    decided_by: Optional[str] = None
    retry_count: int = 0
    contact_count: int = 0
    amount_recovered: int = 0
    attribution: Optional[AttributionType] = None
    shap_explanation: Optional[list[dict]] = None
    llm_reasoning: Optional[str] = None
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None


# =============================================================
# AUDIT MODELS
# =============================================================

class AuditEvent(BaseModel):
    """Immutable audit event — append-only."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    recovery_session_id: str
    payment_id: str
    event_type: str
    event_data: dict = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================
# BANK HEALTH MODELS
# =============================================================

class BankHealthStatus(BaseModel):
    """Current health status of a bank."""
    bank_code: str
    is_healthy: bool
    success_rate_1h: float = 0.95
    failure_count_1h: int = 0
    is_in_downtime: bool = False
    downtime_severity: Optional[str] = None
    recommendation: str = "RETRY_NOW"


class CircuitBreakerState(BaseModel):
    """Circuit breaker state for a bank."""
    bank_code: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure: Optional[float] = None
    last_check: Optional[float] = None


# =============================================================
# COMPLIANCE MODELS
# =============================================================

class ComplianceCheck(BaseModel):
    """Result of compliance engine check."""
    approved: bool
    blocked_by: list[str] = []
    recommendation: Optional[str] = None


# =============================================================
# API RESPONSE MODELS
# =============================================================

class DashboardStats(BaseModel):
    """Summary stats for dashboard."""
    total_failed: int = 0
    total_attempted: int = 0
    total_recovered: int = 0
    total_exceptions: int = 0
    total_failed_amount: int = 0  # paise
    total_recovered_amount: int = 0  # paise
    recovery_rate: float = 0.0
    avg_recovery_time_minutes: float = 0.0


class BatchReport(BaseModel):
    """Full batch run results."""
    batch_size: int
    total_failed_amount: int
    recovery_attempted: int
    recovery_successful: int
    recovery_rate: float
    amount_recovered: int
    classifier_accuracy: float
    classifier_f1: float
    avg_recovery_time_minutes: float
    false_positive_count: int
    false_positive_cost: float
    exceptions: list[dict]
    per_class_breakdown: dict[str, dict]
