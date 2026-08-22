"""LLM Reasoner — Gemini Flash for complex/high-value recovery cases."""

from google import genai
from google.genai.types import GenerateContentConfig
from app.config import get_settings
from app.models.schemas import (
    FailedPayment, ClassificationResult, LLMRecoveryDecision,
    RecoveryDecision, RecoveryStrategy, BankHealthStatus,
)
import structlog
import json

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are a payment recovery specialist at an Indian fintech company (Razorpay).

CONTEXT: A payment has failed. You have the failure details, the ML classifier's prediction (with confidence), and the current bank health data.

TASK: Determine the optimal recovery strategy for this specific payment.

HARD CONSTRAINTS (NEVER VIOLATE — these are compliance requirements):
- Maximum 3 retry attempts per payment
- Maximum 2 customer contacts per day
- No actions between 10 PM and 8 AM IST (quiet hours)
- Never auto-charge amounts above ₹10,000 (100000 paise)
- If uncertain about the best approach, recommend ESCALATE

AVAILABLE STRATEGIES:
- IMMEDIATE_RETRY: Retry the same payment method immediately. Best for transient network issues.
- DELAYED_RETRY: Wait for bank recovery, then retry same method. Best for bank downtime. Specify delay_minutes.
- LINK_SAME_METHOD: Send payment link to customer, suggesting the same method. Best for auth timeout.
- LINK_ALT_METHOD: Send payment link suggesting a different method (e.g., switch from card to UPI). Best for permanent method-specific decline.
- SCHEDULED_RETRY: Schedule retry for a specific optimal time (e.g., after salary credit). Specify delay_minutes.
- ESCALATE: Alert the merchant for manual review. Use when automated recovery is unlikely to succeed.

INDIAN PAYMENT CONTEXT:
- Salary credits typically happen on 1st, 7th, or 15th of month
- UPI has ~99.2% success rate, cards ~85-90%, netbanking ~90-95%
- Bank maintenance windows are typically 12 AM - 6 AM
- Hinglish communication is common and effective

You MUST output valid JSON matching this exact schema:
{
  "reasoning": "Step-by-step analysis of why you chose this strategy",
  "strategy": "One of the 6 strategies above",
  "delay_minutes": 0,
  "preferred_method": "upi or card or netbanking or null",
  "message_tone": "friendly or urgent or informational",
  "confidence": 0.0 to 1.0,
  "risk_factors": ["list of concerns about this approach"]
}"""


class LLMReasoner:
    """
    Uses Gemini Flash for structured reasoning about complex recovery cases.
    
    Triggered when:
    - Classifier confidence < 0.7
    - Payment amount > ₹10,000
    - Customer has conflicting signals
    
    Cost: ~₹0.50 per call | Latency: ~1-2s | Usage: <10% of cases
    """

    def __init__(self):
        settings = get_settings()
        self.client = None
        if settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)

    async def reason(
        self,
        payment: FailedPayment,
        classification: ClassificationResult,
        bank_health: BankHealthStatus,
    ) -> RecoveryDecision:
        """Get LLM reasoning for a complex recovery case."""

        if not self.client:
            logger.warning("llm_reasoner.no_api_key")
            # Fallback to simple rule
            from app.strategy.bandit import FALLBACK_RULES, STRATEGY_DELAYS
            strategy = FALLBACK_RULES.get(classification.root_cause.value, "ESCALATE")
            return RecoveryDecision(
                strategy=RecoveryStrategy(strategy),
                reasoning="LLM unavailable — using rule-based fallback",
                confidence=0.5,
                delay_minutes=STRATEGY_DELAYS.get(strategy, 0),
                decided_by="rule",
            )

        # Build the context prompt
        context = self._build_context(payment, classification, bank_health)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"{SYSTEM_PROMPT}\n\n---\n\nPAYMENT CONTEXT:\n{context}",
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=1000,
                ),
            )

            # Parse structured output
            llm_decision = LLMRecoveryDecision.model_validate_json(response.text)

            # Validate strategy is valid
            strategy = llm_decision.strategy
            if strategy not in [s.value for s in RecoveryStrategy]:
                strategy = "ESCALATE"

            # Enforce hard limits (LLM can hallucinate)
            if payment.amount > 1_000_000 and strategy in ("IMMEDIATE_RETRY",):
                strategy = "LINK_SAME_METHOD"  # Force customer action for > ₹10K

            logger.info(
                "llm_reasoner.success",
                payment_id=payment.payment_id,
                strategy=strategy,
                confidence=llm_decision.confidence,
            )

            return RecoveryDecision(
                strategy=RecoveryStrategy(strategy),
                reasoning=llm_decision.reasoning,
                confidence=llm_decision.confidence,
                delay_minutes=llm_decision.delay_minutes,
                preferred_method=None,
                message_template=llm_decision.message_tone,
                decided_by="llm",
            )

        except Exception as e:
            logger.error("llm_reasoner.error", error=str(e))
            # Fallback
            from app.strategy.bandit import FALLBACK_RULES, STRATEGY_DELAYS
            strategy = FALLBACK_RULES.get(classification.root_cause.value, "ESCALATE")
            return RecoveryDecision(
                strategy=RecoveryStrategy(strategy),
                reasoning=f"LLM error ({str(e)[:100]}) — using rule-based fallback",
                confidence=0.4,
                delay_minutes=STRATEGY_DELAYS.get(strategy, 0),
                decided_by="rule",
            )

    def _build_context(
        self,
        payment: FailedPayment,
        classification: ClassificationResult,
        bank_health: BankHealthStatus,
    ) -> str:
        """Build context string for the LLM."""
        return f"""
Payment Details:
- Payment ID: {payment.payment_id}
- Amount: ₹{payment.amount_rupees:,.2f} ({payment.amount} paise)
- Method: {payment.method.value}
- Bank: {payment.bank or 'unknown'}
- Error: {payment.error_reason} (source: {payment.error_source.value}, step: {payment.error_step})
- Time: {payment.created_at.strftime('%Y-%m-%d %H:%M IST')}
- Is Recurring: {payment.is_recurring}

ML Classifier Prediction:
- Root Cause: {classification.root_cause.value} (confidence: {classification.confidence:.2f})
- Top SHAP features: {', '.join(f'{e.feature}={e.value}' for e in classification.shap_explanations[:3])}

Bank Health:
- {payment.bank} success rate (1h): {bank_health.success_rate_1h:.2f}
- Concurrent failures (1h): {bank_health.failure_count_1h}
- In downtime: {bank_health.is_in_downtime}
- Downtime severity: {bank_health.downtime_severity or 'N/A'}

Current retry count for this payment: 0
Customer contacts today: 0
"""


# Singleton
llm_reasoner = LLMReasoner()
