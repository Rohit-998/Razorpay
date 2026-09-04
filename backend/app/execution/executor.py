"""Recovery Executor — coordinates strategy execution with Razorpay."""

from app.models.schemas import FailedPayment, RecoveryDecision, RecoveryStrategy
from app.audit.event_store import event_store
from app.execution.razorpay_client import razorpay_client
from app.execution.compliance import compliance_engine
import structlog
import time

logger = structlog.get_logger()


class RecoveryExecutor:
    """Executes the chosen recovery strategy."""

    async def execute(
        self,
        payment: FailedPayment,
        session_id: str,
        decision: RecoveryDecision,
    ) -> bool:
        """Execute the strategy. Returns True if action was taken successfully."""
        
        strategy = decision.strategy
        logger.info("executor.starting", payment_id=payment.payment_id, strategy=strategy.value)

        # 1. IMMEDIATE_RETRY (We mock auto-retry for the buildathon since we don't have tokenized cards)
        if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
            event_store.log_recovery_attempt(session_id, payment.payment_id, "IMMEDIATE_RETRY_MOCKED")
            # In a real app with tokenization/recurring mandates, we'd hit Razorpay charges API here.
            # For the buildathon, if it's IMMEDIATE_RETRY, we might simulate a success or convert it to a link.
            logger.info("executor.mocked_auto_retry", payment_id=payment.payment_id)
            return True

        # 2. DELAYED_RETRY / SCHEDULED_RETRY
        elif strategy in (RecoveryStrategy.DELAYED_RETRY, RecoveryStrategy.SCHEDULED_RETRY):
            # We would enqueue a task in ARQ with a countdown/eta
            delay = decision.delay_minutes or 30
            event_store.log_recovery_attempt(
                session_id, payment.payment_id, "RETRY_SCHEDULED", {"delay_minutes": delay}
            )
            # await arq_queue.enqueue_job('execute_delayed_retry', payment.payment_id, session_id, _defer_by=delay*60)
            logger.info("executor.scheduled", payment_id=payment.payment_id, delay_minutes=delay)
            return True

        # 3. LINK_SAME_METHOD / LINK_ALT_METHOD
        elif strategy in (RecoveryStrategy.LINK_SAME_METHOD, RecoveryStrategy.LINK_ALT_METHOD):
            description = f"Complete your payment for Order {payment.order_id}"
            if decision.message_template:
                description += f" ({decision.message_template} tone)"

            expire_by = int(time.time()) + (24 * 3600)  # 24 hours from now

            link_data = await razorpay_client.create_payment_link(
                amount=payment.amount,
                currency=payment.currency,
                reference_id=payment.payment_id,
                description=description,
                customer_contact=payment.customer_contact or "",
                customer_email=payment.customer_email or "",
                expire_by=expire_by
            )

            if link_data:
                event_store.log_recovery_attempt(
                    session_id, payment.payment_id, "PAYMENT_LINK_SENT", 
                    {"link_id": link_data.get("id"), "short_url": link_data.get("short_url")}
                )
                await compliance_engine.record_contact(
                    compliance_engine.contact_key(payment)
                )
                return True
            else:
                event_store.log_exception(session_id, payment.payment_id, "Failed to create payment link", "EXECUTION_ERROR")
                return False

        # 4. ESCALATE
        elif strategy == RecoveryStrategy.ESCALATE:
            event_store.log_escalation(session_id, payment.payment_id, decision.reasoning)
            # An agent telephoning the customer spends one of the day's contact slots.
            # The compliance engine already *checks* escalation against that budget, so
            # not recording it here meant the ledger it checked against was missing its
            # most intrusive entry — two messages and a phone call all counted as two.
            await compliance_engine.record_contact(
                compliance_engine.contact_key(payment)
            )
            # Would typically push to Zendesk, Slack, or merchant dashboard webhook
            logger.info("executor.escalated", payment_id=payment.payment_id)
            return True

        return False


# Singleton
executor = RecoveryExecutor()
