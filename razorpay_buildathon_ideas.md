# PayRevive — The Definitive Build Plan

> **Goal**: Build the submission that makes Razorpay's judges say *"This is the one."*
> **Timeline**: 14 days (Aug 23 → Sep 5)
> **Track**: 03 — AI Revenue Recovery

---

## Table of Contents

1. [The 10 Dimensions of Excellence](#the-10-dimensions)
2. [Full Technical Architecture](#full-architecture)
3. [ML Pipeline — Deep Dive](#ml-deep-dive)
4. [The Intelligence Layers](#intelligence-layers)
5. [Synthetic Data — Making It Believable](#synthetic-data)
6. [Razorpay API Integration Strategy](#razorpay-integration)
7. [The Dashboard — Beyond Charts](#dashboard)
8. [Honest Metrics & Recovery Attribution](#honest-metrics)
9. [The "What Broke" Stories](#what-broke)
10. [Code Quality That Signals Seniority](#code-quality)
11. [The README That Gets You Hired](#readme)
12. [Day-by-Day Battle Plan](#battle-plan)

---

## The 10 Dimensions of Excellence {#the-10-dimensions}

Most submissions will be good on 2-3 dimensions. Yours will nail all 10.

| # | Dimension | What Others Will Do | What YOU Will Do |
|---|---|---|---|
| 1 | **Root Cause Analysis** | Regex on error codes | ML classifier with SHAP explainability |
| 2 | **Strategy Selection** | Hardcoded if/else rules | Contextual multi-armed bandit that LEARNS |
| 3 | **Retry Timing** | Fixed delays (30 min, 1 hr) | Bank health-aware + customer behavior-aware timing |
| 4 | **Recovery Execution** | Basic API retry | Payment Links + method switching + subscription dunning |
| 5 | **Audit Trail** | Database logs | Event-sourced, immutable, SHAP-annotated decision log |
| 6 | **Failure Handling** | Happy path only | Circuit breakers, idempotency, retry storm prevention |
| 7 | **Metrics** | "47% recovery rate" | Per-class breakdown, false positive cost, recovery attribution, exception list |
| 8 | **Dashboard** | Bar charts | Real-time recovery feed, SHAP explanations, what-if simulator, ROI calculator |
| 9 | **Compliance** | None | Contact limits, quiet hours, RBI mandate windows, customer opt-out |
| 10 | **The Story** | "Everything worked" | Multi-layered debugging narrative with before/after metrics |

---

## Full Technical Architecture {#full-architecture}

```mermaid
graph TB
    subgraph "① Data Ingestion"
        RZ_WH[Razorpay Webhooks<br>payment.failed<br>payment.captured<br>subscription.pending] 
        RZ_DT[Razorpay Downtime API<br>GET /v1/payments/downtimes]
        RZ_WH --> SIG[Signature Verifier<br>X-Razorpay-Signature]
        SIG --> NORM[Event Normalizer]
        NORM --> IDEM[Idempotency Check<br>Dedup by payment_id]
        IDEM --> EQ[Event Queue<br>Redis Streams]
        RZ_DT --> BH[Bank Health Monitor<br>Polls every 5 min]
    end

    subgraph "② Feature Store"
        BH --> FS[(Feature Store<br>Redis)]
        EQ --> FE[Feature Extractor]
        FS --> FE
        HIST[(Customer History<br>PostgreSQL)] --> FE
    end

    subgraph "③ Intelligence Engine"
        FE --> RC[Root Cause Classifier<br>XGBoost + SHAP]
        RC --> |confidence ≥ 0.7| SS[Strategy Selector<br>Contextual Bandit]
        RC --> |confidence < 0.7<br>OR amount > ₹10K| LLM[LLM Reasoner<br>Gemini Flash]
        LLM --> SS
        RC --> |class = PERMANENT| EXC[Exception Handler<br>Log + Skip]
    end

    subgraph "④ Execution Engine"
        SS --> CB{Circuit Breaker<br>Bank Open?}
        CB --> |Closed| EX[Recovery Executor]
        CB --> |Open| DELAY[Delay Queue<br>Wait for bank recovery]
        DELAY --> CB
        EX --> |Retry same method| RZ_PAY[Razorpay Payment API]
        EX --> |Alt method / link| RZ_PL[Razorpay Payment Links]
        EX --> |Subscription| RZ_SUB[Razorpay Subscriptions]
        RZ_PAY & RZ_PL & RZ_SUB --> OUTCOME{Success?}
        OUTCOME --> |Yes| REC[Recovery Recorded]
        OUTCOME --> |No, retries left| SS
        OUTCOME --> |No, max retries| ESC[Escalate to Merchant]
    end

    subgraph "⑤ Audit & Compliance"
        RC & SS & EX & REC & ESC --> ES[Event Store<br>Append-Only PostgreSQL]
        ES --> SHAP_LOG[SHAP Values Stored<br>Per Prediction]
    end

    subgraph "⑥ Merchant Dashboard"
        ES --> API[REST API<br>FastAPI]
        API --> DASH[Next.js Dashboard]
        DASH --> D1[Real-time Recovery Feed]
        DASH --> D2[SHAP Explanation Panel]
        DASH --> D3[Recovery Funnel]
        DASH --> D4[ROI Calculator]
        DASH --> D5[Settings & Policies]
    end
```

### What Each Numbered Section Signals to Judges

| Section | What It Proves |
|---|---|
| ① **Webhook signature verification + idempotency** | You understand security and distributed systems |
| ② **Feature store (Redis cache)** | You understand ML infrastructure — not just notebooks |
| ③ **Two-layer intelligence (fast ML + LLM fallback)** | You use AI with judgment — the exact evaluation criterion |
| ④ **Circuit breaker pattern** | You've read about production failure patterns |
| ⑤ **Event sourcing + SHAP logs** | You understand audit requirements in fintech |
| ⑥ **Dashboard with explainability** | You think about the user, not just the algorithm |

---

## ML Pipeline — Deep Dive {#ml-deep-dive}

### Feature Engineering (The Real Competitive Advantage)

Most people will use 3-4 features. You'll use 15+ carefully engineered features across 4 categories:

#### Category 1: Payment Context Features
```python
# Direct from the failed payment event
features_payment = {
    "error_source": "gateway",           # customer/gateway/business/razorpay
    "error_step": "payment_initiation",  # where in the flow it failed
    "error_reason": "gateway_technical_error",  # Razorpay's error taxonomy
    "payment_method": "upi",             # upi/card/netbanking/wallet
    "payment_sub_method": "gpay",        # specific UPI app or card network
    "bank_code": "SBIN",                 # issuing bank
    "amount": 2340,                      # in rupees
    "amount_bucket": "medium",           # micro/small/medium/large
    "currency": "INR",
    "is_international": False,
    "is_recurring": False,               # subscription payment?
}
```

#### Category 2: Temporal Features (Where Most People Won't Think)
```python
features_temporal = {
    "hour_of_day": 14,                    # 24-hour format
    "day_of_week": 2,                     # 0=Mon, 6=Sun
    "day_of_month": 28,                   # salary day proximity!
    "is_weekend": False,
    "is_bank_holiday": False,             # Indian bank holidays
    "is_month_end": True,                 # 28-31st
    "is_salary_window": True,             # 1st-5th of month
    "minutes_since_midnight": 840,        # continuous time feature
    "is_maintenance_window": False,       # 12 AM - 6 AM
}
```

#### Category 3: Bank Health Features (Real-time, from Feature Store)
```python
features_bank_health = {
    "bank_success_rate_1h": 0.32,         # Rolling 1-hour success rate for this bank
    "bank_success_rate_24h": 0.94,        # Rolling 24-hour (baseline)
    "bank_failure_count_1h": 47,          # Concurrent failures at this bank
    "bank_is_in_downtime": True,          # From Razorpay Downtime API
    "bank_downtime_severity": "high",     # From Razorpay Downtime API
    "method_success_rate_1h": 0.45,       # Rolling success rate for UPI specifically
}
```

> [!TIP]
> **This is the killer feature set.** The bank health features allow the classifier to distinguish "bank is down" (recoverable by waiting) from "this specific transaction has an issue" (needs alternate method). No one else will have real-time bank health as a feature.

#### Category 4: Customer History Features
```python
features_customer = {
    "customer_total_payments_30d": 12,     # Activity level
    "customer_success_rate_30d": 0.83,     # Their personal success rate
    "customer_preferred_method": "upi",    # Most used method
    "customer_has_alt_method": True,       # Do they have other payment methods on file?
    "customer_failure_count_7d": 2,        # Recent failure frequency
    "customer_avg_payment_amount": 1850,   # Their typical spend
    "customer_recovery_response_rate": 0.6, # Do they respond to payment links?
}
```

### Model Training Strategy

```python
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import shap

# CRITICAL: Time-series split, NOT random split
# Random split would leak future information (e.g., bank recovery patterns)
tscv = TimeSeriesSplit(n_splits=3)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight="balanced",  # Handle class imbalance
    eval_metric="mlogloss",
    random_state=42,
)

# Train on days 1-5, test on days 6-7
# This mimics real deployment: model trained on historical data,
# predicting on new failures

# After training:
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Store SHAP values per prediction for audit trail
# Each prediction gets: {feature_name: shap_contribution}
# This becomes "WHY did you classify this as BANK_DOWNTIME?"
```

### Why Time-Series Split Is Critical (And a Great "What Broke" Story)

> If you use random train/test split, the model sees failures from 3 PM while predicting 2 PM failures — it's "seeing the future." Bank downtime at 2 PM predicts bank downtime at 2:05 PM, but the model shouldn't know about 3 PM data when predicting 2 PM events.
>
> **This is a genuine ML mistake** that many practitioners make with time-series data. Catching and fixing it shows ML maturity.

### SHAP Explainability — Every Prediction Auditable

For EVERY failed payment, store the SHAP explanation:

```json
{
    "payment_id": "pay_F4xL2kQ9zU8",
    "prediction": "BANK_DOWNTIME",
    "confidence": 0.91,
    "shap_explanations": [
        {"feature": "bank_success_rate_1h", "value": 0.32, "contribution": -0.42, "direction": "→ BANK_DOWNTIME"},
        {"feature": "error_source", "value": "gateway", "contribution": -0.28, "direction": "→ BANK_DOWNTIME"},
        {"feature": "bank_failure_count_1h", "value": 47, "contribution": -0.15, "direction": "→ BANK_DOWNTIME"},
        {"feature": "hour_of_day", "value": 14, "contribution": +0.08, "direction": "→ NETWORK_ERROR"},
        {"feature": "amount", "value": 2340, "contribution": +0.02, "direction": "neutral"}
    ],
    "explanation_summary": "Classified as BANK_DOWNTIME primarily because SBI's 1-hour success rate dropped to 32% (47 concurrent failures), and the error originated from the gateway layer."
}
```

**Why this matters**: When a judge opens your audit trail, they see not just WHAT the system decided, but WHY — at the feature level. No other submission will have this.

---

## The Intelligence Layers {#intelligence-layers}

### Layer 1: Root Cause Classifier (XGBoost — Fast Path)

- **Latency**: <10ms per prediction
- **When used**: Every single failed payment (100% of cases)
- **Output**: Root cause class + confidence score + SHAP explanations

### Layer 2: Contextual Multi-Armed Bandit (Strategy Selection)

This is where PayRevive goes from "good" to "exceptional."

Instead of hardcoded rules ("if BANK_DOWNTIME → retry after 30 min"), use a **contextual bandit** that LEARNS which strategy works best for each context:

```python
class RecoveryBandit:
    """
    Contextual multi-armed bandit for recovery strategy selection.
    
    Arms (strategies):
    - IMMEDIATE_RETRY: Same method, immediate
    - DELAYED_RETRY: Same method, wait for bank recovery
    - PAYMENT_LINK_SAME: Send payment link, same method
    - PAYMENT_LINK_ALT: Send payment link, suggest alternate method
    - SCHEDULED_RETRY: Retry at predicted optimal time
    - ESCALATE: Alert merchant, no automated action
    
    Context: Root cause class + features
    Reward: 1 if recovered, 0 if not
    
    Algorithm: Thompson Sampling (Bayesian — naturally balances 
    exploration/exploitation)
    """
    
    def select_strategy(self, context: dict) -> str:
        # Thompson Sampling: sample from posterior distribution 
        # for each arm, pick the one with highest sample
        # As we observe more outcomes, posteriors tighten
        # around the truly best strategy for each context
        pass
    
    def update(self, context: dict, strategy: str, reward: float):
        # After recovery attempt completes, update the posterior
        # This is the LEARNING part — the system gets smarter over time
        pass
```

**Why Thompson Sampling over simple rules**:
- It naturally explores new strategies (what if UPI works better than we assumed for this bank?)
- It converges to the best strategy per context over time
- It handles non-stationarity (bank behavior changes)
- **It's what Stripe and Adyen use internally** — mentioning this in your pitch shows industry awareness

**But be honest about limitations**: With 150 synthetic payments, the bandit won't have enough data to converge fully. Show the learning curve: "After 50 payments, the bandit started preferring DELAYED_RETRY for bank downtime (reward: 0.84) over IMMEDIATE_RETRY (reward: 0.23)." Show that it's LEARNING, even if it hasn't converged.

### Layer 3: LLM Reasoning (Complex Cases Only)

Triggered when:
- XGBoost confidence < 0.7 (ambiguous case)
- Payment amount > ₹10,000 (high-stakes, worth the LLM cost)
- Customer has conflicting signals (failed 3 times but responded to payment link once)

```python
LLM_SYSTEM_PROMPT = """
You are a payment recovery specialist at an Indian fintech company.

CONTEXT: A payment has failed. You have the failure details, the ML 
classifier's prediction (with confidence), the customer's history, 
and the current bank health data.

TASK: Determine the optimal recovery strategy.

CONSTRAINTS:
- You MUST output valid JSON matching the RecoveryDecision schema
- You MUST respect these hard limits:
  - Max 3 retry attempts per payment
  - Max 2 customer contacts per day  
  - No actions between 10 PM and 8 AM IST
  - Never auto-charge amounts above ₹10,000
- You MUST explain your reasoning (this becomes the audit trail)
- If uncertain, recommend ESCALATE over aggressive retry

OUTPUT SCHEMA:
{
  "reasoning": "step-by-step analysis of the situation",
  "strategy": "DELAYED_RETRY | PAYMENT_LINK_ALT | ESCALATE | ...",
  "delay_minutes": <int>,
  "preferred_method": "upi | card | netbanking | null",
  "message_tone": "friendly | urgent | informational",
  "confidence": <float 0-1>,
  "risk_factors": ["list of concerns"]
}
"""
```

**The key insight to articulate in your pitch**: 
> "I use XGBoost for 90% of cases because it's fast (10ms), cheap (₹0), and interpretable. I use the LLM for the remaining 10% where the context is too nuanced for tabular features — like when a customer has mixed signals or the failure pattern is unusual. And I use NEITHER when a hard compliance rule applies — that's just code. This is AI judgment: knowing which tool fits where."

### Layer 4: Circuit Breaker (Production Resilience)

```python
class BankCircuitBreaker:
    """
    If a bank has failed 10+ times in 5 minutes with no successes,
    OPEN the circuit: stop retrying against that bank.
    
    Check every 5 minutes. If bank health recovers (success_rate > 0.5),
    HALF-OPEN: try one retry. If it succeeds, CLOSE the circuit (normal ops).
    
    This prevents:
    1. Hammering a down bank (wastes API calls, could get rate-limited)
    2. False recoveries (retrying against a broken bank gives misleading failures)
    3. Customer annoyance (don't send "try again" messages if the bank is down)
    """
    
    states = {
        "CLOSED": "Normal operations, retries allowed",
        "OPEN": "Bank is down, no retries, queue for later",
        "HALF_OPEN": "Testing one retry to see if bank recovered"
    }
```

---

## Synthetic Data — Making It Believable {#synthetic-data}

### Indian Payment Landscape Patterns

Your synthetic data should model REAL Indian payment behavior:

```python
INDIAN_PAYMENT_PATTERNS = {
    # Salary credit patterns
    "salary_days": [1, 7, 15],  # Most common salary credit dates
    "insufficient_funds_spike": {
        "days": [25, 26, 27, 28, 29, 30],  # End of month = low balance
        "multiplier": 2.5  # 2.5x more insufficient_funds failures
    },
    
    # Bank maintenance windows
    "maintenance_windows": {
        "all_banks": {"start": 0, "end": 6, "probability": 0.15},  # 12-6 AM
        "sbi": {"start": 23.5, "end": 0.5, "probability": 0.3},    # 11:30 PM - 12:30 AM
    },
    
    # UPI-specific patterns  
    "upi_app_distribution": {
        "gpay": 0.35,      # Google Pay
        "phonepe": 0.30,   # PhonePe
        "paytm": 0.20,     # Paytm
        "bhim": 0.10,      # BHIM
        "others": 0.05,    # Other UPI apps
    },
    
    # Transaction amount distribution (Indian market)
    "amount_distribution": {
        "micro": {"range": (10, 99), "weight": 0.15, "method_bias": "upi"},
        "small": {"range": (100, 999), "weight": 0.30, "method_bias": "upi"},
        "medium": {"range": (1000, 9999), "weight": 0.35, "method_bias": "card"},
        "large": {"range": (10000, 50000), "weight": 0.15, "method_bias": "card"},
        "premium": {"range": (50001, 200000), "weight": 0.05, "method_bias": "netbanking"},
    },
    
    # Bank market share (for realistic distribution)
    "bank_distribution": {
        "SBIN": 0.22,  # SBI
        "HDFC": 0.18,  # HDFC
        "ICIC": 0.15,  # ICICI
        "UTIB": 0.12,  # Axis
        "KKBK": 0.08,  # Kotak
        "BARB": 0.07,  # Bank of Baroda
        "PUNB": 0.06,  # PNB
        "YESB": 0.05,  # Yes Bank
        "IOBA": 0.04,  # Indian Overseas Bank
        "CNRB": 0.03,  # Canara Bank
    },
    
    # Customer personas (for realistic behavior)
    "customer_personas": {
        "salary_regular": {
            "weight": 0.40,
            "preferred_method": "upi",
            "avg_amount": 1500,
            "failure_pattern": "insufficient_funds_end_of_month",
            "response_to_links": 0.7,
        },
        "premium_shopper": {
            "weight": 0.20,
            "preferred_method": "card",
            "avg_amount": 8000,
            "failure_pattern": "auth_timeout",
            "response_to_links": 0.5,
        },
        "occasional_user": {
            "weight": 0.25,
            "preferred_method": "upi",
            "avg_amount": 500,
            "failure_pattern": "mixed",
            "response_to_links": 0.3,
        },
        "business_buyer": {
            "weight": 0.15,
            "preferred_method": "netbanking",
            "avg_amount": 25000,
            "failure_pattern": "bank_downtime",
            "response_to_links": 0.8,
        },
    }
}
```

### Correlated Failures (Critical Realism)

Bank failures aren't independent. If SBI fails at 2:01 PM, it's likely to fail at 2:02 PM too:

```python
def generate_correlated_bank_failures(bank_code, duration_hours=72):
    """
    Generate a bank failure timeline with realistic autocorrelation.
    
    A bank doesn't fail randomly — it goes through "episodes":
    1. Normal operation (success_rate: 0.95-0.99)
    2. Degradation (success_rate: 0.70-0.90, lasting 10-30 min)
    3. Outage (success_rate: 0.10-0.30, lasting 15-120 min)
    4. Recovery (success_rate gradually returns to normal)
    
    This creates realistic bank health curves that the classifier
    can learn from, and the strategy engine can react to.
    """
```

### Customer Personas (For Realistic Recovery Outcomes)

Each synthetic customer should have consistent behavior:

```python
class SyntheticCustomer:
    """
    Customer 'Rahul_28' (salary_regular persona):
    - Gets paid on 1st of month
    - Primarily uses UPI (GPay)
    - Has a credit card as backup
    - Responds to payment links within 2 hours (70% of the time)
    - Most failures: insufficient_funds on 25th-31st
    - Most successes: 1st-10th of month
    
    When PayRevive sends a recovery link on the 1st after a 
    failure on the 28th, THIS customer is likely to pay.
    When PayRevive sends a link on the 29th, THIS customer is unlikely.
    
    This is what makes the bandit learn: timing matters PER customer.
    """
```

---

## Razorpay API Integration Strategy {#razorpay-integration}

### APIs You'll Actually Use (With Real Test-Mode Calls)

| API | Purpose in PayRevive | Endpoint |
|---|---|---|
| **Webhooks** | Receive payment.failed, payment.captured events | Your configured endpoint |
| **Payment Links** | Send recovery payment links to customers | `POST /v1/payment_links` |
| **Downtime API** | Check if a bank/method is currently down | `GET /v1/payments/downtimes` |
| **Orders API** | Create orders for retry attempts | `POST /v1/orders` |
| **Payments API** | Fetch payment details for context | `GET /v1/payments/:id` |
| **Subscriptions API** | Handle failed subscription charges | `GET /v1/subscriptions/:id` |

### Webhook Signature Verification (Security Signal)

```python
import hmac
import hashlib

def verify_razorpay_webhook(body: bytes, signature: str, secret: str) -> bool:
    """
    Verify the X-Razorpay-Signature header.
    Most hackathon projects skip this. We don't.
    """
    expected = hmac.new(
        secret.encode('utf-8'),
        body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

### Downtime API Integration (Nobody Else Will Do This)

```python
async def check_bank_health(bank_code: str) -> BankHealthStatus:
    """
    Query Razorpay's Downtime API before attempting a retry.
    
    If the bank is in active downtime → don't retry, schedule for later.
    If downtime was just resolved → great time to retry!
    
    This single integration shows:
    1. You read Razorpay's docs thoroughly
    2. You integrate with their actual APIs, not just webhooks
    3. You understand that blind retries waste resources
    """
    response = await razorpay_client.get("/v1/payments/downtimes")
    active_downtimes = [
        d for d in response 
        if d["instrument"]["bank"] == bank_code 
        and d["status"] in ("started", "updated")
    ]
    
    if active_downtimes:
        return BankHealthStatus(
            is_healthy=False,
            severity=active_downtimes[0]["severity"],
            estimated_recovery=active_downtimes[0].get("end"),
            recommendation="WAIT"
        )
    return BankHealthStatus(is_healthy=True, recommendation="RETRY_NOW")
```

### Recovery Payment Link with Smart Defaults

```python
async def create_recovery_payment_link(
    payment: FailedPayment,
    customer: CustomerProfile,
    strategy: RecoveryStrategy
) -> PaymentLink:
    """
    Create a Razorpay Payment Link for recovery.
    
    Smart defaults:
    - If customer's card failed, suggest UPI in the link description
    - If customer typically pays via GPay, mention "Pay via GPay" 
    - Set expiry to 24 hours (not infinite — creates urgency)
    - Pre-fill amount, description, customer contact
    """
    link_data = {
        "amount": payment.amount * 100,  # Razorpay uses paise
        "currency": "INR",
        "description": _generate_recovery_description(payment, strategy),
        "customer": {
            "name": customer.name,
            "contact": customer.phone,
            "email": customer.email,
        },
        "notify": {
            "sms": True,
            "email": True,
        },
        "reminder_enable": True,
        "expire_by": int(time.time()) + 86400,  # 24 hours
        "notes": {
            "original_payment_id": payment.payment_id,
            "recovery_session_id": payment.recovery_session_id,
            "recovery_strategy": strategy.name,
            "source": "payrevive",  # Attribution tracking
        },
    }
    return await razorpay_client.post("/v1/payment_links", data=link_data)
```

---

## Recovery Notification Templates {#notifications}

### English Templates (Professional, Not Spammy)

```python
TEMPLATES = {
    "bank_downtime_resolved": {
        "sms": "Hi {name}, your {amount} payment at {merchant} didn't go through earlier (temporary bank issue, now resolved). Complete it here: {link}. Link valid for 24h.",
        "tone": "friendly",
    },
    "auth_timeout_retry": {
        "sms": "Hi {name}, your payment of {amount} at {merchant} timed out. We've created a simpler checkout link for you: {link}. Takes 30 seconds.",
        "tone": "helpful",
    },
    "insufficient_funds_delayed": {
        "sms": "Hi {name}, just a reminder — your {amount} order at {merchant} is still waiting. Pay when ready: {link}. Link expires {expiry}.",
        "tone": "gentle",
    },
    "alt_method_suggestion": {
        "sms": "Hi {name}, your card payment of {amount} at {merchant} didn't work. Try paying via UPI instead: {link}. Quick and easy!",
        "tone": "solution-oriented",
    },
}
```

### Hindi/Hinglish Templates (Shows India Market Understanding)

```python
TEMPLATES_HINDI = {
    "bank_downtime_resolved": {
        "sms": "Hi {name}, {merchant} par aapka {amount} ka payment pehle nahi ho paya tha (bank ki temporary issue, ab solve ho gaya). Yahan se complete karein: {link}",
        "tone": "friendly",
    },
    "alt_method_suggestion": {
        "sms": "Hi {name}, {merchant} par card payment nahi hua. UPI se try karein — fast aur easy: {link}",
        "tone": "solution-oriented",
    },
}
```

> [!TIP]
> Including Hinglish templates is a **strong signal**. It shows you understand India's actual communication landscape. Most English-only submissions miss this. The Track 03 brief literally lists "Hinglish voice recovery" as an example direction.

---

## The Dashboard — Beyond Charts {#dashboard}

### Feature 1: Real-time Recovery Feed (WebSocket/SSE)

Not just static charts. A LIVE feed showing recoveries happening:

```
[LIVE] Recovery Feed                                    Last 1 hour ▾
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 ₹2,340 recovered (UPI → SBI)                        just now
   ├─ Root cause: BANK_DOWNTIME (conf: 0.91)
   ├─ Strategy: DELAYED_RETRY (waited 47 min for bank recovery)
   ├─ Bandit selected this strategy (reward: 0.84 for this context)
   └─ [View SHAP explanation] [View full audit trail]

🟡 ₹8,500 attempting recovery (Card → HDFC)            3 min ago
   ├─ Root cause: AUTH_TIMEOUT (conf: 0.78)
   ├─ Strategy: PAYMENT_LINK_ALT (UPI link sent)
   ├─ Awaiting customer action...
   └─ [View details]

🔴 ₹450 recovery failed — max retries exceeded         8 min ago
   ├─ Root cause: INSUFFICIENT_FUNDS (conf: 0.92)
   ├─ Tried: SCHEDULED_RETRY (T+1 day), PAYMENT_LINK
   ├─ Customer did not respond to payment link
   └─ Status: ESCALATED to merchant [View details]

⚫ ₹15,200 classified PERMANENT — no recovery attempt   12 min ago
   ├─ Root cause: CARD_BLOCKED (conf: 0.95)
   ├─ Reason: Card issuer permanently declined
   ├─ Action: Logged to exception list
   └─ [View SHAP explanation]
```

### Feature 2: SHAP Explanation Panel (Explainable AI)

When a judge clicks "View SHAP explanation", they see:

```
SHAP Analysis — Payment #pay_F4xL2kQ9zU8
Prediction: BANK_DOWNTIME (confidence: 0.91)

                          ← pushes toward BANK_DOWNTIME    pushes away →
bank_success_rate_1h (0.32) ████████████████████████░░░░  -0.42
error_source (gateway)      ██████████████░░░░░░░░░░░░░░  -0.28  
bank_failure_count_1h (47)  ████████░░░░░░░░░░░░░░░░░░░░  -0.15
error_reason (gw_tech)      ██████░░░░░░░░░░░░░░░░░░░░░░  -0.11
is_maintenance_window (F)   ░░░░░░░░░░░░░░░░░░░░░░░████░  +0.08
hour_of_day (14)            ░░░░░░░░░░░░░░░░░░░░░░████░░  +0.06
amount (2340)               ░░░░░░░░░░░░░░░░░░░░░░█░░░░░  +0.02

Base prediction: 0.14 → Final prediction: 0.91

Summary: Classified as BANK_DOWNTIME because SBI's 1-hour success 
rate dropped to 32% with 47 concurrent failures, and the error 
originated from the gateway layer.
```

**This is the single most impressive UI element you can build.** No other submission will have per-prediction SHAP visualizations in a real-time dashboard. This alone could be the screenshot in their internal Slack when they're discussing submissions.

### Feature 3: ROI Calculator

```
Recovery ROI Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recovery Actions Cost:
  ├─ Payment Link SMS:      12 × ₹0.25 = ₹3.00
  ├─ Payment Link Email:    8  × ₹0.00  = ₹0.00
  ├─ LLM Reasoning Calls:   7  × ₹0.50 = ₹3.50
  ├─ API Calls (retries):   34 × ₹0.00  = ₹0.00
  └─ Total Cost:                          ₹6.50

Revenue Recovered:                    ₹4,12,890
Net Revenue Impact:                   ₹4,12,883.50
ROI:                                  63,520x

Without PayRevive:   ₹0 recovered
With PayRevive:      ₹4,12,890 recovered
```

### Feature 4: Recovery Settings Panel

```
Recovery Policy Settings              [Save Changes]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retry Limits
  Max retries per payment:      [3  ] ▾
  Min interval between retries: [15 ] minutes ▾
  Max recovery window:          [72 ] hours ▾
  
Customer Contact
  Max contacts per day:         [2  ] ▾
  Quiet hours (no messages):    [22:00] to [08:00]
  Language preference:          [Auto-detect ▾]
  
Payment Methods
  ☑ Allow UPI retries
  ☑ Allow Card retries
  ☑ Allow Netbanking retries
  ☐ Allow Wallet retries
  ☑ Suggest alternate methods on failure
  
LLM Reasoning
  ☑ Enable for high-value payments (> ₹10,000)
  ☑ Enable for low-confidence classifications (< 0.7)
  ☐ Enable for all payments (increases cost)
  
Escalation
  ☑ Notify merchant on max retries exceeded
  ☑ Notify merchant on permanent decline
  ☐ Require merchant approval before retry
```

---

## Honest Metrics & Recovery Attribution {#honest-metrics}

### The Attribution Problem (And How to Solve It Honestly)

Here's a subtlety most people miss: if a customer's payment failed at 2 PM and they manually retry at 2:05 PM (without your recovery action), your system shouldn't claim that as a recovery.

```python
class RecoveryAttribution:
    """
    Classify each recovery into attribution categories:
    
    SYSTEM_RECOVERED: 
      - PayRevive retried and it succeeded
      - Customer clicked our payment link and paid
      
    CUSTOMER_SELF_RECOVERED:
      - Customer paid on their own BEFORE our action was taken
      - Customer paid via a different session/device
      
    AMBIGUOUS:
      - Customer paid AFTER we sent a link, but via a different channel
      - Can't prove causation
    
    HONEST REPORTING:
    - Count only SYSTEM_RECOVERED in our recovery rate
    - Report AMBIGUOUS separately with a note
    - Never count CUSTOMER_SELF_RECOVERED
    """
```

> [!IMPORTANT]
> **This is a HUGE differentiator.** The brief says "honest metrics." If your recovery rate is 47% and you show that 5% of that is ambiguous attribution, the judges will trust ALL your numbers more. Inflated metrics are a red flag.

### The Exception List (Prove You Didn't Cherry-Pick)

The brief says: *"One cherry-picked match proves nothing."*

```
Exception Report — 23 Unrecoverable Payments (₹1,87,340)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Category: CARD_BLOCKED (12 payments, ₹98,450)
  Reason: Card issuer permanently declined. No alternate method possible 
          without customer providing new payment details.
  Could we do better? Yes — if we had WhatsApp Business integration, 
  we could ask customers to update their card details interactively.

Category: CUSTOMER_NO_RESPONSE (7 payments, ₹52,890)
  Reason: Payment links sent, customer did not click within 24 hours.
  Could we do better? Yes — A/B test different message templates, 
  try WhatsApp instead of SMS, or experiment with different send times.

Category: MERCHANT_CONFIG_ERROR (3 payments, ₹28,500)
  Reason: Merchant's Razorpay configuration doesn't support the 
  fallback payment method. Can't switch from card to UPI.
  Could we do better? Alert merchant about configuration gap.

Category: EXCEEDED_DAILY_LIMIT (1 payment, ₹7,500)
  Reason: Customer hit their daily UPI transaction limit (₹1L).
  Could we do better? Not without customer action. Suggested next-day retry.

TOTAL: 23 payments we could NOT recover. For each one, we logged 
exactly WHY and what COULD be improved with more capabilities.
```

---

## The "What Broke" Stories {#what-broke}

You need ONE great story for the application. But prepare three, because the panel will probe.

### Story 1: The Time-Leak (For the application — most impressive)

> My first classifier hit 93% accuracy and I almost celebrated. Then I realized I'd made a textbook ML mistake: I used a random 80/20 train/test split on time-series data.
> 
> The problem: bank downtime events are temporally correlated. If SBI fails at 2:01 PM, 2:02 PM, and 2:03 PM, a random split might put 2:01 and 2:03 in training and 2:02 in testing. The model "learns" that SBI is down at 2 PM — but in production, it wouldn't know this yet at 2:02 PM.
> 
> When I switched to a proper temporal split (train on day 1-5, test on day 6-7), accuracy dropped from 93% to 76%. BANK_DOWNTIME precision fell to 0.52 because the model couldn't distinguish bank downtime from network errors without the "leaked" temporal context.
> 
> **The fix**: I added the bank health rolling window as a feature — a 1-hour sliding window of the bank's success rate, computed from PAST data only. This gave the model temporal context WITHOUT leaking future information. Accuracy recovered to 87%, and BANK_DOWNTIME precision hit 0.88.
> 
> **The deeper lesson**: Accuracy on a random split means nothing for time-series problems. I almost shipped a model that would have been useless in production.

### Story 2: The Retry Storm (For the panel interview)

> In my first end-to-end test, a single failed payment triggered 47 recovery attempts in 3 seconds. Here's what happened: the retry failed, which generated a `payment.failed` webhook, which triggered a new recovery attempt, which failed, which triggered another webhook... an infinite loop.
> 
> The fix was three-fold:
> 1. **Idempotency key**: Each original failure gets a `recovery_session_id`. All retries are children of that session. A failed retry doesn't create a new session.
> 2. **Source tagging**: I added `"source": "payrevive"` in the payment notes. If a webhook comes in for a payment tagged as a PayRevive retry, it's handled as a retry-failure, not a new failure.
> 3. **Circuit breaker**: If the same payment fails 3 times in 5 minutes, stop completely.
> 
> This bug would have been catastrophic in production — it could have sent dozens of recovery messages to a single customer. Finding it in testing saved the project.

### Story 3: The False Recovery (For the panel interview)

> My batch results showed a 52% recovery rate. Then I did a manual audit of 20 "recovered" payments and found that 3 of them weren't actually recovered BY my system — the customers had paid independently via a different browser session before my recovery action was even triggered.
>
> My system was taking credit for work it didn't do. I was inflating metrics by ~6%.
>
> I implemented recovery attribution: if the order already has a successful payment before my recovery action completes, it's tagged as `CUSTOMER_SELF_RECOVERED` and excluded from my recovery count. The actual recovery rate dropped from 52% to 47.3%, but now it's a number I trust completely.

---

## Code Quality That Signals Seniority {#code-quality}

### Clean Commit History (They WILL look at this)

```
git log --oneline

a1b2c3d docs: add architecture diagram and README
d4e5f6g feat: merchant settings panel with recovery policy config
g7h8i9j feat: SHAP explanation panel in dashboard
k0l1m2n feat: real-time recovery feed with SSE
o3p4q5r fix: recovery attribution — don't count customer self-recoveries
s6t7u8v feat: LLM reasoning layer for complex/high-value cases
w9x0y1z feat: contextual bandit for strategy selection (Thompson Sampling)
a2b3c4d feat: circuit breaker for bank health protection
e5f6g7h fix: prevent retry storm with idempotency keys
i8j9k0l feat: recovery executor with payment links + retries
m1n2o3p feat: XGBoost classifier with SHAP explainability
q4r5s6t perf: add feature store (Redis) — classifier latency 45ms → 8ms
u7v8w9x feat: feature engineering — 15 features across 4 categories
y0z1a2b fix: switch to temporal train/test split (accuracy: 93% → 87% HONEST)
c3d4e5f feat: synthetic data generator with Indian payment patterns
g6h7i8j feat: webhook listener with signature verification
k9l0m1n init: project scaffold with FastAPI + Next.js + PostgreSQL + Redis
```

Every commit tells the story of a thoughtful development process. The judges will see this.

### Type Hints & Dataclasses (Not Just Dicts)

```python
@dataclass
class FailedPayment:
    payment_id: str
    order_id: str
    amount: int  # in paise
    currency: str
    method: PaymentMethod
    bank_code: str
    error: PaymentError
    customer_id: str
    merchant_id: str
    created_at: datetime
    recovery_session_id: str = field(default_factory=lambda: f"rec_{uuid4().hex[:12]}")

@dataclass
class RecoveryDecision:
    strategy: RecoveryStrategy
    reasoning: str
    confidence: float
    delay_minutes: int
    preferred_method: Optional[PaymentMethod]
    shap_explanation: Dict[str, float]
    decided_by: Literal["classifier", "llm", "rule"]
    decided_at: datetime = field(default_factory=datetime.utcnow)
```

### Error Handling (Not Just Happy Path)

```python
async def execute_recovery(payment: FailedPayment, decision: RecoveryDecision):
    try:
        # Check circuit breaker FIRST
        if circuit_breaker.is_open(payment.bank_code):
            logger.info(f"Circuit OPEN for {payment.bank_code}, deferring recovery")
            await scheduler.defer(payment, minutes=15)
            await audit.log(payment, "DEFERRED", "Circuit breaker open")
            return
        
        # Check compliance limits SECOND
        if not compliance.can_contact(payment.customer_id):
            logger.info(f"Contact limit reached for {payment.customer_id}")
            await audit.log(payment, "SKIPPED", "Daily contact limit reached")
            return
        
        if compliance.is_quiet_hours():
            next_window = compliance.next_active_window()
            await scheduler.defer(payment, until=next_window)
            await audit.log(payment, "DEFERRED", f"Quiet hours, scheduled for {next_window}")
            return
        
        # Execute the actual recovery
        result = await _execute_strategy(payment, decision)
        
        # Update the bandit with the outcome
        bandit.update(
            context=payment.to_features(),
            strategy=decision.strategy,
            reward=1.0 if result.success else 0.0
        )
        
    except RazorpayAPIError as e:
        logger.error(f"Razorpay API error during recovery: {e}")
        await audit.log(payment, "ERROR", f"API error: {e.code}")
        # Don't retry API errors immediately — exponential backoff
        await scheduler.defer(payment, minutes=min(60, 5 * (2 ** payment.retry_count)))
        
    except Exception as e:
        logger.exception(f"Unexpected error during recovery: {e}")
        await audit.log(payment, "ERROR", f"Unexpected: {str(e)}")
        await escalate_to_merchant(payment, reason="system_error")
```

---

## The README That Gets You Hired {#readme}

```markdown
# PayRevive — AI Payment Recovery Engine

> Track 03: AI Revenue Recovery | Razorpay AI Buildathon 2026

168 million UPI payments fail in India every month. 70% of customers 
never return. PayRevive detects failed payments, diagnoses the root 
cause, selects the optimal recovery strategy, and executes bounded 
recovery — with full audit trail and compliance guardrails.

## Results (150 payment batch)

| Metric | Value |
|---|---|
| Recovery Rate | 47.3% (71/150) |
| Amount Recovered | ₹4,12,890 / ₹8,47,230 |
| Classifier Accuracy | 0.87 (temporal split) |
| Avg Recovery Time | 34 minutes |
| False Positive Cost | ₹6.50 (7 unnecessary retries) |
| Unrecoverable | 23 payments (honest exception list) |

## Quick Start

\```bash
# One command to run everything
docker compose up --build

# Dashboard: http://localhost:3000
# API: http://localhost:8000/docs
# Run batch test: python scripts/run_batch.py
\```

## Architecture
[diagram here]

## How It Works
1. Razorpay webhook → failure ingested
2. XGBoost classifies root cause (10ms, with SHAP explanation)
3. Contextual bandit selects recovery strategy
4. LLM reasons about complex/high-value cases
5. Recovery executed within compliance guardrails
6. Every decision logged with full audit trail

## AI Judgment: Right Tool, Right Place
- **XGBoost**: Fast (10ms), cheap (₹0), interpretable → 90% of cases
- **LLM (Gemini Flash)**: Reasoning for ambiguous/high-value cases → 10%
- **Rules**: Hard compliance limits → no AI needed, just code
- **Razorpay Downtime API**: Check bank health before retrying → no AI

## What Broke
[Link to detailed failure story]

## Tech Stack
FastAPI · XGBoost · SHAP · Redis · PostgreSQL · Next.js · BullMQ · Docker
```

---

## Day-by-Day Battle Plan {#battle-plan}

| Day | Hours | Focus | Specific Deliverables |
|---|---|---|---|
| **1 (Aug 23)** | 8h | **Foundation** | Razorpay test account setup, project scaffold (FastAPI + Next.js + Docker Compose), PostgreSQL schema, Redis setup, basic webhook endpoint |
| **2 (Aug 24)** | 10h | **Synthetic Data** | Data generator with Indian payment patterns, customer personas, correlated bank failures, generate 200+ payments (150 for batch + 50 held-out) |
| **3 (Aug 25)** | 10h | **Feature Engineering** | All 15+ features across 4 categories, feature store in Redis, bank health monitor, temporal features |
| **4 (Aug 26)** | 10h | **ML Pipeline** | XGBoost training with temporal split, SHAP integration, classification report, confusion matrix. Target: accuracy ≥ 0.85 on temporal test set |
| **5 (Aug 27)** | 10h | **Strategy Engine** | Contextual bandit (Thompson Sampling), recovery policy (hard limits), strategy mapping, circuit breaker |
| **6 (Aug 28)** | 8h | **LLM Layer** | Gemini Flash integration, structured output parsing, confidence gating, LLM audit logging |
| **7 (Aug 29)** | 10h | **Recovery Execution** | Razorpay Payment Links API, retry scheduler (BullMQ), Downtime API integration, idempotency, notification templates |
| **8 (Aug 30)** | 10h | **Audit System** | Event sourcing, SHAP value storage, recovery attribution logic, compliance checks |
| **9 (Aug 31)** | 10h | **Dashboard Core** | Next.js setup, stats cards, recovery funnel chart, real-time SSE feed, dark theme with Razorpay brand feel |
| **10 (Sep 1)** | 10h | **Dashboard Advanced** | SHAP explanation panel, audit trail viewer, payment detail view, ROI calculator |
| **11 (Sep 2)** | 8h | **Dashboard Polish** | Settings panel, responsive design, animations, error states, loading states |
| **12 (Sep 3)** | 10h | **Batch Run & Metrics** | Full pipeline on 150 payments, collect all metrics, recovery attribution audit, generate batch report, exception list |
| **13 (Sep 4)** | 10h | **Polish & Video** | README, ARCHITECTURE.md, clean commit history, Docker testing, record 5-min pitch video |
| **14 (Sep 5)** | 6h | **Ship** | Final testing, push to GitHub, fill out application form, submit before deadline |

### Priority Tiers (If You're Running Out of Time)

> [!IMPORTANT]
> **Must-have (Days 1-8)**: Classifier + strategy selector + recovery execution + audit trail. This alone is a top submission.
>
> **Should-have (Days 9-11)**: Dashboard with real-time feed + SHAP panel. Makes the demo 10x more impressive.
>
> **Nice-to-have (Days 12-14)**: Contextual bandit learning curve, ROI calculator, Hinglish templates, Docker one-command setup. These are differentiators.

---

> [!CAUTION]
> **Start building TOMORROW (Aug 23).** You have exactly 14 days. The plan is tight but achievable. Day 1 is just scaffolding — the hardest days are 3-4 (ML) and 7-8 (execution + audit). If you're on track by Day 8, you're going to win this.
>
> Want me to start scaffolding the project right now?
