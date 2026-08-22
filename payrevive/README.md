# PayRevive — AI-Powered Payment Recovery Engine

**Razorpay AI Buildathon 2024 — Track 03 (AI Revenue Recovery)**

PayRevive is an intelligent, autonomous agent that recovers failed payments using machine learning, contextual bandits, and LLM reasoning. It strictly adheres to Indian fintech compliance limits while maximizing merchant revenue.

## Core Intelligence Architecture

1. **XGBoost Classifier + SHAP (Root Cause Analysis)**
   - Infers the *true* reason a payment failed across 17 features (bank health, customer history, time of day).
   - Trained on realistic synthetic Indian payment data (UPI distributions, salary windows, correlated bank downtimes).
   - Uses SHAP TreeExplainer to provide human-readable reasoning for every classification.

2. **Thompson Sampling Contextual Bandit (Strategy Selection)**
   - Balances exploration vs. exploitation to find the optimal recovery strategy (e.g., immediate retry, scheduled retry, payment link).
   - Context space: `Root Cause` × `Payment Method` × `Amount Bucket`.
   - Adapts to changing bank success rates in real-time.

3. **Gemini Flash LLM (Complex Case Reasoner)**
   - Invoked dynamically for high-value transactions (>₹10,000) or low-confidence ML predictions.
   - Outputs strict, validated JSON enforcing compliance rules (e.g., no auto-retries for large amounts).

4. **Compliance & Attribution Engines**
   - **Compliance Engine**: Hard overrides for all ML/LLM decisions. Enforces max retries, quiet hours (10PM - 8AM IST), and daily contact caps.
   - **Attribution Engine**: Honestly classifies recoveries as `SYSTEM_RECOVERED`, `CUSTOMER_SELF_RECOVERED`, or `AMBIGUOUS`.

## Tech Stack

* **Backend:** FastAPI, Python 3.11, Pydantic, ARQ (Background Tasks)
* **ML:** XGBoost, SHAP, scikit-learn
* **LLM:** Google Gemini 2.0 Flash
* **Database:** Supabase (PostgreSQL)
* **Caching/State:** Redis Cloud (Feature Store, Circuit Breakers, Bandit Posteriors)
* **Frontend:** Next.js, Tailwind CSS, Recharts

## Setup & Running

This project uses hosted DB/Cache to ensure it runs easily without Docker dependencies on the host machine.

### Prerequisites
1. Python 3.11+
2. Node.js 18+

### Environment Variables
Copy `.env.example` to `.env` and fill in the values:
```bash
cp .env.example .env
```
*(You will need Supabase, Redis Cloud, Razorpay Test, and Google Gemini API keys).*

### Running Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload

# Run Worker (in another terminal)
arq app.worker.WorkerSettings
```

### Running Dashboard
```bash
cd dashboard
npm install
npm run dev
```

## Demo Flow

1. **Generate Data:** `POST /api/v1/batch/generate` creates 150 synthetic failures.
2. **Train Model:** `POST /api/v1/model/train` trains XGBoost on the synthetic data.
3. **Run Pipeline:** `POST /api/v1/batch/run` runs the bandit and execution engines on the open failures.
4. **View Dashboard:** Navigate to `http://localhost:3000` to see recovery rates and live metrics.
