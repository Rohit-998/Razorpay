-- PayRevive — Initial Database Schema
-- Run automatically on first docker compose up

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================
-- CORE TABLES
-- =============================================================

-- Failed payments ingested from Razorpay webhooks
CREATE TABLE payments (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id          VARCHAR(50) UNIQUE NOT NULL,
    order_id            VARCHAR(50),
    amount              INTEGER NOT NULL,  -- in paise
    currency            VARCHAR(3) DEFAULT 'INR',
    method              VARCHAR(20) NOT NULL,
    bank                VARCHAR(10),
    wallet              VARCHAR(20),
    vpa                 VARCHAR(100),
    error_code          VARCHAR(50),
    error_source        VARCHAR(20),
    error_step          VARCHAR(50),
    error_reason        VARCHAR(50),
    error_description   TEXT,
    customer_contact    VARCHAR(20),
    customer_email      VARCHAR(100),
    is_recurring        BOOLEAN DEFAULT FALSE,
    raw_webhook         JSONB,
    created_at          TIMESTAMPTZ NOT NULL,
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Recovery sessions — one per failed payment recovery attempt
CREATE TABLE recovery_sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id          VARCHAR(50) NOT NULL REFERENCES payments(payment_id),
    status              VARCHAR(30) NOT NULL DEFAULT 'OPEN',
    root_cause          VARCHAR(30),
    root_cause_confidence FLOAT,
    strategy            VARCHAR(30),
    decided_by          VARCHAR(20),
    retry_count         INTEGER DEFAULT 0,
    contact_count       INTEGER DEFAULT 0,
    amount_recovered    INTEGER DEFAULT 0,
    attribution         VARCHAR(30),
    shap_explanation    JSONB,
    llm_reasoning       TEXT,
    opened_at           TIMESTAMPTZ DEFAULT NOW(),
    closed_at           TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Append-only audit trail — NO UPDATEs, NO DELETEs
CREATE TABLE audit_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recovery_session_id UUID NOT NULL REFERENCES recovery_sessions(id),
    payment_id          VARCHAR(50) NOT NULL,
    event_type          VARCHAR(50) NOT NULL,
    event_data          JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Bandit posteriors for Thompson Sampling
CREATE TABLE bandit_posteriors (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_key         VARCHAR(100) NOT NULL,
    strategy            VARCHAR(30) NOT NULL,
    alpha               FLOAT DEFAULT 1.0,
    beta                FLOAT DEFAULT 1.0,
    total_trials        INTEGER DEFAULT 0,
    total_successes     INTEGER DEFAULT 0,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(context_key, strategy)
);

-- Synthetic customers for demo
CREATE TABLE customers (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id         VARCHAR(50) UNIQUE NOT NULL,
    name                VARCHAR(100),
    phone               VARCHAR(20),
    email               VARCHAR(100),
    persona             VARCHAR(30),
    preferred_method    VARCHAR(20),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Recovery policy settings (single row)
CREATE TABLE recovery_settings (
    id                  INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    max_retries_per_payment     INTEGER DEFAULT 3,
    min_retry_interval_minutes  INTEGER DEFAULT 15,
    max_recovery_window_hours   INTEGER DEFAULT 72,
    max_contacts_per_day        INTEGER DEFAULT 2,
    quiet_hours_start           INTEGER DEFAULT 22,
    quiet_hours_end             INTEGER DEFAULT 8,
    max_link_amount_paise       INTEGER DEFAULT 5000000,
    require_action_above_paise  INTEGER DEFAULT 1000000,
    enable_llm_reasoning        BOOLEAN DEFAULT TRUE,
    llm_confidence_threshold    FLOAT DEFAULT 0.7,
    llm_amount_threshold_paise  INTEGER DEFAULT 1000000,
    updated_at                  TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default settings
INSERT INTO recovery_settings (id) VALUES (1);

-- =============================================================
-- INDEXES
-- =============================================================

CREATE INDEX idx_payments_payment_id ON payments(payment_id);
CREATE INDEX idx_payments_created_at ON payments(created_at);
CREATE INDEX idx_payments_method ON payments(method);
CREATE INDEX idx_payments_bank ON payments(bank);
CREATE INDEX idx_payments_error_reason ON payments(error_reason);

CREATE INDEX idx_sessions_payment_id ON recovery_sessions(payment_id);
CREATE INDEX idx_sessions_status ON recovery_sessions(status);
CREATE INDEX idx_sessions_root_cause ON recovery_sessions(root_cause);
CREATE INDEX idx_sessions_created_at ON recovery_sessions(created_at);

CREATE INDEX idx_audit_session_id ON audit_events(recovery_session_id);
CREATE INDEX idx_audit_payment_id ON audit_events(payment_id);
CREATE INDEX idx_audit_event_type ON audit_events(event_type);
CREATE INDEX idx_audit_created_at ON audit_events(created_at);

CREATE INDEX idx_bandit_context ON bandit_posteriors(context_key);
CREATE INDEX idx_customers_phone ON customers(phone);
