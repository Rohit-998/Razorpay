export type RecoveryStatus = "RECOVERED" | "IN_FLIGHT" | "FAILED" | "QUEUED" | string;

export type PaymentSummary = {
  id: string;
  amount: number;
  currency?: string;
  bank: string;
  errorCode: string;
  rootCause?: string;
  action?: string;
  status: RecoveryStatus;
  timestamp: string;
};

export type ShapExplanation = {
  feature: string;
  value: string;
  shap_value: number;
  direction: string;
};

export type ActionScore = {
  action: string;
  score: number;
  expectedRecovery?: number;
};

export type PaymentDetails = {
  id: string;
  amount: number;
  currency: string;
  bank: string;
  timestamp: string;
  initialErrorCode: string;
  userId: string;
  rootCause: string;
  confidence: number;
  shapExplanations: ShapExplanation[];
  recoveryAction: string;
  recoveryStatus: RecoveryStatus;
  recoveryAttemptedAt?: string;
  actionScores?: ActionScore[];
  guardrails?: string[];
};

export type BatchRunResult = {
  message?: string;
  processed?: number;
  recovered?: number;
  [key: string]: unknown;
};

export type TrainingResponse = {
  accuracy?: number;
  f1_score?: number;
  f1Score?: number;
  confusion_matrix?: number[][];
  confusionMatrix?: number[][];
  [key: string]: unknown;
};

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Request failed (${response.status})`);
  }

  return response.json() as Promise<T>;
}

function numberFrom(value: unknown, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeStatus(value: unknown): RecoveryStatus {
  return String(value ?? "QUEUED").toUpperCase().replaceAll(" ", "_");
}

function normalizePayment(raw: Record<string, unknown>): PaymentSummary {
  return {
    id: String(raw.id ?? raw.payment_id ?? raw.paymentId ?? "unknown"),
    amount: numberFrom(raw.amount ?? raw.amount_inr ?? raw.value),
    currency: String(raw.currency ?? "INR"),
    bank: String(raw.bank ?? raw.bank_name ?? raw.issuer ?? "—"),
    errorCode: String(raw.initial_error_code ?? raw.error_code ?? raw.errorCode ?? "UNKNOWN"),
    rootCause: raw.root_cause ? String(raw.root_cause) : raw.predicted_root_cause ? String(raw.predicted_root_cause) : undefined,
    action: raw.recovery_action ? String(raw.recovery_action) : raw.action ? String(raw.action) : undefined,
    status: normalizeStatus(raw.recovery_status ?? raw.status),
    timestamp: String(raw.timestamp ?? raw.created_at ?? raw.failed_at ?? new Date().toISOString()),
  };
}

export async function getPayments(): Promise<PaymentSummary[]> {
  const response = await request<unknown>("/api/v1/payments?limit=12");
  const envelope = response as Record<string, unknown>;
  const rows = Array.isArray(response)
    ? response
    : Array.isArray(envelope.payments)
      ? envelope.payments
      : Array.isArray(envelope.items)
        ? envelope.items
        : Array.isArray(envelope.data)
          ? envelope.data
          : [];
  return rows.map((row) => normalizePayment(row as Record<string, unknown>));
}

export async function getPayment(paymentId: string): Promise<PaymentDetails> {
  const response = await request<unknown>(`/api/v1/payments/${encodeURIComponent(paymentId)}`);
  const envelope = response as Record<string, unknown>;
  const raw = (envelope.payment ?? envelope.data ?? response) as Record<string, unknown>;
  const prediction = (raw.prediction ?? raw.root_cause_prediction ?? {}) as Record<string, unknown>;
  const recovery = (raw.recovery_attempt ?? raw.recovery ?? raw.bandit_action ?? {}) as Record<string, unknown>;
  const explanations = raw.shap_explanations ?? raw.shap ?? prediction.shap_explanations ?? [];
  const candidateActions = recovery.action_scores ?? recovery.candidate_actions ?? raw.action_scores ?? raw.candidate_actions;
  const guardrails = recovery.guardrails ?? recovery.stopping_rules ?? raw.guardrails ?? raw.stopping_rules;

  return {
    id: String(raw.id ?? raw.payment_id ?? paymentId),
    amount: numberFrom(raw.amount ?? raw.amount_inr),
    currency: String(raw.currency ?? "INR"),
    bank: String(raw.bank ?? raw.bank_name ?? raw.issuer ?? "—"),
    timestamp: String(raw.timestamp ?? raw.created_at ?? raw.failed_at ?? ""),
    initialErrorCode: String(raw.initial_error_code ?? raw.error_code ?? "UNKNOWN"),
    userId: String(raw.user_id ?? raw.customer_id ?? "—"),
    rootCause: String(prediction.root_cause ?? raw.predicted_root_cause ?? raw.root_cause ?? "UNCLASSIFIED"),
    confidence: numberFrom(prediction.confidence ?? raw.confidence ?? raw.confidence_score),
    shapExplanations: Array.isArray(explanations)
      ? explanations.map((row) => {
          const explanation = row as Record<string, unknown>;
          return {
            feature: String(explanation.feature ?? "unknown_feature"),
            value: String(explanation.value ?? "—"),
            shap_value: numberFrom(explanation.shap_value),
            direction: String(explanation.direction ?? (numberFrom(explanation.shap_value) >= 0 ? "supports" : "opposes")),
          };
        })
      : [],
    recoveryAction: String(recovery.action ?? recovery.strategy ?? raw.recovery_action ?? raw.action ?? "PENDING"),
    recoveryStatus: normalizeStatus(recovery.status ?? raw.recovery_status ?? raw.status),
    recoveryAttemptedAt: recovery.attempted_at ? String(recovery.attempted_at) : undefined,
    actionScores: Array.isArray(candidateActions)
      ? candidateActions.map((row) => {
          const action = row as Record<string, unknown>;
          return {
            action: String(action.action ?? action.strategy ?? "UNKNOWN"),
            score: numberFrom(action.score ?? action.reward ?? action.probability),
            expectedRecovery: action.expected_recovery === undefined ? undefined : numberFrom(action.expected_recovery),
          };
        })
      : undefined,
    guardrails: Array.isArray(guardrails) ? guardrails.map(String) : undefined,
  };
}

export function runBatchRecovery() {
  return request<BatchRunResult>("/api/v1/batch/run", { method: "POST" });
}

export function trainModel() {
  return request<TrainingResponse>("/api/v1/model/train", { method: "POST" });
}
