/**
 * The API client. Every number this dashboard shows comes through here.
 *
 * Two rules, both of which the version this replaced broke.
 *
 * **No fabricated fallback.** The old client, on any failure, rendered `lib/demo-data.ts` —
 * hand-written rows using root causes the backend does not have (`USER_ABANDONMENT`,
 * `MERCHANT_ISSUE`) and SHAP features that are not among the seventeen (`bank_latency_p95`,
 * `user_balance`) — behind a small "Showing sample data" badge. A reviewer clicking through a
 * dashboard whose backend was not running would have seen a complete, plausible, entirely
 * invented product. Failure is a value here (`Result<T>`), it is rendered as itself, and the
 * message carries the command that fixes it.
 *
 * **No arithmetic.** The harness computes the lift, the interval, the shares and the gate
 * counts; these functions parse and pass them on. A rate the browser derives is a rate nobody
 * reviewed, and the one the old dashboard derived — recovered ÷ total, labelled "recovery
 * rate" — is the exact quantity `REPORT.md` argues is not a measurement of anything.
 */

import { when } from "./format";

const BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/+$/, "");
const V1 = `${BASE}/api/v1`;


/** Why a panel is empty, in the words of whatever refused to answer. */
export type Failure = {
  kind: "unreachable" | "not_ready" | "not_found" | "error";
  /** One sentence, safe to render. */
  message: string;
  /** The shell command that fixes it, when the backend named one. */
  fix?: string;
  /** What not having this costs, when the backend said. */
  consequence?: string;
  status?: number;
};

export type Result<T> = { ok: true; data: T } | { ok: false; error: Failure };

async function get<T>(path: string): Promise<Result<T>> {
  let response: Response;
  try {
    response = await fetch(`${V1}${path}`, { cache: "no-store" });
  } catch {
    return {
      ok: false,
      error: {
        kind: "unreachable",
        message: `Nothing is answering at ${BASE}.`,
        fix: "cd backend && uvicorn app.main:app --reload",
      },
    };
  }

  if (response.ok) {
    try {
      return { ok: true, data: (await response.json()) as T };
    } catch {
      return { ok: false, error: { kind: "error", message: "The response was not JSON.", status: response.status } };
    }
  }

  return { ok: false, error: await failureFrom(response, path) };
}

/** Turn a non-2xx into a `Failure`, preserving what the backend said about the fix.
 *
 * `/eval/*` answers 503 with a sentence naming `python -m app.eval`; `/model/metrics` answers
 * with `{error, consequence, fix}`. Both are the endpoint explaining what is missing and how to
 * produce it, which is more useful than any message this file could invent, so both survive to
 * the screen intact. */
async function failureFrom(response: Response, path: string): Promise<Failure> {
  const kind: Failure["kind"] =
    response.status === 503 ? "not_ready" : response.status === 404 ? "not_found" : "error";
  let detail: unknown;
  try {
    detail = (await response.json())?.detail;
  } catch {
    detail = undefined;
  }

  if (typeof detail === "string") {
    const fix = detail.match(/`([^`]+)`/)?.[1];
    return { kind, message: fix ? detail.replace(/`/g, "") : detail, fix, status: response.status };
  }
  if (detail && typeof detail === "object") {
    const d = detail as Record<string, string>;
    return {
      kind,
      message: d.error ?? `${path} answered ${response.status}.`,
      fix: d.fix,
      consequence: d.consequence,
      status: response.status,
    };
  }
  return { kind, message: `${path} answered ${response.status}.`, status: response.status };
}

async function post<T>(path: string): Promise<Result<T>> {
  let response: Response;
  try {
    response = await fetch(`${V1}${path}`, { method: "POST", cache: "no-store" });
  } catch {
    return {
      ok: false,
      error: {
        kind: "unreachable",
        message: `Nothing is answering at ${BASE}.`,
        fix: "cd backend && uvicorn app.main:app --reload",
      },
    };
  }
  if (!response.ok) return { ok: false, error: await failureFrom(response, path) };
  return { ok: true, data: (await response.json()) as T };
}

// ── The measurement ───────────────────────────────────────────────────────────────────

export type Health = {
  status: "healthy" | "degraded" | string;
  services: Record<string, boolean>;
};

export const getHealth = () => get<Health>("/health");


/** A rupee figure with the bootstrap interval it was resampled to. */
export type Estimate = {
  mean: number;
  low: number;
  high: number;
  seeds: number;
  excludes_zero: boolean;
};

export type LadderRow = {
  policy: string;
  blurb: string;
  is_proposal: boolean;
  is_ceiling: boolean;
  is_baseline: boolean;
  lift: Estimate;
  share_of_achievable: number;
};

export type Ladder = {
  scenario: string;
  scenarios_available: string[];
  generated_at: string;
  design: {
    baseline_policy: string;
    ceiling_policy: string;
    scenarios: string[];
    seeds: number[];
    batches_run: number;
    pairing: string;
  };
  policies: LadderRow[];
  net_lift?: Estimate;
  regret_vs_ceiling?: Estimate;
  seeds_beating_baseline?: { count: number; of: number };
  totals?: Record<string, number>;
  concerns?: Record<string, number>;
  hard_limits?: Record<string, number>;
  self_inflicted_block_rate?: { count: number; of: number; rate: number };
  shippable?: boolean;
};

export const getLadder = (scenario?: string) =>
  get<Ladder>(`/eval/ladder${scenario ? `?scenario=${encodeURIComponent(scenario)}` : ""}`);

export type Gate = { key: string; label: string; count: number; of: number; passed: boolean };

export type ShippabilityRow = {
  policy: string;
  blurb: string;
  is_proposal: boolean;
  is_ceiling: boolean;
  gates: Gate[];
  harm: { label: string; count: number; of: number; rate: number };
  failed_gates: string[];
  verdict: string;
};

export type Shippability = {
  generated_at: string;
  batches_run: number;
  gate_keys: string[];
  policies: ShippabilityRow[];
};

export const getShippability = () => get<Shippability>("/eval/shippability");

export type CauseRow = {
  cause: string;
  note: string;
  payments: number;
  at_risk_rupees: number;
  by_policy: Record<
    string,
    {
      recovered_rupees: number;
      system_recovered_rupees: number;
      spend_rupees: number;
      escalations: number;
      share_of_at_risk: number;
    }
  >;
};

export type Causes = {
  generated_at: string;
  policy: string;
  compared_with: string[];
  causes: CauseRow[];
};

export const getCauses = (policy?: string) =>
  get<Causes>(`/eval/causes${policy ? `?policy=${encodeURIComponent(policy)}` : ""}`);

// ── The diagnosis ─────────────────────────────────────────────────────────────────────

export type ClassMetrics = { precision: number; recall: number; f1: number; support: number };

export type ModelMetrics = {
  loaded: boolean;
  accuracy: number;
  macro_f1: number;
  reference_points: {
    majority_class: string;
    majority_class_rate: number;
    bayes_optimal_error_fields_only: number;
    bayes_optimal_error_fields_only_uniform_prior: number;
    distinct_error_signatures: number;
    error_fields_only_model: number;
    error_fields_only_standard_error: number;
  };
  reads_above_the_bound: number;
  per_class: Record<string, ClassMetrics>;
  confusion_matrix: number[][];
  class_order: string[];
  under_distribution_shift: Record<string, { accuracy: number; payments: number }>;
  data: {
    train_scenario: string;
    train_seeds: number[];
    test_seeds: number[];
    split: string;
    labels: string;
    features_built_by: string;
  };
  hardest_class: string;
};

export const getModelMetrics = () => get<ModelMetrics>("/model/metrics");
export const trainModel = () => post<{ status: string; metrics: unknown }>("/model/train");

// ── The running system ────────────────────────────────────────────────────────────────

export type DashboardStats = {
  sessions_total: number;
  by_status: Record<string, number>;
  open: number;
  at_risk_paise: number;
  /** Ingested minus returned — subtracted by the endpoint, not by the page. */
  unrecovered_paise: number;
  attributed: Record<string, { label: string; sessions: number; amount_paise: number }>;
  unattributed: { label: string; sessions: number; amount_paise: number; why: string };
  attribution_order: string[];
  /** The one share the dashboard draws, computed by the endpoint. See its `note`. */
  provably_ours: {
    amount_paise: number;
    sessions: number;
    recovered_paise: number;
    share_of_recovered: number;
    /** Recoveries excluded from the numerator because no audit event decided them. */
    unestablished_sessions: number;
    unestablished_paise: number;
    /** The same question over one population: recoveries whose cause is established.
     *
     *  `share_of_recovered` divides a real numerator by a denominator that still holds 217
     *  legacy rows a deleted `random.random()` wrote, so it reads ~0% and is honest about the
     *  wrong thing. This is counted in sessions because every verdict except
     *  `SYSTEM_RECOVERED` books zero rupees on purpose. */
    established: {
      sessions: number;
      ours_sessions: number;
      self_recovered_sessions: number;
      share_of_established_sessions: number;
      note: string;
    };
    /** Present only when something was excluded. A bare 0% cannot tell "nothing worked"
     *  from "nothing was measured", so the endpoint says which one it is. */
    caveat: string | null;
    note: string;
  };
  counterfactual: { available_at: string; note: string };
  on_recovery_rates: string;
};

export const getStats = () => get<DashboardStats>("/dashboard/stats");

export type ExceptionRow = {
  payment_id: string;
  session_id: string;
  category: string;
  reason: string;
  logged_at: string;
};

export type Exceptions = {
  count: number;
  by_category: Record<string, number>;
  exceptions: ExceptionRow[];
  note: string;
};

export const getExceptions = (limit = 25) => get<Exceptions>(`/dashboard/exceptions?limit=${limit}`);

/** One limit the compliance engine enforces, as the engine's own settings report it. */
export type PolicyLimit = {
  key: string;
  label: string;
  value: number | string;
  applies_to: string[];
  why: string;
};

export type CompliancePolicy = {
  note: string;
  limits: PolicyLimit[];
  always_allowed: { actions: string[]; why: string };
};

export const getCompliancePolicy = () => get<CompliancePolicy>("/compliance/policy");


export type BanditArm = { alpha: number; beta: number; mean: number; trials: number };
export const getBandit = () =>
  get<{ contexts: Record<string, Record<string, BanditArm>> }>("/metrics/bandit");

export type Payment = {
  payment_id: string;
  amount: number;
  currency?: string;
  method?: string;
  bank?: string | null;
  error_code?: string | null;
  error_reason?: string | null;
  created_at?: string;
  [key: string]: unknown;
};

export const listPayments = (limit = 40) =>
  get<{ payments: Payment[]; count: number }>(`/payments?limit=${limit}`);

export type AuditEvent = {
  id?: string;
  event_type: string;
  event_data: Record<string, unknown>;
  created_at: string;
};

export type Session = {
  id?: string;
  status?: string;
  root_cause?: string | null;
  root_cause_confidence?: number | null;
  strategy?: string | null;
  decided_by?: string | null;
  attribution?: string | null;
  amount_recovered?: number | null;
  attempt_count?: number | null;
  next_action_at?: string | null;
  [key: string]: unknown;
};

export const fetchPaymentDetail = (id: string) =>
  get<{ payment: Payment; session: Session | null; audit_trail: AuditEvent[] }>(
    `/payments/${encodeURIComponent(id)}`,
  );

/** One row of the pipeline view: the payment joined to what the worker decided about it. */
export type PipelineRow = {
  payment_id: string;
  amount: number;
  currency: string;
  method: string | null;
  bank: string | null;
  error_code: string | null;
  error_reason: string | null;
  created_at: string | null;
  root_cause: string | null;
  confidence: number | null;
  strategy: string | null;
  decided_by: string | null;
  recovery_status: string | null;
  amount_recovered: number;
  shap_explanation: unknown[] | null;
  llm_reasoning: string | null;
  audit_event: string | null;
};

export type PipelineSummary = {
  total_payments: number;
  total_sessions: number;
  recovered: number;
  failed: number;
  audited_on_page: number;
};

export const getPipeline = (limit = 50) =>
  get<{ rows: PipelineRow[]; summary: PipelineSummary }>(`/pipeline/data?limit=${limit}`);

export type BatchRunResult = {
  message?: string;
  processed?: number;
  recovered?: number;
  [key: string]: unknown;
};

export const runBatch = () => post<BatchRunResult>("/batch/run");
export const generateBatch = () => post<Record<string, unknown>>("/batch/generate");

export type SandboxOutcomes = {
  status: string;
  sessions_considered?: number;
  customer_behaviour?: { paid_on_our_link: number; paid_their_own_way: number; no_response: number };
  skipped?: Record<string, number>;
  verdicts?: Record<string, number>;
  decided_here?: string;
  decided_by_production_code?: string;
  [key: string]: unknown;
};

// The fourth step, and the one the pipeline is incomplete without. `/batch/run` takes actions
// and stops — outcomes are not ours to decide, and the version of it that decided them wrote
// coin flips to the database as proven recoveries. So on a sandbox key nothing ever closes: the
// attribution split is three zeros and the overview has nothing to show. This endpoint lets the
// simulator's customers respond and routes each response through the real webhook handler, which
// is what produces a verdict, an audit event and a bandit update.
export const deliverOutcomes = () => post<SandboxOutcomes>("/sandbox/outcomes");

// ── View models ───────────────────────────────────────────────────────────────────────
//
// The feed and the detail page want a payment's *decision*, not its row: the cause the
// classifier inferred, the strategy the bandit picked, where it ended up. That is a join
// across three tables, and `/pipeline/data` is the endpoint that does it — so the list views
// read from there rather than from `/payments`, which knows only what the gateway sent.

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
  attribution?: string | null;
  timestamp: string;
};

/** `OPEN` is not a failure and must not render as one.
 *
 * A session stays OPEN until Razorpay reports a capture or a paid link, and one blocked by
 * quiet hours is OPEN with a wake-up time — a sleeping payment has not failed. The old feed had
 * no OPEN state at all, so anything not yet recovered was drawn in the failed colour. */
function statusOf(row: PipelineRow): RecoveryStatus {
  const raw = (row.recovery_status ?? "").toUpperCase();
  if (!raw) return "QUEUED";
  if (raw === "OPEN") return "IN_FLIGHT";
  return raw;
}

export async function getPayments(limit = 50): Promise<Result<PaymentSummary[]>> {
  const result = await getPipeline(limit);
  if (!result.ok) return result;
  return {
    ok: true,
    data: result.data.rows.map((row) => ({
      id: row.payment_id,
      amount: (row.amount ?? 0) / 100,
      currency: row.currency ?? "INR",
      bank: row.bank ?? row.method ?? "—",
      errorCode: row.error_code ?? "—",
      // The cause is the classifier's prediction, and the feed says so rather than
      // presenting it as the reason the payment failed.
      rootCause: row.root_cause ?? undefined,
      action: row.strategy ?? undefined,
      status: statusOf(row),
      timestamp: row.created_at ?? "",
    })),
  };
}


export const API_BASE_URL = BASE;

// ── The detail view: assembled out of the audit trail ──────────────────────────────────
//
// Every field below is read from an event the worker wrote. That is the point of the page:
// it is the audit trail rendered, so anything on it can be traced to the row that produced
// it. The version this replaced filled the same layout from `demo-data.ts` whenever the
// fetch failed — and, worse, printed three invented sentences under "Safety checks applied"
// even when the fetch succeeded, because the API had never served them.

export type ShapExplanation = {
  feature: string;
  value: number | string;
  shap_value: number;
  direction?: string;
};

/** A strategy the bandit could have picked, with its posterior mean in this context. */
export type ActionScore = { action: string; score: number; trials?: number };

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
  /** The classifier's own sentence, when it wrote one. */
  explanation?: string;
  recoveryAction: string;
  recoveryStatus: RecoveryStatus;
  recoveryAttemptedAt?: string;
  actionScores?: ActionScore[];
  /** Where the choice came from — `bandit` or `llm` — and the sentence it gave. */
  decidedBy?: string;
  reasoning?: string;
  guardrails?: string[];
  /** The attribution verdict, once a capture has been seen. Never guessed here. */
  attribution?: string | null;
};

type Trail = AuditEvent[];

const eventsOf = (trail: Trail, type: string) => trail.filter((e) => e.event_type === type);
const lastOf = (trail: Trail, ...types: string[]) =>
  [...trail].reverse().find((e) => types.includes(e.event_type));

/** Every event type that means an action went out, in both of the names the store has used.
 *
 * Mirrors `event_store.ACTION_EVENTS_READ`. The old writer stamped all five as
 * `RETRY_ATTEMPTED`, so a trail can carry either name and the detail page has to match both or
 * the "attempted at" line goes blank for half the sessions in the table. */
const ACTION_EVENTS = [
  "PAYMENT_LINK_SENT",
  "RETRY_SCHEDULED",
  "IMMEDIATE_RETRY_MOCKED",
  "DELAYED_RETRY_WOKE_UP",
  "ESCALATED",
  "RETRY_ATTEMPTED",
];

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" && value ? value : fallback;
}

function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** The bandit context the choice was drawn from, as the reasoning sentence records it.
 *
 * `bandit.py` writes `"...from context {root_cause}_{method}_{amount_bucket} (n observations)"`.
 * The amount bucket is the feature builder's, so the browser cannot reconstruct the key — but
 * it does not have to, because the sentence in the audit row already contains it. That key is
 * what makes the "other options considered" list real rather than a single row with a full bar. */
function contextKeyFrom(reasoning: string): string | null {
  return reasoning.match(/from context (\S+?)\s*\(/)?.[1] ?? null;
}

function shapFrom(event: AuditEvent | undefined): ShapExplanation[] {
  const raw = event?.event_data?.shap_explanations;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const r = item as Record<string, unknown>;
    const feature = str(r.feature);
    if (!feature) return [];
    return [
      {
        feature,
        value: typeof r.value === "number" || typeof r.value === "string" ? r.value : "—",
        shap_value: num(r.shap_value),
        direction: typeof r.direction === "string" ? r.direction : undefined,
      },
    ];
  });
}

/** What the compliance engine did to this payment, and the limits it was doing it under.
 *
 * Refusals come from the `COMPLIANCE_CHECKED` events — `blocked_by` holds the engine's own
 * sentences, with the numbers the verdict was reached on — and the remedy from
 * `COMPLIANCE_REMEDY`, which is the action the worker took *instead*. A refusal is not a
 * dropped payment, and the panel should not read as though it were. The rulebook is appended
 * after, so a payment that breached nothing still shows what it was measured against. */
function guardrailsFrom(trail: Trail, policy: CompliancePolicy | null): string[] {
  const lines: string[] = [];

  for (const event of eventsOf(trail, "COMPLIANCE_CHECKED")) {
    const blocked = event.event_data?.blocked_by;
    if (Array.isArray(blocked)) {
      for (const reason of blocked) if (typeof reason === "string") lines.push(`Refused: ${reason}`);
    }
  }

  for (const event of eventsOf(trail, "COMPLIANCE_REMEDY")) {
    const remedy = str(event.event_data?.recommendation);
    const wake = str(event.event_data?.wake_at);
    if (remedy) {
      lines.push(
        wake
          ? `Instead: ${remedy.replaceAll("_", " ").toLowerCase()}, waking at ${when(wake)}`
          : `Instead: ${remedy.replaceAll("_", " ").toLowerCase()}`,
      );
    }
  }

  for (const limit of policy?.limits ?? []) lines.push(limit.label);
  return lines;
}

/** The candidate strategies, scored by the posterior the bandit sampled from.
 *
 * Returns a single-entry list when the context has no arms yet — a cold bandit is a real
 * state and should look like one, rather than like six options that happened to tie. */
function scoresFrom(
  contextKey: string | null,
  contexts: Record<string, Record<string, BanditArm>> | null,
): ActionScore[] | undefined {
  if (!contextKey || !contexts) return undefined;
  const arms = contexts[contextKey];
  if (!arms) return undefined;
  const scored = Object.entries(arms).map(([action, arm]) => ({
    action,
    score: arm.mean,
    trials: arm.trials,
  }));
  return scored.length ? scored.sort((a, b) => b.score - a.score) : undefined;
}

export async function getPayment(id: string): Promise<Result<PaymentDetails>> {
  const [detail, bandit, policy] = await Promise.all([
    fetchPaymentDetail(id),
    getBandit(),
    getCompliancePolicy(),
  ]);
  if (!detail.ok) return detail;

  const { payment, session, audit_trail: trail = [] } = detail.data;
  const classified = lastOf(trail, "CLASSIFIED");
  const chosen = lastOf(trail, "STRATEGY_SELECTED");
  const attempted = lastOf(trail, ...ACTION_EVENTS);
  const reasoning = str(chosen?.event_data?.reasoning);

  // The session row is the worker's conclusion and the audit event is its working; where both
  // exist they agree, and the row wins because a replay rewrites it.
  const rootCause = session?.root_cause ?? str(classified?.event_data?.root_cause, "UNCLASSIFIED");
  const rawConfidence = session?.root_cause_confidence ?? classified?.event_data?.confidence;

  return {
    ok: true,
    data: {
      id: payment.payment_id ?? id,
      amount: num(payment.amount) / 100,
      currency: payment.currency ?? "INR",
      bank: payment.bank ?? payment.method ?? "—",
      timestamp: payment.created_at ?? trail[0]?.created_at ?? "",
      initialErrorCode: payment.error_code ?? "—",
      userId: str(payment.customer_id) || str(payment.customer_contact) || str(payment.customer_email) || "—",
      rootCause,
      confidence: num(rawConfidence),
      shapExplanations: shapFrom(classified),
      explanation: str(classified?.event_data?.explanation_summary) || undefined,
      recoveryAction: session?.strategy ?? str(chosen?.event_data?.strategy, "PENDING"),
      recoveryStatus: (session?.status ?? "QUEUED").toUpperCase() === "OPEN"
        ? "IN_FLIGHT"
        : (session?.status ?? "QUEUED").toUpperCase(),
      recoveryAttemptedAt: attempted?.created_at,
      actionScores: scoresFrom(
        contextKeyFrom(reasoning),
        bandit.ok ? bandit.data.contexts : null,
      ),
      decidedBy: session?.decided_by ?? (str(chosen?.event_data?.decided_by) || undefined),
      reasoning: reasoning || undefined,
      guardrails: guardrailsFrom(trail, policy.ok ? policy.data : null),
      attribution: session?.attribution ?? null,
    },
  };
}

