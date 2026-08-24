"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AppShell } from "../../components/app-shell";
import { Spark, Check, ArrowUpRight } from "../../components/icons";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");
const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

type StepState = "pending" | "running" | "complete" | "error";

type PipelineRow = {
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

type PipelineSummary = {
  total_payments: number;
  total_sessions: number;
  recovered: number;
  failed: number;
  audit_events: number;
};

function conciseDate(ts: string | null) {
  if (!ts) return "—";
  const d = new Date(ts);
  return Number.isNaN(d.valueOf()) ? ts : new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short" }).format(d);
}

function confidencePct(c: number | null) {
  if (c === null || c === undefined) return "—";
  return (c * 100).toFixed(1) + "%";
}

function statusClass(status: string | null) {
  if (!status) return "";
  const s = status.toLowerCase();
  if (s === "recovered") return "recovered";
  if (s === "failed") return "failed";
  if (s === "escalated") return "escalated";
  return "in_flight";
}

export default function PipelinePage() {
  const [rows, setRows] = useState<PipelineRow[]>([]);
  const [summary, setSummary] = useState<PipelineSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"table" | "pipeline">("table");
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  /* Pipeline runner state */
  const [generateState, setGenerateState] = useState<StepState>("pending");
  const [trainState, setTrainState] = useState<StepState>("pending");
  const [recoverState, setRecoverState] = useState<StepState>("pending");
  const [trainResults, setTrainResults] = useState<Record<string, unknown> | null>(null);
  const [batchResults, setBatchResults] = useState<Record<string, unknown> | null>(null);

  const loadData = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/pipeline/data?limit=50`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setRows(data.rows || []);
      setSummary(data.summary || null);
    } catch {
      /* keep empty */
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  async function runStep(url: string, setState: (s: StepState) => void, setResult?: (r: Record<string, unknown>) => void) {
    setState("running");
    try {
      const res = await fetch(`${API_BASE_URL}${url}`, { method: "POST", headers: { "Content-Type": "application/json" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult?.(data);
      setState("complete");
      return true;
    } catch {
      setState("error");
      return false;
    }
  }

  async function handleFullPipeline() {
    setGenerateState("pending");
    setTrainState("pending");
    setRecoverState("pending");
    setTrainResults(null);
    setBatchResults(null);

    const ok1 = await runStep("/api/v1/batch/generate", setGenerateState);
    if (!ok1) return;

    const ok2 = await runStep("/api/v1/model/train", setTrainState, setTrainResults);
    if (!ok2) return;

    const ok3 = await runStep("/api/v1/batch/run", setRecoverState, setBatchResults);
    if (ok3) {
      await loadData(); // Reload table data after pipeline completes
    }
  }

  function stepClass(state: StepState) {
    return "pipeline-step " + state;
  }

  const isRunning = generateState === "running" || trainState === "running" || recoverState === "running";

  return (
    <AppShell active="pipeline">
      <header className="simple-page-header">
        <div>
          <p className="simple-eyebrow">AI PIPELINE</p>
          <h1>Recovery pipeline &amp; data</h1>
          <p>View live Supabase data flowing through the AI pipeline — payments, root-cause classification, and recovery outcomes.</p>
        </div>
        <button onClick={handleFullPipeline} className="simple-primary-button" disabled={isRunning}>
          <Spark />
          {isRunning ? "Running…" : "Run full pipeline"}
        </button>
      </header>

      {/* Summary stats */}
      {summary && (
        <section className="simple-summary-grid" style={{ marginBottom: 24 }}>
          <article className="simple-summary-main">
            <span>Pipeline throughput</span>
            <strong>{summary.total_sessions} sessions</strong>
            <p>{summary.total_payments} payments processed through the AI pipeline</p>
            <div className="simple-progress">
              <span><b>{summary.total_sessions > 0 ? Math.round(summary.recovered / summary.total_sessions * 100) : 0}%</b> recovery rate</span>
              <i><em style={{ width: (summary.total_sessions > 0 ? Math.round(summary.recovered / summary.total_sessions * 100) : 0) + "%" }} /></i>
            </div>
          </article>
          <article className="simple-stat">
            <span>Recovered</span>
            <strong style={{ color: "#167a4d" }}>{summary.recovered}</strong>
            <p>Payments successfully recovered by AI</p>
          </article>
          <article className="simple-stat">
            <span>Failed / Escalated</span>
            <strong style={{ color: "#c0524b" }}>{summary.failed}</strong>
            <p>Payments that could not be auto-recovered</p>
          </article>
        </section>
      )}

      {/* Tab switcher */}
      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid #dce6df", marginBottom: 20 }}>
        <button
          onClick={() => setActiveTab("table")}
          style={{
            padding: "10px 20px", fontSize: 12, fontWeight: 600, cursor: "pointer",
            background: "none", border: "none", borderBottom: activeTab === "table" ? "2px solid #167a4d" : "2px solid transparent",
            color: activeTab === "table" ? "#167a4d" : "#718078",
          }}
        >
          Data table
        </button>
        <button
          onClick={() => setActiveTab("pipeline")}
          style={{
            padding: "10px 20px", fontSize: 12, fontWeight: 600, cursor: "pointer",
            background: "none", border: "none", borderBottom: activeTab === "pipeline" ? "2px solid #167a4d" : "2px solid transparent",
            color: activeTab === "pipeline" ? "#167a4d" : "#718078",
          }}
        >
          Pipeline runner
        </button>
      </div>

      {activeTab === "table" && (
        <section>
          {isLoading ? (
            <div style={{ padding: 40, textAlign: "center", color: "#718078", fontSize: 13 }}>Loading pipeline data from Supabase…</div>
          ) : rows.length === 0 ? (
            <div style={{ padding: 40, textAlign: "center", color: "#718078", fontSize: 13 }}>No pipeline data found. Click &quot;Run full pipeline&quot; to generate, train, and recover.</div>
          ) : (
            <div style={{ overflowX: "auto", borderRadius: 10, border: "1px solid #dce6df" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, background: "#fff" }}>
                <thead>
                  <tr style={{ background: "#f5f8f6", borderBottom: "1px solid #dce6df" }}>
                    <th style={thStyle}>Payment ID</th>
                    <th style={thStyle}>Amount</th>
                    <th style={thStyle}>Bank</th>
                    <th style={thStyle}>Error</th>
                    <th style={thStyle}>Root cause</th>
                    <th style={thStyle}>Confidence</th>
                    <th style={thStyle}>Strategy</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Recovered</th>
                    <th style={{ ...thStyle, width: 30 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <>
                      <tr
                        key={row.payment_id}
                        onClick={() => setExpandedRow(expandedRow === row.payment_id ? null : row.payment_id)}
                        style={{
                          borderBottom: "1px solid #eef2f0",
                          cursor: "pointer",
                          transition: "background .15s",
                          background: expandedRow === row.payment_id ? "#f0f7f3" : "transparent",
                        }}
                        onMouseEnter={(e) => { if (expandedRow !== row.payment_id) e.currentTarget.style.background = "#fafcfb"; }}
                        onMouseLeave={(e) => { if (expandedRow !== row.payment_id) e.currentTarget.style.background = "transparent"; }}
                      >
                        <td style={tdStyle}>
                          <Link href={"/payment/" + encodeURIComponent(row.payment_id)} style={{ color: "#167a4d", fontWeight: 600, textDecoration: "none" }} onClick={(e) => e.stopPropagation()}>
                            {row.payment_id}
                          </Link>
                        </td>
                        <td style={tdStyle}>{money.format(row.amount)}</td>
                        <td style={tdStyle}>{row.bank ?? "—"}</td>
                        <td style={tdStyle}><code style={{ fontSize: 10, background: "#f5f8f6", padding: "2px 5px", borderRadius: 3 }}>{row.error_code ?? "—"}</code></td>
                        <td style={tdStyle}><span style={{ fontWeight: 600, color: "#23332b" }}>{row.root_cause ?? "—"}</span></td>
                        <td style={tdStyle}>{confidencePct(row.confidence)}</td>
                        <td style={tdStyle}>{row.strategy ?? "—"}</td>
                        <td style={tdStyle}>
                          <span className={"simple-status " + statusClass(row.recovery_status)}>
                            {row.recovery_status?.replaceAll("_", " ") ?? "PENDING"}
                          </span>
                        </td>
                        <td style={tdStyle}>{row.amount_recovered ? money.format(row.amount_recovered) : "—"}</td>
                        <td style={tdStyle}>
                          <span style={{ transform: expandedRow === row.payment_id ? "rotate(90deg)" : "rotate(0)", display: "inline-block", transition: "transform .2s", color: "#718078" }}>▸</span>
                        </td>
                      </tr>
                      {expandedRow === row.payment_id && (
                        <tr key={row.payment_id + "-detail"}>
                          <td colSpan={10} style={{ padding: 0 }}>
                            <div style={{ background: "#f8faf9", padding: "16px 20px", borderBottom: "2px solid #dce6df" }}>
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, fontSize: 12 }}>
                                <div>
                                  <p style={{ color: "#718078", marginBottom: 6, fontWeight: 600, textTransform: "uppercase", fontSize: 10, letterSpacing: ".05em" }}>Payment details</p>
                                  <div style={{ display: "grid", gridTemplateColumns: "110px 1fr", gap: "4px 10px", color: "#23332b" }}>
                                    <span style={{ color: "#718078" }}>Method</span><span>{row.method ?? "—"}</span>
                                    <span style={{ color: "#718078" }}>Error reason</span><span>{row.error_reason ?? "—"}</span>
                                    <span style={{ color: "#718078" }}>Created at</span><span>{conciseDate(row.created_at)}</span>
                                    <span style={{ color: "#718078" }}>Decided by</span><span>{row.decided_by ?? "—"}</span>
                                    <span style={{ color: "#718078" }}>Audit event</span><span>{row.audit_event ?? "—"}</span>
                                  </div>
                                </div>
                                <div>
                                  {row.shap_explanation && Array.isArray(row.shap_explanation) && row.shap_explanation.length > 0 && (
                                    <>
                                      <p style={{ color: "#718078", marginBottom: 6, fontWeight: 600, textTransform: "uppercase", fontSize: 10, letterSpacing: ".05em" }}>SHAP explanation</p>
                                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: "3px 10px", fontSize: 11 }}>
                                        {(row.shap_explanation as Array<Record<string, unknown>>).slice(0, 4).map((s, i) => (
                                          <div key={i} style={{ display: "contents" }}>
                                            <span style={{ color: "#23332b", fontWeight: 500 }}>{String(s.feature)}</span>
                                            <span style={{ color: "#718078" }}>{String(s.value)}</span>
                                            <span style={{ color: Number(s.shap_value) > 0 ? "#167a4d" : "#c0524b", fontWeight: 600, fontFamily: "var(--mono)" }}>
                                              {Number(s.shap_value) > 0 ? "+" : ""}{Number(s.shap_value).toFixed(3)}
                                            </span>
                                          </div>
                                        ))}
                                      </div>
                                    </>
                                  )}
                                  {row.llm_reasoning && (
                                    <>
                                      <p style={{ color: "#718078", marginTop: 12, marginBottom: 4, fontWeight: 600, textTransform: "uppercase", fontSize: 10, letterSpacing: ".05em" }}>LLM reasoning</p>
                                      <p style={{ color: "#23332b", fontSize: 11, lineHeight: 1.5 }}>{String(row.llm_reasoning).slice(0, 300)}{String(row.llm_reasoning).length > 300 ? "…" : ""}</p>
                                    </>
                                  )}
                                </div>
                              </div>
                              <div style={{ marginTop: 12 }}>
                                <Link
                                  href={"/payment/" + encodeURIComponent(row.payment_id)}
                                  style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "#167a4d", fontSize: 11, fontWeight: 600, textDecoration: "none" }}
                                >
                                  View full detail <ArrowUpRight />
                                </Link>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {activeTab === "pipeline" && (
        <section>
          <ol className="simple-process">
            <li className={stepClass(generateState)}>
              <b>{generateState === "complete" ? "✓" : "1"}</b>
              <div>
                <strong>Generate training data</strong>
                <span>{generateState === "running" ? "Generating synthetic payment data…" : generateState === "complete" ? "Training data generated successfully" : generateState === "error" ? "Failed to generate data" : "Creates labelled payment data from Supabase"}</span>
              </div>
            </li>
            <li className={stepClass(trainState)}>
              <b>{trainState === "complete" ? "✓" : "2"}</b>
              <div>
                <strong>Train XGBoost classifier</strong>
                <span>{trainState === "running" ? "Training model on generated data…" : trainState === "complete" ? `Training complete${trainResults && typeof trainResults === "object" && "metrics" in trainResults ? " — accuracy " + ((trainResults.metrics as Record<string, unknown>).accuracy as number * 100).toFixed(0) + "%" : ""}` : trainState === "error" ? "Training failed" : "Fits the root-cause classifier and evaluates on holdout set"}</span>
              </div>
            </li>
            <li className={stepClass(recoverState)}>
              <b>{recoverState === "complete" ? "✓" : "3"}</b>
              <div>
                <strong>Run batch recovery</strong>
                <span>{recoverState === "running" ? "Processing failed payments…" : recoverState === "complete" && batchResults ? `Done — ${(batchResults as Record<string, unknown>).recovered ?? 0} recovered` : recoverState === "error" ? "Recovery failed" : "Classifies and recovers all pending failed payments"}</span>
              </div>
            </li>
          </ol>

          {batchResults && (
            <article className="json-panel" style={{ marginTop: 24 }}>
              <div><span>PIPELINE RESULTS</span><small>Batch recovery output</small></div>
              <pre>{JSON.stringify(batchResults, null, 2)}</pre>
            </article>
          )}
        </section>
      )}
    </AppShell>
  );
}

const thStyle: React.CSSProperties = {
  padding: "10px 12px",
  textAlign: "left",
  fontWeight: 600,
  color: "#718078",
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: ".04em",
  whiteSpace: "nowrap",
};

const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  verticalAlign: "middle",
  whiteSpace: "nowrap",
};
