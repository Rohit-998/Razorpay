"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AppShell } from "./app-shell";
import { ArrowUpRight, Check, Spark } from "./icons";
import {
  deliverOutcomes,
  getPayments,
  getStats,
  runBatch,
  type DashboardStats,
  type Failure,
  type PaymentSummary,
  type SandboxOutcomes,
} from "../lib/api";
import { paise, share } from "../lib/format";

const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

function conciseTime(timestamp: string) {
  const date = new Date(timestamp);
  return Number.isNaN(date.valueOf())
    ? timestamp
    : new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false }).format(date);
}

function statusLabel(status: string) {
  return status.replaceAll("_", " ");
}

/**
 * The recovery overview.
 *
 * Every figure on this page is served. The version this replaced computed four of them in the
 * browser from whatever page of payments it had fetched — including `recovered / total`, drawn
 * as a progress bar labelled "recovery rate", which is the single number `reports/REPORT.md`
 * argues is not a measurement of anything: it counts every customer who would have paid anyway
 * as a win. `/dashboard/stats` deliberately does not serve it. What it does serve is the
 * attribution split, and the bar now shows the share of recovered rupees Razorpay's webhook
 * named our payment link for — a claim we can defend.
 *
 * And when the API is unreachable the page says so, with the command that fixes it. It used to
 * fall back to `lib/demo-data.ts` behind a small "Showing sample data" badge: invented rows,
 * invented root causes, rendered as a working product.
 */
export function Dashboard() {
  const [payments, setPayments] = useState<PaymentSummary[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [batchState, setBatchState] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [batchResult, setBatchResult] = useState<SandboxOutcomes | null>(null);
  const [batchError, setBatchError] = useState<Failure | null>(null);

  const loadPayments = useCallback(async () => {
    const [feed, summary] = await Promise.all([getPayments(), getStats()]);
    if (feed.ok) setPayments(feed.data);
    if (summary.ok) setStats(summary.data);
    setFailure(feed.ok ? (summary.ok ? null : summary.error) : feed.error);
    setIsLoading(false);
  }, []);

  useEffect(() => {
    void loadPayments();
  }, [loadPayments]);

  // Two calls, because the pipeline is two things. `/batch/run` classifies, checks compliance
  // and takes one action per payment — and stops, because deciding the outcome is exactly what
  // the deleted version of it did with `random.random()`. `/sandbox/outcomes` then lets the
  // simulator's customers respond and routes each response through the real webhook handler,
  // which is what produces a verdict. Reporting "batch complete" after the first call alone
  // would put a number on the screen for work that has not finished.
  async function handleBatchRun() {
    setBatchState("running");
    setBatchResult(null);
    setBatchError(null);
    const result = await runBatch();
    if (!result.ok) {
      setBatchError(result.error);
      setBatchState("error");
      return;
    }
    const observed = await deliverOutcomes();
    if (!observed.ok) {
      setBatchError(observed.error);
      setBatchState("error");
      return;
    }
    setBatchResult(observed.data);
    setBatchState("complete");
    await loadPayments();
  }

  const ours = stats?.provably_ours;

  return (
    <AppShell active="feed">
      <header className="simple-page-header">
        <div>
          <p className="simple-eyebrow">PAYMENT RECOVERY</p>
          <h1>Recovery overview</h1>
          <p>PayRevive explains every failed payment and chooses the safest next recovery action.</p>
        </div>
        <button onClick={handleBatchRun} className="simple-primary-button" disabled={batchState === "running"}>
          <Spark />
          {batchState === "running" ? "Running recovery…" : "Run AI batch recovery"}
        </button>
      </header>

      {failure && (
        <div className="simple-notice error" role="status">
          {failure.message}{failure.fix ? " Run: " + failure.fix : ""}
        </div>
      )}

      {batchState !== "idle" && (
        <div className={"simple-notice " + (batchState === "error" ? "error" : "")} role="status">
          {batchState === "running" && <><span className="spinner" /> Analysing payments, selecting recovery actions, and waiting for outcomes…</>}
          {batchState === "complete" && <><Check /> Batch complete{batchResult?.verdicts ? " — " + (batchResult.verdicts.SYSTEM_RECOVERED ?? 0) + " paid on our link, " + (batchResult.verdicts.CUSTOMER_SELF_RECOVERED ?? 0) + " came back on their own, " + (batchResult.verdicts.AMBIGUOUS ?? 0) + " unprovable" : "."}</>}
          {batchState === "error" && <>{batchError?.message ?? "Could not run the recovery batch."}{batchError?.fix ? " Run: " + batchError.fix : ""}</>}
        </div>
      )}

      <section className="simple-summary-grid" aria-label="Recovery metrics">
        <article className="simple-summary-main">
          <span>Recovered — provably ours</span>
          <strong>{ours ? paise(ours.amount_paise) : "—"}</strong>
          <p>{isLoading ? "Loading payments…" : !ours ? "No attribution data yet" : ours.sessions + " payment" + (ours.sessions === 1 ? "" : "s") + " Razorpay reported as paid on our link" + (ours.unestablished_sessions ? ". " + ours.unestablished_sessions + " more came back with no callback to say why — excluded, not assumed" : "")}</p>
          <div className="simple-progress">
            <span><b>{ours ? share(ours.established.share_of_established_sessions, 0) : "—"}</b> of recoveries with an established cause{ours && ours.established.self_recovered_sessions ? " — the other " + ours.established.self_recovered_sessions + " came back on their own" : ""}</span>
            <i><em style={{ width: (ours ? ours.established.share_of_established_sessions * 100 : 0) + "%" }} /></i>
          </div>
        </article>
        <article className="simple-stat">
          <span>Still not recovered</span>
          <strong>{stats ? paise(stats.unrecovered_paise) : "—"}</strong>
          <p>{stats ? "Of " + paise(stats.at_risk_paise) + " in failed payments" : "Across payments that still need action"}</p>
        </article>
        <article className="simple-stat">
          <span>In progress</span>
          <strong>{stats ? stats.open : "—"}</strong>
          <p>Sessions still inside the recovery window</p>
        </article>
      </section>

      <section className="simple-process-section">
        <div className="simple-section-title">
          <div><p className="simple-eyebrow">HOW PAYREVIVE WORKS</p><h2>One clear path from failure to recovery.</h2></div>
          <span className={failure ? "simple-data-badge sample" : "simple-data-badge"}>{failure ? "API unreachable" : "Live data"}</span>
        </div>
        <ol className="simple-process">
          <li><b>1</b><div><strong>Detect a failed payment</strong><span>Capture payment, bank, error and customer context.</span></div></li>
          <li><b>2</b><div><strong>Explain the root cause</strong><span>The model predicts why it failed and shows the evidence.</span></div></li>
          <li><b>3</b><div><strong>Take the right action</strong><span>The recovery policy selects a safe intervention with guardrails.</span></div></li>
        </ol>
      </section>

      <section className="simple-feed-section">
        <div className="simple-section-title">
          <div><p className="simple-eyebrow">RECOVERY QUEUE</p><h2>Payments needing attention</h2><p className="simple-section-copy">Open any payment to see the complete AI decision and recovery result.</p></div>
        </div>
        <div className="simple-feed" role="list">
          <div className="simple-feed-head" aria-hidden="true"><span>Payment</span><span>AI diagnosis</span><span>Recommended action</span><span>Status</span><span /></div>
          {isLoading ? [...Array(5)].map((_, index) => <div className="simple-feed-skeleton" key={index}><i /><i /><i /><i /></div>) : payments.map((payment) => (
            <Link href={"/payment/" + encodeURIComponent(payment.id)} className="simple-payment-row" role="listitem" key={payment.id}>
              <span className="simple-payment-id"><b>{payment.id}</b><small>{money.format(payment.amount)} · {payment.bank} · {conciseTime(payment.timestamp)}</small></span>
              <span className="simple-cause">{payment.rootCause ?? payment.errorCode}</span>
              <span className="simple-action">{payment.action ?? "Analysing"}</span>
              <span className={"simple-status " + payment.status.toLowerCase()}>{statusLabel(payment.status)}</span>
              <ArrowUpRight />
            </Link>
          ))}
          {!isLoading && !payments.length && !failure && (
            <p className="simple-empty">No failed payments in the queue. Run a batch to generate some.</p>
          )}
        </div>
      </section>
    </AppShell>
  );
}
