"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AppShell } from "./app-shell";
import { ArrowUpRight, Check, Spark } from "./icons";
import { demoPayments } from "../lib/demo-data";
import { getPayments, runBatchRecovery, type BatchRunResult, type PaymentSummary } from "../lib/api";

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

export function Dashboard() {
  const [payments, setPayments] = useState<PaymentSummary[]>(demoPayments);
  const [isLoading, setIsLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);
  const [batchState, setBatchState] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [batchResult, setBatchResult] = useState<BatchRunResult | null>(null);

  const loadPayments = useCallback(async () => {
    try {
      const livePayments = await getPayments();
      if (livePayments.length) setPayments(livePayments);
      setIsDemo(livePayments.length === 0);
    } catch {
      setIsDemo(true);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPayments();
  }, [loadPayments]);

  async function handleBatchRun() {
    setBatchState("running");
    setBatchResult(null);
    try {
      const result = await runBatchRecovery();
      setBatchResult(result);
      setBatchState("complete");
      await loadPayments();
    } catch {
      setBatchState("error");
    }
  }

  const recovered = payments.filter((item) => item.status === "RECOVERED").length;
  const active = payments.filter((item) => item.status === "IN_FLIGHT" || item.status === "QUEUED").length;
  const atRisk = payments.filter((item) => item.status !== "RECOVERED").reduce((sum, item) => sum + item.amount, 0);
  const restoredValue = payments.filter((item) => item.status === "RECOVERED").reduce((sum, item) => sum + item.amount, 0);
  const recoveryRate = payments.length ? Math.round((recovered / payments.length) * 100) : 0;

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

      {batchState !== "idle" && (
        <div className={"simple-notice " + (batchState === "error" ? "error" : "")} role="status">
          {batchState === "running" && <><span className="spinner" /> Analysing payments and selecting recovery actions…</>}
          {batchState === "complete" && <><Check /> Batch complete{batchResult?.recovered !== undefined ? " — " + batchResult.recovered + " payments recovered" : "."}</>}
          {batchState === "error" && <>Could not run the recovery batch. Check the FastAPI server and try again.</>}
        </div>
      )}

      <section className="simple-summary-grid" aria-label="Recovery metrics">
        <article className="simple-summary-main">
          <span>Recovered today</span>
          <strong>{isLoading ? "—" : money.format(restoredValue)}</strong>
          <p>{isLoading ? "Loading payments…" : recovered + " payment" + (recovered === 1 ? "" : "s") + " successfully recovered"}</p>
          <div className="simple-progress"><span><b>{recoveryRate}%</b> recovery rate</span><i><em style={{ width: recoveryRate + "%" }} /></i></div>
        </article>
        <article className="simple-stat">
          <span>Revenue at risk</span>
          <strong>{isLoading ? "—" : money.format(atRisk)}</strong>
          <p>Across payments that still need action</p>
        </article>
        <article className="simple-stat">
          <span>In progress</span>
          <strong>{isLoading ? "—" : active}</strong>
          <p>Recovery actions currently running</p>
        </article>
      </section>

      <section className="simple-process-section">
        <div className="simple-section-title">
          <div><p className="simple-eyebrow">HOW PAYREVIVE WORKS</p><h2>One clear path from failure to recovery.</h2></div>
          <span className={isDemo ? "simple-data-badge sample" : "simple-data-badge"}>{isDemo ? "Showing sample data" : "Live data"}</span>
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
        </div>
      </section>
    </AppShell>
  );
}
