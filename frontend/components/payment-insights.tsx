"use client";

import Link from "next/link";
import { useCallback, useEffect, useState, type CSSProperties } from "react";
import { AppShell } from "./app-shell";
import { ArrowLeft, Check, Spark } from "./icons";
import { demoPaymentDetails } from "../lib/demo-data";
import { getPayment, type PaymentDetails, type RecoveryStatus } from "../lib/api";

const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

function formatTime(timestamp: string) {
  const date = new Date(timestamp);
  return Number.isNaN(date.valueOf()) ? timestamp || "—" : new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function readable(value: string) {
  return value.replaceAll("_", " ");
}

function StatusChip({ status }: { status: RecoveryStatus }) {
  return <span className={"simple-status " + status.toLowerCase()}>{readable(status)}</span>;
}

export function PaymentInsights({ paymentId }: { paymentId: string }) {
  const [payment, setPayment] = useState<PaymentDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);

  const loadPayment = useCallback(async () => {
    setLoading(true);
    try {
      setPayment(await getPayment(paymentId));
      setIsDemo(false);
    } catch {
      setPayment({ ...demoPaymentDetails, id: paymentId });
      setIsDemo(true);
    } finally {
      setLoading(false);
    }
  }, [paymentId]);

  useEffect(() => {
    void loadPayment();
  }, [loadPayment]);

  if (loading || !payment) {
    return <AppShell><div className="simple-loading"><span className="spinner" /> Loading payment insight…</div></AppShell>;
  }

  const confidence = payment.confidence > 1 ? payment.confidence : payment.confidence * 100;
  const maxImpact = Math.max(...payment.shapExplanations.map((item) => Math.abs(item.shap_value)), 0.01);
  const actionScores = payment.actionScores?.length
    ? [...payment.actionScores].sort((a, b) => b.score - a.score)
    : [{ action: payment.recoveryAction, score: 1 }];
  const maxActionScore = Math.max(...actionScores.map((item) => item.score), 0.01);
  const guardrails = payment.guardrails?.length
    ? payment.guardrails
    : ["One recovery action per customer in 24 hours", "Stop the workflow immediately after payment succeeds", "Respect consent and contact-time limits"];

  return (
    <AppShell active="feed">
      <header className="simple-detail-header">
        <Link href="/" className="simple-back-link"><ArrowLeft /> Back to recovery queue</Link>
        {isDemo && <span className="simple-data-badge sample">Showing sample data</span>}
      </header>

      <section className="simple-case-heading">
        <div><p className="simple-eyebrow">PAYMENT INSIGHT</p><h1>Why did this payment fail?</h1><p>Review the transaction, the model’s evidence, and the recovery action in one place.</p></div>
        <div className="simple-case-amount"><span>Failed payment</span><strong>{money.format(payment.amount)}</strong><small>{payment.id}</small></div>
      </section>

      <section className="simple-facts" aria-label="Transaction facts">
        <div><span>Bank</span><b>{payment.bank}</b></div>
        <div><span>Initial error</span><b className="simple-error">{payment.initialErrorCode}</b></div>
        <div><span>Time</span><b>{formatTime(payment.timestamp)}</b></div>
        <div><span>User ID</span><b>{payment.userId}</b></div>
        <div><span>Recovery status</span><StatusChip status={payment.recoveryStatus} /></div>
      </section>

      <section className="simple-decision-card">
        <div className="simple-card-label"><b>1</b><span>AI DIAGNOSIS</span></div>
        <div className="simple-diagnosis">
          <div><span>The model predicts the payment failed because of</span><h2>{readable(payment.rootCause)}</h2><p>Confidence is based on the payment, customer and issuer signals available at the time of failure.</p></div>
          <div className="simple-confidence"><strong>{Math.round(confidence)}%</strong><span>model confidence</span></div>
        </div>
      </section>

      <section className="simple-explanation-card">
        <div className="simple-card-header"><div><div className="simple-card-label"><b>2</b><span>WHY THE MODEL THINKS THIS</span></div><h2>Evidence behind the diagnosis</h2><p>Higher bars had more influence on the prediction.</p></div></div>
        <div className="simple-explanation-list">
          {payment.shapExplanations.length ? payment.shapExplanations.map((item, index) => {
            const supports = item.shap_value >= 0;
            const impact = Math.max(8, Math.abs(item.shap_value) / maxImpact * 100);
            return <div className="simple-explanation-row" key={item.feature + index}>
              <div><b>{item.feature}</b><span>{item.value}</span></div>
              <i><em className={supports ? "supports" : "opposes"} style={{ width: impact + "%" }} /></i>
              <p><strong>{supports ? "Makes " + readable(payment.rootCause) + " more likely" : "Makes " + readable(payment.rootCause) + " less likely"}</strong></p>
            </div>;
          }) : <p className="simple-empty">No explanation values were returned for this payment.</p>}
        </div>
      </section>

      <section className="simple-action-card">
        <div className="simple-card-label"><b>3</b><span>RECOVERY ACTION</span></div>
        <div className="simple-action-heading">
          <div className="simple-action-icon"><Spark /></div>
          <div><span>Recommended action</span><h2>{readable(payment.recoveryAction)}</h2><p>This action had the best expected chance of recovering the payment without over-contacting the customer.</p></div>
          <div className="simple-action-status"><span>Current result</span><StatusChip status={payment.recoveryStatus} /><small>{payment.recoveryAttemptedAt ? "Attempted " + formatTime(payment.recoveryAttemptedAt) : "Waiting to run"}</small></div>
        </div>
        <div className="simple-action-details">
          <div>
            <h3>Other options considered</h3>
            {actionScores.map((candidate) => {
              const selected = candidate.action === payment.recoveryAction;
              const score = candidate.score > 1 ? candidate.score : candidate.score * 100;
              const width = Math.max(5, candidate.score / maxActionScore * 100);
              return <div className={"simple-option " + (selected ? "selected" : "")} key={candidate.action}><span>{readable(candidate.action)}</span><i><em style={{ width: width + "%" } as CSSProperties} /></i><b>{score.toFixed(0)}%</b></div>;
            })}
          </div>
          <div className="simple-guardrails">
            <h3>Safety checks applied</h3>
            <ul>{guardrails.map((guardrail) => <li key={guardrail}><Check />{guardrail}</li>)}</ul>
          </div>
        </div>
      </section>
    </AppShell>
  );
}
