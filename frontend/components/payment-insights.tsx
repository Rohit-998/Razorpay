"use client";

import Link from "next/link";
import { useCallback, useEffect, useState, type CSSProperties } from "react";
import { AppShell } from "./app-shell";
import { ArrowLeft, Check, Spark } from "./icons";
import { getPayment, type Failure, type PaymentDetails, type RecoveryStatus } from "../lib/api";
import { share, when } from "../lib/format";

const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

function readable(value: string) {
  return value.replaceAll("_", " ");
}

function StatusChip({ status }: { status: RecoveryStatus }) {
  return <span className={"simple-status " + status.toLowerCase()}>{readable(status)}</span>;
}

/**
 * One payment, as its audit trail records it.
 *
 * Nothing on this page is written here. The diagnosis and its SHAP rows come from the
 * `CLASSIFIED` event, the action and the sentence justifying it from `STRATEGY_SELECTED`, the
 * candidate scores from the bandit's posterior for the exact context that decision was drawn
 * from, and the safety panel from the `COMPLIANCE_CHECKED` refusals plus the limits
 * `/compliance/policy` reports as in force.
 *
 * That last one is the reason this file changed. The panel headed "Safety checks applied" used
 * to print three hard-coded sentences whenever the API did not supply any — which was always,
 * because no endpoint served them. Two of the three were false: the limit is two contacts a
 * day, not one, and there is no consent field in the schema to respect. Under a heading
 * claiming compliance, next to a checkmark.
 */
export function PaymentInsights({ paymentId }: { paymentId: string }) {
  const [payment, setPayment] = useState<PaymentDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState<Failure | null>(null);

  const loadPayment = useCallback(async () => {
    setLoading(true);
    const result = await getPayment(paymentId);
    if (result.ok) {
      setPayment(result.data);
      setFailure(null);
    } else {
      setPayment(null);
      setFailure(result.error);
    }
    setLoading(false);
  }, [paymentId]);

  useEffect(() => {
    void loadPayment();
  }, [loadPayment]);

  if (loading) {
    return <AppShell><div className="simple-loading"><span className="spinner" /> Loading payment insight…</div></AppShell>;
  }

  if (!payment) {
    return (
      <AppShell active="feed">
        <header className="simple-detail-header">
          <Link href="/" className="simple-back-link"><ArrowLeft /> Back to recovery queue</Link>
        </header>
        <div className="simple-notice error" role="status">
          {failure?.message ?? "Could not load this payment."}
          {failure?.fix ? " Run: " + failure.fix : ""}
        </div>
      </AppShell>
    );
  }

  const confidence = payment.confidence > 1 ? payment.confidence : payment.confidence * 100;
  const maxImpact = Math.max(...payment.shapExplanations.map((item) => Math.abs(item.shap_value)), 0.01);
  const actionScores = payment.actionScores ?? [];
  const maxActionScore = Math.max(...actionScores.map((item) => item.score), 0.01);
  const guardrails = payment.guardrails ?? [];

  return (
    <AppShell active="feed">
      <header className="simple-detail-header">
        <Link href="/" className="simple-back-link"><ArrowLeft /> Back to recovery queue</Link>
        {payment.attribution && <span className="simple-data-badge">{readable(payment.attribution)}</span>}
      </header>

      <section className="simple-case-heading">
        <div><p className="simple-eyebrow">PAYMENT INSIGHT</p><h1>Why did this payment fail?</h1><p>Review the transaction, the model’s evidence, and the recovery action in one place.</p></div>
        <div className="simple-case-amount"><span>Failed payment</span><strong>{money.format(payment.amount)}</strong><small>{payment.id}</small></div>
      </section>

      <section className="simple-facts" aria-label="Transaction facts">
        <div><span>Bank</span><b>{payment.bank}</b></div>
        <div><span>Initial error</span><b className="simple-error">{payment.initialErrorCode}</b></div>
        <div><span>Time</span><b>{when(payment.timestamp)}</b></div>
        <div><span>User ID</span><b>{payment.userId}</b></div>
        <div><span>Recovery status</span><StatusChip status={payment.recoveryStatus} /></div>
      </section>

      <section className="simple-decision-card">
        <div className="simple-card-label"><b>1</b><span>AI DIAGNOSIS</span></div>
        <div className="simple-diagnosis">
          <div>
            <span>The model predicts the payment failed because of</span>
            <h2>{readable(payment.rootCause)}</h2>
            {/* The classifier's own sentence when it wrote one. It is a prediction from the
                error fields and the payment context, and inference from those alone is bounded
                at 68.38% under this cause mix — so the page says "predicts", not "failed
                because of", and a confusable cause should be read as a shortlist. */}
            <p>{payment.explanation ?? "Predicted from the payment, customer and issuer signals available at the time of failure."}</p>
          </div>
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
          <div>
            <span>{payment.decidedBy ? "Chosen by " + readable(payment.decidedBy) : "Recommended action"}</span>
            <h2>{readable(payment.recoveryAction)}</h2>
            <p>{payment.reasoning ?? "Selected by the recovery policy under the limits listed below."}</p>
          </div>
          <div className="simple-action-status"><span>Current result</span><StatusChip status={payment.recoveryStatus} /><small>{payment.recoveryAttemptedAt ? "Attempted " + when(payment.recoveryAttemptedAt) : "Waiting to run"}</small></div>
        </div>
        <div className="simple-action-details">
          <div>
            <h3>Other options considered</h3>
            {actionScores.length ? actionScores.map((candidate) => {
              const selected = candidate.action === payment.recoveryAction;
              const width = Math.max(5, candidate.score / maxActionScore * 100);
              return <div className={"simple-option " + (selected ? "selected" : "")} key={candidate.action}><span>{readable(candidate.action)}</span><i><em style={{ width: width + "%" } as CSSProperties} /></i><b>{share(candidate.score, 0)}</b></div>;
            }) : <p className="simple-empty">The bandit has no posterior for this context yet, so there is no scored comparison to show. Bars would be six equal priors dressed up as a decision.</p>}
          </div>
          <div className="simple-guardrails">
            <h3>Safety checks applied</h3>
            {guardrails.length
              ? <ul>{guardrails.map((guardrail) => <li key={guardrail}><Check />{guardrail}</li>)}</ul>
              : <p className="simple-empty">No compliance decision has been recorded for this payment yet.</p>}
          </div>
        </div>
      </section>
    </AppShell>
  );
}
