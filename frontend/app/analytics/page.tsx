"use client";

import { useCallback, useEffect, useState, type CSSProperties } from "react";
import { AppShell } from "../../components/app-shell";
import { getPayments, type PaymentSummary } from "../../lib/api";
import { demoPayments } from "../../lib/demo-data";

export default function AnalyticsPage() {
  const [payments, setPayments] = useState<PaymentSummary[]>(demoPayments);
  const [isLoading, setIsLoading] = useState(true);

  const loadPayments = useCallback(async () => {
    try {
      const live = await getPayments();
      if (live.length) setPayments(live);
    } catch { /* keep demo */ }
    finally { setIsLoading(false); }
  }, []);

  useEffect(() => { void loadPayments(); }, [loadPayments]);

  // Compute stats
  const total = payments.length;
  const recovered = payments.filter((p) => p.status === "RECOVERED").length;
  const failed = payments.filter((p) => p.status === "FAILED").length;
  const inFlight = payments.filter((p) => p.status === "IN_FLIGHT" || p.status === "QUEUED").length;
  const recoveryRate = total ? Math.round((recovered / total) * 100) : 0;

  // Root cause breakdown
  const causeCounts: Record<string, number> = {};
  payments.forEach((p) => {
    const cause = p.rootCause || p.errorCode || "UNKNOWN";
    causeCounts[cause] = (causeCounts[cause] || 0) + 1;
  });
  const sortedCauses = Object.entries(causeCounts).sort((a, b) => b[1] - a[1]);
  const maxCauseCount = Math.max(...sortedCauses.map(([, c]) => c), 1);

  // Recovery by status
  const statusBreakdown = [
    { label: "Recovered", count: recovered, color: "#29a66b" },
    { label: "In Flight", count: inFlight, color: "#d4a847" },
    { label: "Failed", count: failed, color: "#d17a73" },
  ];
  const maxStatus = Math.max(...statusBreakdown.map((s) => s.count), 1);

  return (
    <AppShell active="analytics">
      <header className="simple-page-header">
        <div>
          <p className="simple-eyebrow">ANALYTICS</p>
          <h1>Recovery analytics</h1>
          <p>Visualise recovery performance, root cause distribution, and pipeline outcomes.</p>
        </div>
      </header>

      {isLoading ? (
        <div className="simple-loading"><span className="spinner" /> Loading analytics…</div>
      ) : (
        <>
          {/* Summary stats */}
          <section className="simple-summary-grid" aria-label="Analytics summary">
            <article className="simple-summary-main">
              <span>Recovery rate</span>
              <strong>{recoveryRate}%</strong>
              <p>{recovered} of {total} payments recovered</p>
              <div className="simple-progress"><span><b>{recoveryRate}%</b> overall</span><i><em style={{ width: recoveryRate + "%" }} /></i></div>
            </article>
            <article className="simple-stat">
              <span>Total payments</span>
              <strong>{total}</strong>
              <p>All payments processed by PayRevive</p>
            </article>
            <article className="simple-stat">
              <span>Active recoveries</span>
              <strong>{inFlight}</strong>
              <p>Currently in-flight or queued</p>
            </article>
          </section>

          {/* Root cause distribution */}
          <section style={{ marginTop: 36 }}>
            <div className="simple-section-title">
              <div><p className="simple-eyebrow">ROOT CAUSE DISTRIBUTION</p><h2>Why payments are failing</h2></div>
            </div>
            <div style={{ marginTop: 18, display: "grid", gap: 8 }}>
              {sortedCauses.map(([cause, count]) => (
                <div key={cause} className="simple-option" style={{ gridTemplateColumns: "180px 1fr 50px" }}>
                  <span style={{ fontSize: 11 }}>{cause.replaceAll("_", " ")}</span>
                  <i><em style={{ width: (count / maxCauseCount * 100) + "%", background: "#29a66b" } as CSSProperties} /></i>
                  <b style={{ textAlign: "right", fontSize: 11 }}>{count}</b>
                </div>
              ))}
            </div>
          </section>

          {/* Recovery status breakdown */}
          <section style={{ marginTop: 36 }}>
            <div className="simple-section-title">
              <div><p className="simple-eyebrow">RECOVERY OUTCOMES</p><h2>Status breakdown</h2></div>
            </div>
            <div style={{ marginTop: 18, display: "grid", gap: 10 }}>
              {statusBreakdown.map((s) => (
                <div key={s.label} className="simple-option" style={{ gridTemplateColumns: "120px 1fr 50px" }}>
                  <span style={{ fontSize: 11 }}>{s.label}</span>
                  <i><em style={{ width: (s.count / maxStatus * 100) + "%", background: s.color } as CSSProperties} /></i>
                  <b style={{ textAlign: "right", fontSize: 11 }}>{s.count}</b>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </AppShell>
  );
}
