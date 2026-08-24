"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AppShell } from "../../components/app-shell";
import { ArrowUpRight } from "../../components/icons";
import { demoPayments } from "../../lib/demo-data";
import { getPayments, type PaymentSummary } from "../../lib/api";

const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

function conciseTime(timestamp: string) {
  const date = new Date(timestamp);
  return Number.isNaN(date.valueOf()) ? timestamp : new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function statusLabel(status: string) {
  return status.replaceAll("_", " ");
}

export default function PaymentsPage() {
  const [payments, setPayments] = useState<PaymentSummary[]>(demoPayments);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState("");

  const loadPayments = useCallback(async () => {
    try {
      const livePayments = await getPayments();
      if (livePayments.length) setPayments(livePayments);
    } catch {
      /* keep demo data */
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPayments();
  }, [loadPayments]);

  const filtered = filter
    ? payments.filter((p) => p.status.toLowerCase().includes(filter.toLowerCase()) || p.rootCause?.toLowerCase().includes(filter.toLowerCase()) || p.bank.toLowerCase().includes(filter.toLowerCase()) || p.id.toLowerCase().includes(filter.toLowerCase()))
    : payments;

  return (
    <AppShell active="payments">
      <header className="simple-page-header">
        <div>
          <p className="simple-eyebrow">ALL PAYMENTS</p>
          <h1>Payment history</h1>
          <p>Browse all failed payments, their AI diagnosis and recovery outcomes.</p>
        </div>
      </header>

      <div style={{ margin: "0 0 20px" }}>
        <input
          type="text"
          placeholder="Filter by ID, bank, status, or root cause…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{
            width: "100%", maxWidth: 420, padding: "10px 14px", fontSize: 12,
            border: "1px solid #dce6df", borderRadius: 7, background: "#fff",
            color: "#23332b", outline: "none",
          }}
        />
      </div>

      <div className="simple-feed" role="list">
        <div className="simple-feed-head" aria-hidden="true">
          <span>Payment</span><span>AI diagnosis</span><span>Recommended action</span><span>Status</span><span />
        </div>
        {isLoading
          ? [...Array(5)].map((_, i) => <div className="simple-feed-skeleton" key={i}><i /><i /><i /><i /></div>)
          : filtered.map((payment) => (
            <Link href={"/payment/" + encodeURIComponent(payment.id)} className="simple-payment-row" role="listitem" key={payment.id}>
              <span className="simple-payment-id"><b>{payment.id}</b><small>{money.format(payment.amount)} · {payment.bank} · {conciseTime(payment.timestamp)}</small></span>
              <span className="simple-cause">{payment.rootCause ?? payment.errorCode}</span>
              <span className="simple-action">{payment.action ?? "Analysing"}</span>
              <span className={"simple-status " + payment.status.toLowerCase()}>{statusLabel(payment.status)}</span>
              <ArrowUpRight />
            </Link>
          ))
        }
        {!isLoading && filtered.length === 0 && <p style={{ padding: 24, color: "#718078", fontSize: 12 }}>No payments match your filter.</p>}
      </div>
    </AppShell>
  );
}
