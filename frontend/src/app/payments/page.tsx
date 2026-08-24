"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Search,
  ChevronRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Payment {
  payment_id: string;
  amount: number;
  method: string;
  bank: string | null;
  error_code: string;
  error_reason: string;
  error_source: string;
  created_at: string;
}

interface Session {
  payment_id: string;
  status: string;
  root_cause: string | null;
  strategy: string | null;
  root_cause_confidence: number | null;
}

export default function PaymentsPage() {
  const [payments, setPayments] = useState<Payment[]>([]);
  const [sessions, setSessions] = useState<Record<string, Session>>({});
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");
  const [search, setSearch] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/api/v1/payments?limit=150`).then((r) => r.json()),
      fetch(`${API_URL}/api/v1/metrics/batch`).then((r) => r.json()),
    ])
      .then(([payData]) => {
        setPayments(payData.payments || []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));

    // Fetch session data for all payments
    fetch(`${API_URL}/api/v1/dashboard/stats`)
      .then((r) => r.json())
      .catch(() => {});
  }, []);

  // Fetch session info for displayed payments
  useEffect(() => {
    if (payments.length === 0) return;
    const fetchSessions = async () => {
      const sessionMap: Record<string, Session> = {};
      // Fetch in parallel, max 20 at a time
      const batch = payments.slice(0, 50);
      await Promise.all(
        batch.map(async (p) => {
          try {
            const res = await fetch(
              `${API_URL}/api/v1/payments/${p.payment_id}`
            );
            const data = await res.json();
            if (data.session) {
              sessionMap[p.payment_id] = data.session;
            }
          } catch {}
        })
      );
      setSessions(sessionMap);
    };
    fetchSessions();
  }, [payments]);

  const formatAmount = (paise: number) =>
    `₹${(paise / 100).toLocaleString("en-IN")}`;

  const getStatusIcon = (status: string | undefined) => {
    switch (status) {
      case "RECOVERED":
        return <CheckCircle2 className="w-4 h-4 text-green-400" />;
      case "FAILED":
        return <XCircle className="w-4 h-4 text-red-400" />;
      case "ESCALATED":
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string | undefined) => {
    switch (status) {
      case "RECOVERED":
        return "text-green-400 bg-green-500/10";
      case "FAILED":
        return "text-red-400 bg-red-500/10";
      case "ESCALATED":
        return "text-yellow-400 bg-yellow-500/10";
      default:
        return "text-gray-400 bg-gray-500/10";
    }
  };

  const filtered = payments.filter((p) => {
    const session = sessions[p.payment_id];
    if (filter !== "all" && session?.status !== filter) return false;
    if (
      search &&
      !p.payment_id.toLowerCase().includes(search.toLowerCase()) &&
      !p.bank?.toLowerCase().includes(search.toLowerCase())
    )
      return false;
    return true;
  });

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1">Failed Payments</h1>
        <p className="text-gray-500 text-sm">
          All intercepted payment failures with AI analysis results
        </p>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 mb-6">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input
            type="text"
            placeholder="Search by payment ID or bank..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-gray-900 border border-gray-800 rounded-lg text-sm focus:outline-none focus:border-blue-500/50 placeholder-gray-600"
          />
        </div>
        <div className="flex gap-2">
          {["all", "RECOVERED", "FAILED", "ESCALATED", "OPEN"].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-2 text-xs font-medium rounded-lg border transition-colors ${
                filter === f
                  ? "bg-blue-500/15 border-blue-500/30 text-blue-400"
                  : "border-gray-800 text-gray-500 hover:text-gray-300 hover:border-gray-700"
              }`}
            >
              {f === "all" ? "All" : f.charAt(0) + f.slice(1).toLowerCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left px-6 py-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Payment ID
                </th>
                <th className="text-left px-6 py-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Amount
                </th>
                <th className="text-left px-6 py-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Method / Bank
                </th>
                <th className="text-left px-6 py-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Error
                </th>
                <th className="text-left px-6 py-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  AI Root Cause
                </th>
                <th className="text-left px-6 py-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Strategy
                </th>
                <th className="text-left px-6 py-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-4" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {loading
                ? Array.from({ length: 8 }).map((_, i) => (
                    <tr key={i}>
                      <td colSpan={8} className="px-6 py-4">
                        <div className="h-8 shimmer rounded" />
                      </td>
                    </tr>
                  ))
                : filtered.slice(0, 50).map((p) => {
                    const session = sessions[p.payment_id];
                    return (
                      <tr
                        key={p.payment_id}
                        className="hover:bg-gray-800/30 transition-colors"
                      >
                        <td className="px-6 py-4">
                          <Link
                            href={`/payments/${p.payment_id}`}
                            className="text-sm font-mono text-blue-400 hover:text-blue-300"
                          >
                            {p.payment_id.slice(0, 18)}...
                          </Link>
                        </td>
                        <td className="px-6 py-4 text-sm font-medium">
                          {formatAmount(p.amount)}
                        </td>
                        <td className="px-6 py-4">
                          <span className="text-sm uppercase">
                            {p.method}
                          </span>
                          <span className="text-xs text-gray-500 block">
                            {p.bank || "—"}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span className="text-xs text-red-400">
                            {p.error_reason?.replace(/_/g, " ") || "—"}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span className="text-xs font-medium text-orange-400">
                            {session?.root_cause?.replace(/_/g, " ") || "—"}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span className="text-xs text-cyan-400">
                            {session?.strategy?.replace(/_/g, " ") || "—"}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span
                            className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(
                              session?.status
                            )}`}
                          >
                            {getStatusIcon(session?.status)}
                            {session?.status || "OPEN"}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <Link href={`/payments/${p.payment_id}`}>
                            <ChevronRight className="w-4 h-4 text-gray-600 hover:text-gray-300" />
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
            </tbody>
          </table>
        </div>
        {!loading && filtered.length === 0 && (
          <div className="py-12 text-center text-gray-500 text-sm">
            No payments match your filters.
          </div>
        )}
      </div>
    </div>
  );
}
