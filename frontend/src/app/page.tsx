"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowUpRight,
  AlertTriangle,
  CheckCircle2,
  Zap,
  DollarSign,
  TrendingUp,
  ExternalLink,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Stats {
  total_failed: number;
  total_recovered: number;
  total_failed_permanent: number;
  total_escalated: number;
  total_failed_amount: number;
  total_recovered_amount: number;
  recovery_rate: number;
}

interface Payment {
  payment_id: string;
  amount: number;
  method: string;
  bank: string | null;
  error_reason: string;
  created_at: string;
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [recentPayments, setRecentPayments] = useState<Payment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/v1/dashboard/stats`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null));

    fetch(`${API_URL}/api/v1/payments?limit=8`)
      .then((r) => r.json())
      .then((data) => setRecentPayments(data.payments || []))
      .catch(() => setRecentPayments([]))
      .finally(() => setLoading(false));
  }, []);

  const formatAmount = (paise: number) => {
    return `₹${(paise / 100).toLocaleString("en-IN")}`;
  };

  const funnelData = stats
    ? [
        { name: "Total Failed", value: stats.total_failed, color: "#ef4444" },
        {
          name: "AI Attempted",
          value: stats.total_failed - stats.total_escalated,
          color: "#3b82f6",
        },
        { name: "Recovered", value: stats.total_recovered, color: "#22c55e" },
      ]
    : [];

  const pieData = stats
    ? [
        {
          name: "Recovered",
          value: stats.total_recovered,
          color: "#22c55e",
        },
        {
          name: "Failed",
          value: stats.total_failed_permanent,
          color: "#ef4444",
        },
        { name: "Escalated", value: stats.total_escalated, color: "#eab308" },
      ]
    : [];

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1">Recovery Dashboard</h1>
        <p className="text-gray-500 text-sm">
          Real-time AI-powered payment failure recovery overview
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          title="Total Failed"
          value={stats?.total_failed || 0}
          icon={<AlertTriangle className="w-5 h-5" />}
          color="red"
          subtitle={stats ? formatAmount(stats.total_failed_amount) : "—"}
          loading={loading}
        />
        <StatCard
          title="AI Recovered"
          value={stats?.total_recovered || 0}
          icon={<CheckCircle2 className="w-5 h-5" />}
          color="green"
          subtitle={stats ? formatAmount(stats.total_recovered_amount) : "—"}
          loading={loading}
        />
        <StatCard
          title="Recovery Rate"
          value={stats ? `${stats.recovery_rate}%` : "—"}
          icon={<TrendingUp className="w-5 h-5" />}
          color="blue"
          subtitle="XGBoost + Bandit"
          loading={loading}
        />
        <StatCard
          title="Escalated"
          value={stats?.total_escalated || 0}
          icon={<ArrowUpRight className="w-5 h-5" />}
          color="yellow"
          subtitle="Sent to merchant"
          loading={loading}
        />
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Live Feed */}
        <div className="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse-dot" />
              Live Recovery Feed
            </h2>
            <Link
              href="/payments"
              className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
            >
              View All <ExternalLink className="w-3 h-3" />
            </Link>
          </div>

          <div className="space-y-3">
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="h-16 shimmer rounded-lg" />
              ))
            ) : recentPayments.length === 0 ? (
              <p className="text-gray-500 text-sm py-8 text-center">
                No payments found. Run the AI pipeline first.
              </p>
            ) : (
              recentPayments.slice(0, 6).map((p) => (
                <Link
                  key={p.payment_id}
                  href={`/payments/${p.payment_id}`}
                  className="flex items-center justify-between p-3 bg-gray-950 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors cursor-pointer group"
                >
                  <div>
                    <p className="font-medium text-sm group-hover:text-blue-400 transition-colors">
                      {p.payment_id}
                    </p>
                    <p className="text-xs text-gray-500 uppercase">
                      {p.method} • {p.bank || "N/A"}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-sm">
                      {formatAmount(p.amount)}
                    </p>
                    <p className="text-xs text-red-400">
                      {p.error_reason.replace(/_/g, " ")}
                    </p>
                  </div>
                </Link>
              ))
            )}
          </div>
        </div>

        {/* Outcome Pie Chart */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-blue-400" />
            AI Outcomes
          </h2>
          <div className="h-48">
            {loading || pieData.length === 0 ? (
              <div className="h-full shimmer rounded-lg" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    dataKey="value"
                    stroke="none"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#111827",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                    itemStyle={{ color: "#e5e7eb" }}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
          {/* Legend */}
          <div className="space-y-2 mt-4">
            {pieData.map((d) => (
              <div key={d.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: d.color }}
                  />
                  <span className="text-xs text-gray-400">{d.name}</span>
                </div>
                <span className="text-xs font-medium">{d.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recovery Funnel */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <DollarSign className="w-5 h-5 text-blue-400" />
          Recovery Funnel
        </h2>
        <div className="h-48 w-full">
          {loading || funnelData.length === 0 ? (
            <div className="h-full shimmer rounded-lg" />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={funnelData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#1f2937"
                  horizontal={false}
                />
                <XAxis type="number" stroke="#6b7280" fontSize={12} />
                <YAxis
                  dataKey="name"
                  type="category"
                  stroke="#6b7280"
                  width={110}
                  fontSize={12}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#111827",
                    border: "1px solid #374151",
                    borderRadius: "8px",
                  }}
                  itemStyle={{ color: "#e5e7eb" }}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {funnelData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
  color,
  subtitle,
  loading,
}: {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  color: "red" | "green" | "blue" | "yellow";
  subtitle: string;
  loading: boolean;
}) {
  const colors = {
    red: "text-red-400 bg-red-500/10 border-red-500/20",
    green: "text-green-400 bg-green-500/10 border-green-500/20",
    blue: "text-blue-400 bg-blue-500/10 border-blue-500/20",
    yellow: "text-yellow-400 bg-yellow-500/10 border-yellow-500/20",
  };

  return (
    <div className={`rounded-xl border p-5 ${colors[color]}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-400">{title}</span>
        {icon}
      </div>
      <div className="text-3xl font-bold mb-1">
        {loading ? <div className="h-9 w-20 shimmer rounded" /> : value}
      </div>
      <div className="text-xs text-gray-500">{subtitle}</div>
    </div>
  );
}
