"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  DollarSign,
  AlertTriangle,
  CheckCircle2,
  Zap,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
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
    // Fetch stats
    fetch(`${API_URL}/api/v1/dashboard/stats`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null));

    // Fetch recent payments
    fetch(`${API_URL}/api/v1/payments?limit=5`)
      .then((r) => r.json())
      .then((data) => setRecentPayments(data.payments || []))
      .catch(() => setRecentPayments([]))
      .finally(() => setLoading(false));
  }, []);

  const formatAmount = (paise: number) => {
    return `₹${(paise / 100).toLocaleString("en-IN")}`;
  };

  const funnelData = stats ? [
    { name: "Total Failed", value: stats.total_failed, color: "#ef4444" },
    { name: "Attempted Recovery", value: stats.total_failed - stats.total_escalated, color: "#3b82f6" },
    { name: "Recovered", value: stats.total_recovered, color: "#22c55e" },
  ] : [];

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Zap className="w-8 h-8 text-blue-400" />
          <h1 className="text-2xl font-bold">PayRevive</h1>
          <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-300 rounded-full">
            AI Recovery Engine
          </span>
        </div>
        <p className="text-gray-400 text-sm">
          Track 03 — AI Revenue Recovery | Razorpay AI Buildathon
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
          title="Recovered"
          value={stats?.total_recovered || 0}
          icon={<CheckCircle2 className="w-5 h-5" />}
          color="green"
          subtitle={stats ? formatAmount(stats.total_recovered_amount) : "—"}
          loading={loading}
        />
        <StatCard
          title="Recovery Rate"
          value={stats ? `${stats.recovery_rate}%` : "—"}
          icon={<Activity className="w-5 h-5" />}
          color="blue"
          subtitle="System + bandit"
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

      {/* Main Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live Feed */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-400" />
            Live Recovery Feed
          </h2>
          
          <div className="space-y-4">
            {loading ? (
              <p className="text-gray-500 text-sm">Loading...</p>
            ) : recentPayments.length === 0 ? (
              <p className="text-gray-500 text-sm">No payments found. Run batch pipeline.</p>
            ) : (
              recentPayments.map((p) => (
                <div key={p.payment_id} className="flex items-center justify-between p-3 bg-gray-950 rounded-lg border border-gray-800">
                  <div>
                    <p className="font-medium text-sm">{p.payment_id}</p>
                    <p className="text-xs text-gray-500 uppercase">{p.method} • {p.bank || 'N/A'}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-sm">{formatAmount(p.amount)}</p>
                    <p className="text-xs text-red-400">{p.error_reason.replace(/_/g, ' ')}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Funnel Chart */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-blue-400" />
            Recovery Funnel
          </h2>
          
          <div className="h-64 w-full mt-4">
            {loading || funnelData.length === 0 ? (
              <p className="text-gray-500 text-sm flex items-center justify-center h-full">Loading data...</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={funnelData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                  <XAxis type="number" stroke="#9ca3af" />
                  <YAxis dataKey="name" type="category" stroke="#9ca3af" width={120} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
                    itemStyle={{ color: '#e5e7eb' }}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
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
        {loading ? "—" : value}
      </div>
      <div className="text-xs text-gray-500">{subtitle}</div>
    </div>
  );
}
