"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  DollarSign,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Zap,
} from "lucide-react";

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

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/v1/dashboard/stats`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null))
      .finally(() => setLoading(false));
  }, []);

  const formatAmount = (paise: number) => {
    return `₹${(paise / 100).toLocaleString("en-IN")}`;
  };

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

      {/* Placeholder sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-400" />
            Live Recovery Feed
          </h2>
          <p className="text-gray-500 text-sm">
            Run the batch pipeline to see live recovery events here.
          </p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-blue-400" />
            Recovery Funnel
          </h2>
          <p className="text-gray-500 text-sm">
            Recovery funnel chart will render after batch data is available.
          </p>
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
