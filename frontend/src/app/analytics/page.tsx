"use client";

import { useEffect, useState } from "react";
import {
  BarChart3,
  TrendingUp,
  Activity,
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
  Legend,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PerClass {
  total: number;
  recovered: number;
  failed: number;
}

interface BatchMetrics {
  batch_size: number;
  recovery_rate: number;
  amount_recovered: number;
  per_class: Record<string, PerClass>;
}

interface BanditContext {
  strategies: Record<string, { count: number; avg_reward: number }>;
}

const COLORS = [
  "#3b82f6",
  "#22c55e",
  "#eab308",
  "#ef4444",
  "#a855f7",
  "#06b6d4",
  "#f97316",
];

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<BatchMetrics | null>(null);
  const [bandit, setBandit] = useState<Record<string, BanditContext> | null>(
    null
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/api/v1/metrics/batch`).then((r) => r.json()),
      fetch(`${API_URL}/api/v1/metrics/bandit`).then((r) => r.json()),
    ])
      .then(([batchData, banditData]) => {
        setMetrics(batchData);
        setBandit(banditData.contexts);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  // Per-class recovery data
  const classData = metrics
    ? Object.entries(metrics.per_class).map(([name, data], i) => ({
        name: name.replace(/_/g, " "),
        recovered: data.recovered,
        failed: data.failed,
        total: data.total,
        rate: data.total > 0 ? Math.round((data.recovered / data.total) * 100) : 0,
        color: COLORS[i % COLORS.length],
      }))
    : [];

  // Root cause distribution (pie)
  const rootCausePie = classData.map((d) => ({
    name: d.name,
    value: d.total,
    color: d.color,
  }));

  // Radar data for bandit learning
  const radarData = bandit
    ? Object.entries(bandit).map(([context, data]) => {
        const strategies = data.strategies || {};
        const entry: Record<string, unknown> = {
          context: context.replace(/_/g, " "),
        };
        Object.entries(strategies).forEach(([strategy, info]) => {
          entry[strategy.replace(/_/g, " ")] = Math.round(
            (info.avg_reward || 0) * 100
          );
        });
        return entry;
      })
    : [];

  // Strategy names for radar
  const allStrategies = new Set<string>();
  if (bandit) {
    Object.values(bandit).forEach((ctx) => {
      Object.keys(ctx.strategies || {}).forEach((s) =>
        allStrategies.add(s.replace(/_/g, " "))
      );
    });
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1">Analytics</h1>
        <p className="text-gray-500 text-sm">
          Deep dive into recovery performance, root cause distribution, and
          bandit learning curves
        </p>
      </div>

      {/* Top Stats */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-blue-400" />
              <span className="text-xs text-gray-500">Total Processed</span>
            </div>
            <p className="text-3xl font-bold">{metrics.batch_size}</p>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-xs text-gray-500">Recovery Rate</span>
            </div>
            <p className="text-3xl font-bold text-green-400">
              {metrics.recovery_rate}%
            </p>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span className="text-xs text-gray-500">Amount Recovered</span>
            </div>
            <p className="text-3xl font-bold text-cyan-400">
              ₹{((metrics.amount_recovered || 0) / 100).toLocaleString("en-IN")}
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Per-Class Recovery Rates */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-sm font-semibold mb-4">
            Recovery Rate by Root Cause
          </h3>
          <div className="h-72">
            {loading || classData.length === 0 ? (
              <div className="h-full shimmer rounded-lg" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={classData} margin={{ left: 10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    dataKey="name"
                    stroke="#6b7280"
                    fontSize={10}
                    angle={-25}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis stroke="#6b7280" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#111827",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar
                    dataKey="recovered"
                    name="Recovered"
                    fill="#22c55e"
                    radius={[4, 4, 0, 0]}
                    stackId="a"
                  />
                  <Bar
                    dataKey="failed"
                    name="Failed"
                    fill="#ef4444"
                    radius={[4, 4, 0, 0]}
                    stackId="a"
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Root Cause Distribution */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-sm font-semibold mb-4">
            Root Cause Distribution
          </h3>
          <div className="h-72">
            {loading || rootCausePie.length === 0 ? (
              <div className="h-full shimmer rounded-lg" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={rootCausePie}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    dataKey="value"
                    stroke="none"
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                    labelLine={{ stroke: "#4b5563" }}
                  >
                    {rootCausePie.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#111827",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Bandit Learning */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-8">
        <h3 className="text-sm font-semibold mb-2">
          Contextual Bandit — Strategy Rewards by Root Cause
        </h3>
        <p className="text-xs text-gray-500 mb-6">
          The bandit learns which recovery strategy works best for each failure type
        </p>

        {bandit && Object.keys(bandit).length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(bandit).map(([context, data]) => (
              <div
                key={context}
                className="bg-gray-950 border border-gray-800 rounded-lg p-4"
              >
                <p className="text-xs font-medium text-orange-400 mb-3">
                  {context.replace(/_/g, " ")}
                </p>
                <div className="space-y-2">
                  {Object.entries(data.strategies || {})
                    .sort(([, a], [, b]) => b.avg_reward - a.avg_reward)
                    .map(([strategy, info]) => (
                      <div
                        key={strategy}
                        className="flex items-center justify-between"
                      >
                        <span className="text-[10px] text-gray-400">
                          {strategy.replace(/_/g, " ")}
                        </span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-gray-800 rounded-full h-1.5">
                            <div
                              className="h-1.5 rounded-full bg-cyan-400"
                              style={{
                                width: `${Math.round(
                                  (info.avg_reward || 0) * 100
                                )}%`,
                              }}
                            />
                          </div>
                          <span className="text-[10px] font-mono text-gray-300 w-10 text-right">
                            {(info.avg_reward * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-600 text-sm py-8 text-center">
            No bandit learning data yet. Run the pipeline first.
          </p>
        )}
      </div>

      {/* Per-Class Details Table */}
      {classData.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-sm font-semibold mb-4">
            Detailed Breakdown by Root Cause
          </h3>
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase">
                  Root Cause
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-500 uppercase">
                  Total
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-500 uppercase">
                  Recovered
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-500 uppercase">
                  Failed
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-500 uppercase">
                  Rate
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {classData.map((d) => (
                <tr key={d.name} className="hover:bg-gray-800/20">
                  <td className="px-4 py-3 text-sm font-medium">{d.name}</td>
                  <td className="px-4 py-3 text-sm text-right">{d.total}</td>
                  <td className="px-4 py-3 text-sm text-right text-green-400">
                    {d.recovered}
                  </td>
                  <td className="px-4 py-3 text-sm text-right text-red-400">
                    {d.failed}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span
                      className={`text-sm font-medium ${
                        d.rate > 50
                          ? "text-green-400"
                          : d.rate > 20
                          ? "text-yellow-400"
                          : "text-red-400"
                      }`}
                    >
                      {d.rate}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
