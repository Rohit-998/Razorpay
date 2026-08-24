"use client";

import { useEffect, useState } from "react";
import {
  Settings,
  Brain,
  RefreshCcw,
  CheckCircle2,
  Loader2,
  Server,
  Database,
  Cpu,
  Shield,
  AlertCircle,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface HealthData {
  status: string;
  model_loaded: boolean;
  environment: string;
}

interface TrainMetrics {
  accuracy: number;
  macro_f1: number;
  per_class: Record<
    string,
    { precision: number; recall: number; f1: number; support: number }
  >;
  confusion_matrix: number[][];
  train_size: number;
  test_size: number;
}

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [training, setTraining] = useState(false);
  const [trainMetrics, setTrainMetrics] = useState<TrainMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/v1/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const trainModel = async () => {
    setTraining(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/model/train`, {
        method: "POST",
      });
      const data = await res.json();
      if (data.status === "error") throw new Error(data.message);
      setTrainMetrics(data.metrics);
      // Refresh health
      const hRes = await fetch(`${API_URL}/api/v1/health`);
      setHealth(await hRes.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setTraining(false);
    }
  };

  const classNames = trainMetrics
    ? Object.keys(trainMetrics.per_class)
    : [];

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1">Model Control Panel</h1>
        <p className="text-gray-500 text-sm">
          System status, model training, and performance monitoring
        </p>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <StatusCard
          icon={<Server className="w-5 h-5" />}
          label="API Server"
          status={health ? "online" : "offline"}
          detail={health?.environment || "—"}
        />
        <StatusCard
          icon={<Brain className="w-5 h-5" />}
          label="XGBoost Model"
          status={health?.model_loaded ? "loaded" : "not loaded"}
          detail={health?.model_loaded ? "Ready for predictions" : "Train the model first"}
        />
        <StatusCard
          icon={<Database className="w-5 h-5" />}
          label="Supabase"
          status="connected"
          detail="PostgreSQL + Realtime"
        />
      </div>

      {/* Train Model */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Cpu className="w-4 h-4 text-blue-400" />
              Model Training
            </h3>
            <p className="text-xs text-gray-500 mt-1">
              Retrain the XGBoost classifier on current labeled data from
              Supabase
            </p>
          </div>
          <button
            onClick={trainModel}
            disabled={training}
            className={`px-6 py-2.5 rounded-lg text-sm font-medium flex items-center gap-2 transition-all duration-200 ${
              training
                ? "bg-gray-800 text-gray-500 cursor-not-allowed"
                : "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white"
            }`}
          >
            {training ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <RefreshCcw className="w-4 h-4" />
                Retrain Model
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 mb-4">
            <AlertCircle className="w-4 h-4 text-red-400" />
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {trainMetrics && (
          <div className="space-y-6">
            {/* Accuracy & F1 */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricBox
                label="Accuracy"
                value={`${Math.round(trainMetrics.accuracy * 100)}%`}
                color="text-blue-400"
              />
              <MetricBox
                label="Macro F1"
                value={trainMetrics.macro_f1.toFixed(4)}
                color="text-cyan-400"
              />
              <MetricBox
                label="Train Size"
                value={trainMetrics.train_size.toString()}
                color="text-purple-400"
              />
              <MetricBox
                label="Test Size"
                value={trainMetrics.test_size.toString()}
                color="text-orange-400"
              />
            </div>

            {/* Per-class metrics */}
            <div>
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-3">
                Per-Class Performance
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-800">
                      <th className="text-left px-4 py-2 text-xs text-gray-500">
                        Class
                      </th>
                      <th className="text-right px-4 py-2 text-xs text-gray-500">
                        Precision
                      </th>
                      <th className="text-right px-4 py-2 text-xs text-gray-500">
                        Recall
                      </th>
                      <th className="text-right px-4 py-2 text-xs text-gray-500">
                        F1 Score
                      </th>
                      <th className="text-right px-4 py-2 text-xs text-gray-500">
                        Support
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800/50">
                    {Object.entries(trainMetrics.per_class).map(
                      ([cls, metrics]) => (
                        <tr
                          key={cls}
                          className="hover:bg-gray-800/20 transition-colors"
                        >
                          <td className="px-4 py-2.5 text-xs font-medium">
                            {cls.replace(/_/g, " ")}
                          </td>
                          <td className="px-4 py-2.5 text-xs text-right font-mono">
                            {metrics.precision.toFixed(2)}
                          </td>
                          <td className="px-4 py-2.5 text-xs text-right font-mono">
                            {metrics.recall.toFixed(2)}
                          </td>
                          <td className="px-4 py-2.5 text-xs text-right font-mono">
                            <span
                              className={
                                metrics.f1 > 0.8
                                  ? "text-green-400"
                                  : metrics.f1 > 0.5
                                  ? "text-yellow-400"
                                  : "text-red-400"
                              }
                            >
                              {metrics.f1.toFixed(2)}
                            </span>
                          </td>
                          <td className="px-4 py-2.5 text-xs text-right text-gray-500">
                            {metrics.support}
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Confusion Matrix */}
            {trainMetrics.confusion_matrix && (
              <div>
                <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-3">
                  Confusion Matrix
                </h4>
                <div className="overflow-x-auto">
                  <table className="text-xs">
                    <thead>
                      <tr>
                        <th className="px-2 py-1 text-gray-600">Actual ↓ / Pred →</th>
                        {classNames.map((c) => (
                          <th
                            key={c}
                            className="px-2 py-1 text-gray-500 text-center"
                          >
                            {c.slice(0, 6)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {trainMetrics.confusion_matrix.map((row, i) => (
                        <tr key={i}>
                          <td className="px-2 py-1 text-gray-500 font-medium">
                            {classNames[i]?.slice(0, 10)}
                          </td>
                          {row.map((val, j) => (
                            <td
                              key={j}
                              className={`px-2 py-1 text-center font-mono ${
                                i === j && val > 0
                                  ? "text-green-400 bg-green-500/10"
                                  : val > 0
                                  ? "text-red-400 bg-red-500/5"
                                  : "text-gray-700"
                              }`}
                            >
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Tech Stack */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <Shield className="w-4 h-4 text-gray-400" />
          Technology Stack
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <TechItem name="XGBoost" desc="ML Classifier" />
          <TechItem name="SHAP" desc="Explainability" />
          <TechItem name="Contextual Bandit" desc="RL Agent" />
          <TechItem name="FastAPI" desc="Backend API" />
          <TechItem name="Supabase" desc="Database" />
          <TechItem name="Redis" desc="Feature Store" />
          <TechItem name="Next.js" desc="Frontend" />
          <TechItem name="Azure" desc="Cloud Deploy" />
        </div>
      </div>
    </div>
  );
}

function StatusCard({
  icon,
  label,
  status,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  status: string;
  detail: string;
}) {
  const isOnline =
    status === "online" || status === "loaded" || status === "connected";
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400">{icon}</span>
        <span
          className={`flex items-center gap-1.5 text-xs font-medium ${
            isOnline ? "text-green-400" : "text-red-400"
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isOnline ? "bg-green-400 animate-pulse-dot" : "bg-red-400"
            }`}
          />
          {status}
        </span>
      </div>
      <p className="text-sm font-medium">{label}</p>
      <p className="text-xs text-gray-500">{detail}</p>
    </div>
  );
}

function MetricBox({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-gray-950 rounded-lg p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
    </div>
  );
}

function TechItem({ name, desc }: { name: string; desc: string }) {
  return (
    <div className="bg-gray-950 border border-gray-800/50 rounded-lg p-3">
      <p className="text-xs font-medium">{name}</p>
      <p className="text-[10px] text-gray-600">{desc}</p>
    </div>
  );
}
