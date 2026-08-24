"use client";

import { useState } from "react";
import {
  Brain,
  Play,
  CheckCircle2,
  Loader2,
  Database,
  Target,
  Shield,
  ArrowRight,
  Zap,
  AlertCircle,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface BatchResult {
  total: number;
  processed: number;
  recovered: number;
  failed: number;
  escalated: number;
}

interface TrainResult {
  accuracy: number;
  macro_f1: number;
  per_class: Record<
    string,
    { precision: number; recall: number; f1: number; support: number }
  >;
  train_size: number;
  test_size: number;
}

type PipelineStep = "idle" | "generating" | "training" | "running" | "done";

export default function PipelinePage() {
  const [step, setStep] = useState<PipelineStep>("idle");
  const [genResult, setGenResult] = useState<{
    stored: number;
    distribution: Record<string, number>;
  } | null>(null);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runFullPipeline = async () => {
    setError(null);
    try {
      // Step 1: Generate synthetic data
      setStep("generating");
      const genRes = await fetch(
        `${API_URL}/api/v1/batch/generate?count=150`,
        { method: "POST" }
      );
      const genData = await genRes.json();
      if (genData.status === "error") throw new Error(genData.message);
      setGenResult({
        stored: genData.stored,
        distribution: genData.distribution,
      });

      // Step 2: Train model
      setStep("training");
      const trainRes = await fetch(`${API_URL}/api/v1/model/train`, {
        method: "POST",
      });
      const trainData = await trainRes.json();
      if (trainData.status === "error") throw new Error(trainData.message);
      setTrainResult(trainData.metrics);

      // Step 3: Run batch
      setStep("running");
      const batchRes = await fetch(`${API_URL}/api/v1/batch/run`, {
        method: "POST",
      });
      const batchData = await batchRes.json();
      setBatchResult(batchData.results);

      setStep("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Pipeline failed");
      setStep("idle");
    }
  };

  const steps = [
    {
      key: "generating",
      label: "Generate Synthetic Data",
      desc: "Create 150 realistic failed payment records in Supabase",
      icon: Database,
      color: "text-purple-400",
      bgColor: "bg-purple-500/15",
    },
    {
      key: "training",
      label: "Train XGBoost Model",
      desc: "Train the ML classifier on labeled failure data",
      icon: Brain,
      color: "text-blue-400",
      bgColor: "bg-blue-500/15",
    },
    {
      key: "running",
      label: "Run AI Recovery Pipeline",
      desc: "Classify → SHAP Explain → Bandit Strategy → Recover",
      icon: Zap,
      color: "text-green-400",
      bgColor: "bg-green-500/15",
    },
  ];

  const getStepStatus = (stepKey: string) => {
    const order = ["generating", "training", "running"];
    const currentIdx = order.indexOf(step);
    const stepIdx = order.indexOf(stepKey);

    if (step === "done") return "done";
    if (step === "idle") return "pending";
    if (stepIdx < currentIdx) return "done";
    if (stepIdx === currentIdx) return "active";
    return "pending";
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1">AI Pipeline</h1>
        <p className="text-gray-500 text-sm">
          Run the full end-to-end recovery pipeline: Generate → Train → Classify
          → Recover
        </p>
      </div>

      {/* Pipeline Steps */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 mb-8">
        <div className="flex items-center justify-between mb-8">
          {steps.map((s, i) => {
            const status = getStepStatus(s.key);
            const Icon = s.icon;

            return (
              <div key={s.key} className="flex items-center flex-1">
                <div className="flex items-center gap-3 flex-1">
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300 ${
                      status === "done"
                        ? "bg-green-500/15"
                        : status === "active"
                        ? `${s.bgColor} animate-pulse`
                        : "bg-gray-800"
                    }`}
                  >
                    {status === "done" ? (
                      <CheckCircle2 className="w-6 h-6 text-green-400" />
                    ) : status === "active" ? (
                      <Loader2
                        className={`w-6 h-6 ${s.color} animate-spin`}
                      />
                    ) : (
                      <Icon className="w-6 h-6 text-gray-600" />
                    )}
                  </div>
                  <div>
                    <p
                      className={`text-sm font-medium ${
                        status === "pending" ? "text-gray-600" : "text-gray-200"
                      }`}
                    >
                      {s.label}
                    </p>
                    <p className="text-xs text-gray-600">{s.desc}</p>
                  </div>
                </div>
                {i < steps.length - 1 && (
                  <ArrowRight className="w-5 h-5 text-gray-700 mx-4 flex-shrink-0" />
                )}
              </div>
            );
          })}
        </div>

        {/* Action Button */}
        <div className="flex justify-center">
          <button
            onClick={runFullPipeline}
            disabled={step !== "idle" && step !== "done"}
            className={`px-8 py-3 rounded-xl font-medium text-sm flex items-center gap-2 transition-all duration-200 ${
              step === "idle" || step === "done"
                ? "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-500/20"
                : "bg-gray-800 text-gray-500 cursor-not-allowed"
            }`}
          >
            {step === "idle" ? (
              <>
                <Play className="w-4 h-4" />
                Run Full Pipeline
              </>
            ) : step === "done" ? (
              <>
                <Play className="w-4 h-4" />
                Run Again
              </>
            ) : (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing...
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-400" />
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
      </div>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Generation Results */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Database className="w-4 h-4 text-purple-400" />
            Data Generation
          </h3>
          {genResult ? (
            <div className="space-y-3">
              <div className="bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">Records Created</p>
                <p className="text-2xl font-bold text-purple-400">
                  {genResult.stored}
                </p>
              </div>
              <div className="bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-2">
                  Root Cause Distribution
                </p>
                <div className="space-y-1.5">
                  {Object.entries(genResult.distribution).map(
                    ([cause, count]) => (
                      <div
                        key={cause}
                        className="flex items-center justify-between"
                      >
                        <span className="text-[10px] text-gray-400">
                          {cause.replace(/_/g, " ")}
                        </span>
                        <span className="text-xs font-medium">{count}</span>
                      </div>
                    )
                  )}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-600 text-sm py-8 text-center">
              Waiting for pipeline...
            </p>
          )}
        </div>

        {/* Training Results */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Brain className="w-4 h-4 text-blue-400" />
            Model Training
          </h3>
          {trainResult ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-950 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">Accuracy</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {Math.round(trainResult.accuracy * 100)}%
                  </p>
                </div>
                <div className="bg-gray-950 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">F1 Score</p>
                  <p className="text-2xl font-bold text-cyan-400">
                    {trainResult.macro_f1.toFixed(2)}
                  </p>
                </div>
              </div>
              <div className="bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-2">Per-Class F1</p>
                <div className="space-y-1.5">
                  {Object.entries(trainResult.per_class)
                    .filter(([, v]) => v.support > 0)
                    .map(([cls, v]) => (
                      <div
                        key={cls}
                        className="flex items-center justify-between"
                      >
                        <span className="text-[10px] text-gray-400">
                          {cls.replace(/_/g, " ")}
                        </span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-gray-800 rounded-full h-1.5">
                            <div
                              className="h-1.5 rounded-full bg-blue-400"
                              style={{ width: `${v.f1 * 100}%` }}
                            />
                          </div>
                          <span className="text-xs font-medium w-8 text-right">
                            {v.f1.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-600 text-sm py-8 text-center">
              Waiting for pipeline...
            </p>
          )}
        </div>

        {/* Batch Results */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Target className="w-4 h-4 text-green-400" />
            Recovery Results
          </h3>
          {batchResult ? (
            <div className="space-y-3">
              <div className="bg-gray-950 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">Recovery Rate</p>
                <p className="text-3xl font-bold text-green-400">
                  {batchResult.processed > 0
                    ? Math.round(
                        (batchResult.recovered / batchResult.processed) * 100
                      )
                    : 0}
                  %
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <ResultCard
                  label="Processed"
                  value={batchResult.processed}
                  color="text-gray-300"
                />
                <ResultCard
                  label="Recovered"
                  value={batchResult.recovered}
                  color="text-green-400"
                />
                <ResultCard
                  label="Failed"
                  value={batchResult.failed}
                  color="text-red-400"
                />
                <ResultCard
                  label="Escalated"
                  value={batchResult.escalated}
                  color="text-yellow-400"
                />
              </div>
            </div>
          ) : (
            <p className="text-gray-600 text-sm py-8 text-center">
              Waiting for pipeline...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="bg-gray-950 rounded-lg p-3">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-xl font-bold ${color}`}>{value}</p>
    </div>
  );
}
