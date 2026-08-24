"use client";

import { useState } from "react";
import { AppShell } from "../../components/app-shell";
import { Spark, Check } from "../../components/icons";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");

type StepState = "pending" | "running" | "complete" | "error";

export default function PipelinePage() {
  const [generateState, setGenerateState] = useState<StepState>("pending");
  const [trainState, setTrainState] = useState<StepState>("pending");
  const [recoverState, setRecoverState] = useState<StepState>("pending");
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [trainResults, setTrainResults] = useState<Record<string, unknown> | null>(null);

  async function runStep(url: string, setState: (s: StepState) => void, setResult?: (r: Record<string, unknown>) => void) {
    setState("running");
    try {
      const res = await fetch(`${API_BASE_URL}${url}`, { method: "POST", headers: { "Content-Type": "application/json" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult?.(data);
      setState("complete");
      return true;
    } catch {
      setState("error");
      return false;
    }
  }

  async function handleFullPipeline() {
    setGenerateState("pending");
    setTrainState("pending");
    setRecoverState("pending");
    setResults(null);
    setTrainResults(null);

    const ok1 = await runStep("/api/v1/batch/generate", setGenerateState);
    if (!ok1) return;

    const ok2 = await runStep("/api/v1/model/train", setTrainState, setTrainResults);
    if (!ok2) return;

    await runStep("/api/v1/batch/run", setRecoverState, setResults);
  }

  function stepClass(state: StepState) {
    return "pipeline-step " + state;
  }

  return (
    <AppShell active="pipeline">
      <header className="simple-page-header">
        <div>
          <p className="simple-eyebrow">AI PIPELINE</p>
          <h1>End-to-end recovery pipeline</h1>
          <p>Run the full pipeline: generate training data, train the model, then recover payments.</p>
        </div>
        <button onClick={handleFullPipeline} className="simple-primary-button" disabled={generateState === "running" || trainState === "running" || recoverState === "running"}>
          <Spark />
          {generateState === "running" || trainState === "running" || recoverState === "running" ? "Running…" : "Run full pipeline"}
        </button>
      </header>

      <section style={{ marginTop: 30 }}>
        <ol className="simple-process">
          <li className={stepClass(generateState)}>
            <b>{generateState === "complete" ? "✓" : "1"}</b>
            <div>
              <strong>Generate training data</strong>
              <span>{generateState === "running" ? "Generating synthetic payment data…" : generateState === "complete" ? "Training data generated successfully" : generateState === "error" ? "Failed to generate data" : "Creates labelled payment data from Supabase"}</span>
            </div>
          </li>
          <li className={stepClass(trainState)}>
            <b>{trainState === "complete" ? "✓" : "2"}</b>
            <div>
              <strong>Train XGBoost classifier</strong>
              <span>{trainState === "running" ? "Training model on generated data…" : trainState === "complete" ? `Training complete — accuracy ${trainResults && typeof trainResults === "object" && "metrics" in trainResults ? ((trainResults.metrics as Record<string, unknown>).accuracy as number * 100).toFixed(0) + "%" : ""}` : trainState === "error" ? "Training failed" : "Fits the root-cause classifier and evaluates on holdout set"}</span>
            </div>
          </li>
          <li className={stepClass(recoverState)}>
            <b>{recoverState === "complete" ? "✓" : "3"}</b>
            <div>
              <strong>Run batch recovery</strong>
              <span>{recoverState === "running" ? "Processing failed payments…" : recoverState === "complete" && results ? `Done — ${(results as Record<string, unknown>).recovered ?? (results.results as Record<string, unknown>)?.recovered ?? 0} recovered` : recoverState === "error" ? "Recovery failed" : "Classifies and recovers all pending failed payments"}</span>
            </div>
          </li>
        </ol>
      </section>

      {results && (
        <section style={{ marginTop: 30 }}>
          <article className="json-panel">
            <div><span>PIPELINE RESULTS</span><small>Batch recovery output</small></div>
            <pre>{JSON.stringify(results, null, 2)}</pre>
          </article>
        </section>
      )}
    </AppShell>
  );
}
