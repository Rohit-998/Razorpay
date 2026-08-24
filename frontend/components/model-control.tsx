"use client";

import { useState } from "react";
import { AppShell } from "./app-shell";
import { Check, Spark } from "./icons";
import { trainModel, type TrainingResponse } from "../lib/api";

function percentage(value: number | undefined) {
  if (value === undefined) return "—";
  return `${(value > 1 ? value : value * 100).toFixed(1)}%`;
}

export function ModelControl() {
  const [state, setState] = useState<"idle" | "training" | "complete" | "error">("idle");
  const [response, setResponse] = useState<TrainingResponse | null>(null);

  async function handleTrain() {
    setState("training");
    setResponse(null);
    try {
      setResponse(await trainModel());
      setState("complete");
    } catch {
      setState("error");
    }
  }

  const accuracy = response?.accuracy;
  const f1 = response?.f1_score ?? response?.f1Score;
  const matrix = response?.confusion_matrix ?? response?.confusionMatrix;

  return (
    <AppShell active="settings">
      <section className="topline reveal"><div className="eyebrow"><span className="live-pip" /> ML OPERATIONS / CONTROL ROOM</div><span className="model-version">production model <b>v2.4.1</b></span></section>

      <header className="settings-header reveal delay-1"><p className="hero-kicker">MODEL CONTROL PANEL</p><h1>Keep the recovery<br /><i>instinct sharp.</i></h1><p>Retrain the root-cause classifier using the latest labelled payment outcomes from your backend.</p></header>

      <section className="train-console reveal delay-2">
        <div className="train-copy"><div className="panel-eyebrow"><b>01</b><span>RETRAIN XGBOOST CLASSIFIER</span></div><h2>Refresh the decision model.</h2><p>Starts a controlled backend training job. New weights should only be promoted after your offline evaluation passes.</p></div>
        <button className="train-button" onClick={handleTrain} disabled={state === "training"}><Spark />{state === "training" ? "Training model…" : "Start model retraining"}</button>
      </section>

      {state === "error" && <div className="batch-notice error">The model training endpoint could not be reached. Confirm that FastAPI is running and your API base URL is configured.</div>}
      {state === "training" && <div className="training-rail"><span /><span /><span /><span /><b>Extracting outcomes · fitting estimator · evaluating holdout set</b></div>}

      <section className="evaluation-area reveal delay-3">
        <div className="evaluation-heading"><div><p className="eyebrow">EVALUATION OUTPUT</p><h2>{response ? "Training run complete." : "Awaiting next training run."}</h2></div>{state === "complete" && <span className="training-complete"><Check /> job complete</span>}</div>
        <div className="results-grid">
          <article className="result-stat"><span>Accuracy</span><strong>{percentage(accuracy)}</strong><small>overall classification correctness</small></article>
          <article className="result-stat"><span>F1 score</span><strong>{percentage(f1)}</strong><small>precision / recall balance</small></article>
          <article className="matrix-card"><span>Confusion matrix</span>{matrix?.length ? <div className="matrix"><div className="matrix-axis">actual ↓ / predicted →</div>{matrix.map((row, r) => <div className="matrix-row" key={r}>{row.map((value, c) => <b key={c} style={{ opacity: Math.min(1, 0.28 + value / Math.max(...matrix.flat()) * 0.72) }}>{value}</b>)}</div>)}</div> : <p>Returned by the training endpoint.</p>}</article>
        </div>
        <article className="json-panel"><div><span>RAW RESPONSE</span><small>{response ? "live /api/v1/model/train payload" : "JSON will appear after a completed training job"}</small></div><pre>{response ? JSON.stringify(response, null, 2) : "{\n  // endpoint response pending\n}"}</pre></article>
      </section>
    </AppShell>
  );
}
