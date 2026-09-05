"use client";

import { useCallback, useEffect, useState } from "react";
import { AppShell } from "./app-shell";
import { Check, Spark } from "./icons";
import { getModelMetrics, trainModel, type Failure, type ModelMetrics } from "../lib/api";
import { share } from "../lib/format";

function percentage(value: number | undefined) {
  if (value === undefined) return "—";
  return `${(value > 1 ? value : value * 100).toFixed(1)}%`;
}

/**
 * The classifier, reported against its ceiling rather than on its own.
 *
 * An accuracy figure by itself says nothing: on this cause mix, predicting the single most
 * common cause every time already scores 40.6%, and *no* model reading only the error code and
 * the error description can exceed 68.38%, because the simulator generates distinct causes that
 * emit identical error fields. So the accuracy card carries the bound next to it, and the gap
 * between them is the part that bank health, customer history and timing actually bought.
 *
 * The page also loads the committed metrics on mount. It used to show empty dashes until you
 * pressed the retrain button, above a badge reading `production model v2.4.1` — a version
 * string that corresponded to nothing, for a model whose real provenance (which scenario it was
 * fitted on, which seeds were held out) `/model/metrics` has always been able to state.
 */
export function ModelControl() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [state, setState] = useState<"idle" | "training" | "complete" | "error">("idle");
  const [trainError, setTrainError] = useState<Failure | null>(null);

  const load = useCallback(async () => {
    const result = await getModelMetrics();
    if (result.ok) {
      setMetrics(result.data);
      setFailure(null);
    } else {
      setMetrics(null);
      setFailure(result.error);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  async function handleTrain() {
    setState("training");
    setTrainError(null);
    const result = await trainModel();
    if (result.ok) {
      setState("complete");
      await load();
    } else {
      setTrainError(result.error);
      setState("error");
    }
  }

  const ref = metrics?.reference_points;
  const matrix = metrics?.confusion_matrix;
  const provenance = metrics?.data;

  return (
    <AppShell active="settings">
      <section className="topline reveal">
        <div className="eyebrow"><span className="live-pip" /> ML OPERATIONS / CONTROL ROOM</div>
        <span className="model-version">{provenance ? <>fitted on <b>{provenance.train_scenario}</b></> : <>classifier <b>{metrics ? "loaded" : "not loaded"}</b></>}</span>
      </section>

      <header className="settings-header reveal delay-1"><p className="hero-kicker">MODEL CONTROL PANEL</p><h1>Keep the recovery<br /><i>instinct sharp.</i></h1><p>Retrain the root-cause classifier using the latest labelled payment outcomes from your backend.</p></header>

      <section className="train-console reveal delay-2">
        <div className="train-copy"><div className="panel-eyebrow"><b>01</b><span>RETRAIN XGBOOST CLASSIFIER</span></div><h2>Refresh the decision model.</h2><p>{provenance ? "Fitted on the simulator's true cause, " + provenance.split + ". Labels: " + provenance.labels + "." : "Starts a controlled backend training job. New weights should only be promoted after your offline evaluation passes."}</p></div>
        <button className="train-button" onClick={handleTrain} disabled={state === "training"}><Spark />{state === "training" ? "Training model…" : "Start model retraining"}</button>
      </section>

      {state === "error" && <div className="batch-notice error">{trainError?.message ?? "The model training endpoint could not be reached."}{trainError?.fix ? " Run: " + trainError.fix : ""}</div>}
      {state === "training" && <div className="training-rail"><span /><span /><span /><span /><b>Extracting outcomes · fitting estimator · evaluating holdout set</b></div>}
      {failure && state !== "training" && <div className="batch-notice error">{failure.message}{failure.consequence ? " " + failure.consequence : ""}{failure.fix ? " Run: " + failure.fix : ""}</div>}

      <section className="evaluation-area reveal delay-3">
        <div className="evaluation-heading">
          <div>
            <p className="eyebrow">EVALUATION OUTPUT</p>
            <h2>{state === "complete" ? "Training run complete." : metrics ? "Committed metrics." : "Awaiting next training run."}</h2>
          </div>
          {state === "complete" && <span className="training-complete"><Check /> job complete</span>}
        </div>
        <div className="results-grid">
          <article className="result-stat">
            <span>Accuracy</span>
            <strong>{percentage(metrics?.accuracy)}</strong>
            <small>{ref ? "ceiling on error fields alone is " + percentage(ref.bayes_optimal_error_fields_only) + " — " + (metrics && metrics.reads_above_the_bound >= 0 ? "+" : "") + share(metrics?.reads_above_the_bound ?? 0) + " from bank, customer and timing signals" : "overall classification correctness"}</small>
          </article>
          <article className="result-stat">
            <span>Macro F1</span>
            <strong>{metrics ? metrics.macro_f1.toFixed(3) : "—"}</strong>
            <small>{metrics ? "unweighted across " + metrics.class_order.length + " causes · hardest is " + metrics.hardest_class.replaceAll("_", " ").toLowerCase() : "precision / recall balance"}</small>
          </article>
          <article className="matrix-card">
            <span>Confusion matrix</span>
            {matrix?.length
              ? <div className="matrix"><div className="matrix-axis">actual ↓ / predicted →</div>{matrix.map((row, r) => <div className="matrix-row" key={r}>{row.map((value, c) => <b key={c} style={{ opacity: Math.min(1, 0.28 + value / Math.max(...matrix.flat(), 1) * 0.72) }}>{value}</b>)}</div>)}</div>
              : <p>Returned by the training endpoint.</p>}
          </article>
        </div>
        <article className="json-panel">
          <div><span>RAW RESPONSE</span><small>{metrics ? "live /api/v1/model/metrics payload" : "JSON will appear once the classifier has been fitted"}</small></div>
          <pre>{metrics ? JSON.stringify(metrics, null, 2) : "{\n  // no metrics on disk yet — run the training job\n}"}</pre>
        </article>
      </section>
    </AppShell>
  );
}
