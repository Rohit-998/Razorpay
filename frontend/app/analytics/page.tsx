"use client";

import { useCallback, useEffect, useState, type CSSProperties } from "react";
import { AppShell } from "../../components/app-shell";
import {
  getLadder,
  getShippability,
  type Failure,
  type Ladder,
  type Shippability,
} from "../../lib/api";
import { count, interval, outOf, rupees, share } from "../../lib/format";

/**
 * The measurement page.
 *
 * This is the number the brief asks for — "measured money recovered across a batch" — and it
 * is the one page here that cannot be sourced from live traffic, because a lift needs a
 * counterfactual and production has no twin batch running under `do_nothing`. So it reads
 * `/eval/ladder`, which the harness computes offline over paired seeds and bootstraps for an
 * interval.
 *
 * What this page used to show was `recovered / total` over whatever 50 payments the feed had
 * fetched, labelled "Recovery rate", above a root-cause histogram of the same 50 rows. That
 * number counts every customer who would have paid anyway as a win, which on these batches is
 * about a third of them — a system that did nothing would have posted a respectable one.
 *
 * When the harness has not been run there is no report to read and the page says so, with the
 * command that produces it. It does not draw an empty chart, and it does not invent one.
 *
 * It reads two endpoints because the two aggregate differently and say so. `/eval/ladder`
 * pools rupee lift into a bootstrap interval and withholds action counts, since summing
 * actions across scenarios of different sizes gives a number that means nothing.
 * `/eval/shippability` sums the defect counts on purpose — a quiet-hour message on outage day
 * is not cancelled out by a clean run on baseline. Reading the counts off the ladder is what
 * this page used to do, and they were never there: the bottom section rendered its heading
 * over nothing at all.
 */
export default function AnalyticsPage() {
  const [ladder, setLadder] = useState<Ladder | null>(null);
  const [gates, setGates] = useState<Shippability | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const load = useCallback(async () => {
    const [result, shipResult] = await Promise.all([getLadder(), getShippability()]);
    if (result.ok) {
      setLadder(result.data);
      setFailure(null);
    } else {
      setLadder(null);
      setFailure(result.error);
    }
    // The gates are a second read against the same report, so one arriving without the other
    // means a partial render, not a broken page. The ladder owns the error notice.
    setGates(shipResult.ok ? shipResult.data : null);
    setIsLoading(false);
  }, []);

  useEffect(() => { void load(); }, [load]);

  const proposal = ladder?.policies.find((p) => p.is_proposal);
  // Off the proposal's own row. These used to be read from the top level of the response,
  // where nothing has ever written them, so the headline number on the measurement page was
  // a dash on every load.
  const net = proposal?.net_lift;
  const beating = proposal?.seeds_beating_baseline;
  const regret = proposal?.regret_vs_ceiling;
  const proposalGates = gates?.policies.find((p) => p.is_proposal);
  const ranked = [...(ladder?.policies ?? [])].sort((a, b) => b.lift.mean - a.lift.mean);
  const maxLift = Math.max(...ranked.map((p) => p.lift.mean), 1);
  const shareOfAchievable = proposal?.share_of_achievable ?? 0;

  return (
    <AppShell active="analytics">
      <header className="simple-page-header">
        <div>
          <p className="simple-eyebrow">ANALYTICS</p>
          <h1>Recovery analytics</h1>
          <p>Measured against a do-nothing baseline on the same customers and the same coin flips.</p>
        </div>
      </header>

      {isLoading ? (
        <div className="simple-loading"><span className="spinner" /> Loading analytics…</div>
      ) : failure || !ladder ? (
        <div className="simple-notice error" role="status">
          {failure?.message ?? "No evaluation report available."}
          {failure?.fix ? " Run: " + failure.fix : ""}
        </div>
      ) : (
        <>
          {/* Summary stats */}
          <section className="simple-summary-grid" aria-label="Analytics summary">
            <article className="simple-summary-main">
              <span>Net lift per batch vs doing nothing</span>
              <strong>{net ? rupees(net.mean) : "—"}</strong>
              <p>{net ? "95% interval " + interval(net.low, net.high) + " over " + count(net.seeds) + " paired seeds" + (net.excludes_zero ? " — excludes zero" : " — includes zero, so not yet a claim") : "Recovered minus what it cost to recover"}</p>
              <div className="simple-progress">
                <span><b>{share(shareOfAchievable, 0)}</b> of what a perfect-information policy could get</span>
                <i><em style={{ width: Math.min(100, shareOfAchievable * 100) + "%" }} /></i>
              </div>
            </article>
            <article className="simple-stat">
              <span>Batches measured</span>
              <strong>{count(ladder.design.batches_run)}</strong>
              <p>{ladder.design.seeds.length} seeds × {ladder.design.scenarios.length} scenario{ladder.design.scenarios.length === 1 ? "" : "s"}, {ladder.design.pairing}</p>
            </article>
            <article className="simple-stat">
              <span>Batches where it won</span>
              <strong>{beating ? beating.count + " / " + beating.of : "—"}</strong>
              <p>Beat the {ladder.design.baseline_policy} baseline on the same batch{beating && beating.count === beating.of ? " — every one of them" : ""}</p>
            </article>
          </section>

          {/* The policy ladder */}
          <section style={{ marginTop: 36 }}>
            <div className="simple-section-title">
              <div><p className="simple-eyebrow">THE LADDER</p><h2>What each policy recovers</h2></div>
              {/* Pooled reads carry no scenario name, and an empty badge looked like a
                  loading state. Naming the count is also the more honest label: the headline
                  is a mix of weeks including the bad ones, not one flattering scenario. */}
              <span className="simple-data-badge">{ladder.scenario ?? `all ${ladder.design.scenarios.length} scenarios`}</span>
            </div>
            <div style={{ marginTop: 18, display: "grid", gap: 8 }}>
              {ranked.map((policy) => (
                <div key={policy.policy} className={"simple-option " + (policy.is_proposal ? "selected" : "")} style={{ gridTemplateColumns: "180px 1fr 110px" }}>
                  <span style={{ fontSize: 11 }}>{policy.policy.replaceAll("_", " ")}</span>
                  <i><em style={{ width: Math.max(2, policy.lift.mean / maxLift * 100) + "%", background: policy.is_ceiling ? "#d4a847" : "#29a66b" } as CSSProperties} /></i>
                  <b style={{ textAlign: "right", fontSize: 11 }}>{rupees(policy.lift.mean)}</b>
                </div>
              ))}
            </div>
            <p className="simple-section-copy" style={{ marginTop: 12 }}>
              Lift over {ladder.design.baseline_policy}. The ceiling is {ladder.design.ceiling_policy} — a policy that
              already knows the true cause and the true outcome, so it is not a target, it is the
              upper bound the gap is measured against.
            </p>
          </section>

          {/* Where the money is not going */}
          <section style={{ marginTop: 36 }}>
            <div className="simple-section-title">
              <div><p className="simple-eyebrow">WHAT IT COSTS TO GET IT</p><h2>Gap to the ceiling, and the refusals along the way</h2></div>
              {proposalGates && <span className="simple-data-badge">{proposalGates.verdict}</span>}
            </div>
            <div style={{ marginTop: 18, display: "grid", gap: 10 }}>
              {regret && (
                <div className="simple-option" style={{ gridTemplateColumns: "220px 1fr 110px" }}>
                  <span style={{ fontSize: 11 }}>Regret vs {ladder.design.ceiling_policy.replaceAll("_", " ")}</span>
                  <i><em style={{ width: Math.min(100, (1 - shareOfAchievable) * 100) + "%", background: "#d17a73" } as CSSProperties} /></i>
                  <b style={{ textAlign: "right", fontSize: 11 }}>{rupees(regret.mean)}</b>
                </div>
              )}
              {/* The four gates, from the endpoint that sums them. `hard_limits` used to be
                  read here as though it were a map of counts; it is a list of key names, so
                  every row rendered its own label as its value — on the pooled response it
                  rendered nothing at all. A gate is pass/fail against zero, so the bar is
                  full when it passes and the count carries the denominator it was judged
                  against, because a count with no denominator cannot be checked. */}
              {(proposalGates?.gates ?? []).map((gate) => (
                <div key={gate.key} className="simple-option" style={{ gridTemplateColumns: "220px 1fr 110px" }}>
                  <span style={{ fontSize: 11 }}>{gate.label}</span>
                  <i><em style={{ width: gate.passed ? "100%" : "0%", background: gate.passed ? "#29a66b" : "#d17a73" } as CSSProperties} /></i>
                  <b style={{ textAlign: "right", fontSize: 11 }}>{gate.of ? outOf(gate.count, gate.of) : count(gate.count)}</b>
                </div>
              ))}
              {proposalGates && (
                <div className="simple-option" style={{ gridTemplateColumns: "220px 1fr 110px" }}>
                  <span style={{ fontSize: 11 }}>{proposalGates.harm.label}</span>
                  <i><em style={{ width: Math.min(100, proposalGates.harm.rate * 100) + "%", background: "#d4a847" } as CSSProperties} /></i>
                  <b style={{ textAlign: "right", fontSize: 11 }}>{outOf(proposalGates.harm.count, proposalGates.harm.of)}</b>
                </div>
              )}
            </div>
            <p className="simple-section-copy" style={{ marginTop: 12 }}>
              {proposalGates
                ? `Four gates, each of them attainable at zero, so any count above zero is a defect no amount of lift buys back — ${ladder.design.ceiling_policy} fails two of them. The last row is a cost rather than a gate: a failed retry can kill a working instrument, so the only policy that breaks none is one that retries nothing (${share(proposalGates.harm.rate)} here). `
                : "Gate counts are unavailable — the report was read but its shippability table could not be. "}
              Report generated {ladder.generated_at.slice(0, 16).replace("T", " ")}.
            </p>
          </section>
        </>
      )}
    </AppShell>
  );
}
