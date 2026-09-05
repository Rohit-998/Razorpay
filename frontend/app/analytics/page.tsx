"use client";

import { useCallback, useEffect, useState, type CSSProperties } from "react";
import { AppShell } from "../../components/app-shell";
import { getLadder, type Failure, type Ladder } from "../../lib/api";
import { count, interval, rupees, share } from "../../lib/format";

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
 */
export default function AnalyticsPage() {
  const [ladder, setLadder] = useState<Ladder | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const load = useCallback(async () => {
    const result = await getLadder();
    if (result.ok) {
      setLadder(result.data);
      setFailure(null);
    } else {
      setLadder(null);
      setFailure(result.error);
    }
    setIsLoading(false);
  }, []);

  useEffect(() => { void load(); }, [load]);

  const proposal = ladder?.policies.find((p) => p.is_proposal);
  const net = ladder?.net_lift;
  const beating = ladder?.seeds_beating_baseline;
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
              <span>Seeds where it won</span>
              <strong>{beating ? beating.count + " / " + beating.of : "—"}</strong>
              <p>Beat the {ladder.design.baseline_policy} baseline on the same batch</p>
            </article>
          </section>

          {/* The policy ladder */}
          <section style={{ marginTop: 36 }}>
            <div className="simple-section-title">
              <div><p className="simple-eyebrow">THE LADDER</p><h2>What each policy recovers</h2></div>
              <span className="simple-data-badge">{ladder.scenario}</span>
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
            </div>
            <div style={{ marginTop: 18, display: "grid", gap: 10 }}>
              {ladder.regret_vs_ceiling && (
                <div className="simple-option" style={{ gridTemplateColumns: "220px 1fr 110px" }}>
                  <span style={{ fontSize: 11 }}>Regret vs {ladder.design.ceiling_policy.replaceAll("_", " ")}</span>
                  <i><em style={{ width: Math.min(100, (1 - shareOfAchievable) * 100) + "%", background: "#d17a73" } as CSSProperties} /></i>
                  <b style={{ textAlign: "right", fontSize: 11 }}>{rupees(ladder.regret_vs_ceiling.mean)}</b>
                </div>
              )}
              {Object.entries(ladder.hard_limits ?? {}).map(([key, value]) => (
                <div key={key} className="simple-option" style={{ gridTemplateColumns: "220px 1fr 110px" }}>
                  <span style={{ fontSize: 11 }}>{key.replaceAll("_", " ")}</span>
                  <i><em style={{ width: "0%" } as CSSProperties} /></i>
                  <b style={{ textAlign: "right", fontSize: 11 }}>{count(value)}</b>
                </div>
              ))}
            </div>
            {ladder.self_inflicted_block_rate && (
              <p className="simple-section-copy" style={{ marginTop: 12 }}>
                {count(ladder.self_inflicted_block_rate.count)} of{" "}
                {count(ladder.self_inflicted_block_rate.of)} actions were refused by the
                compliance engine and re-routed rather than dropped
                ({share(ladder.self_inflicted_block_rate.rate)}). Report generated{" "}
                {ladder.generated_at.slice(0, 16).replace("T", " ")}.
              </p>
            )}
          </section>
        </>
      )}
    </AppShell>
  );
}
