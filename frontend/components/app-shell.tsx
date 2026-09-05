"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Mark, Pulse, ListIcon, PipelineIcon, ChartIcon, SettingsGlyph } from "./icons";
import { getHealth } from "../lib/api";

type AppShellProps = {
  children: React.ReactNode;
  active?: "feed" | "payments" | "pipeline" | "analytics" | "settings";
};

/** The rail's status line, read from `/health` rather than asserted.
 *
 * It used to be the literal string "Operational", rendered next to a green dot on every page
 * including the ones that had just failed to reach the API. A status indicator that cannot
 * report a problem is decoration, and worse than none: it actively contradicts the error notice
 * a page is showing three inches to its right.
 *
 * `/health` checks Supabase and Redis, so "degraded" here means one of them is down — which is
 * the state where the pipeline still serves reads but stops processing. */
function SystemStatus() {
  const [label, setLabel] = useState("Checking…");
  const [tone, setTone] = useState<"ok" | "warn" | "down">("ok");

  useEffect(() => {
    let live = true;
    void getHealth().then((result) => {
      if (!live) return;
      if (!result.ok) {
        setLabel("Unreachable");
        setTone("down");
        return;
      }
      const down = Object.entries(result.data.services ?? {})
        .filter(([, up]) => !up)
        .map(([name]) => name);
      setLabel(down.length ? "Degraded — " + down.join(", ") : "Operational");
      setTone(down.length ? "warn" : "ok");
    });
    return () => {
      live = false;
    };
  }, []);

  const colour = tone === "ok" ? undefined : tone === "warn" ? "#d4a847" : "#d17a73";
  return (
    <div className="rail-bottom">
      <span className="status-dot" style={colour ? { background: colour } : undefined} />
      <span>System status<strong>{label}</strong></span>
    </div>
  );
}

export function AppShell({ children, active = "feed" }: AppShellProps) {
  return (
    <div className="app-shell">
      <aside className="rail">
        <Link className="brand" href="/" aria-label="PayRevive home">
          <Mark />
          <span>Pay<em>Revive</em><small>AI payment recovery</small></span>
        </Link>
        <nav className="rail-nav" aria-label="Main navigation">
          <Link href="/" className={active === "feed" ? "nav-item active" : "nav-item"}>
            <Pulse /> <span>Recovery feed</span>
          </Link>
          <Link href="/payments" className={active === "payments" ? "nav-item active" : "nav-item"}>
            <ListIcon /> <span>Payments</span>
          </Link>
          <Link href="/pipeline" className={active === "pipeline" ? "nav-item active" : "nav-item"}>
            <PipelineIcon /> <span>AI Pipeline</span>
          </Link>
          <Link href="/analytics" className={active === "analytics" ? "nav-item active" : "nav-item"}>
            <ChartIcon /> <span>Analytics</span>
          </Link>
          <Link href="/settings" className={active === "settings" ? "nav-item active" : "nav-item"}>
            <SettingsGlyph /> <span>Model controls</span>
          </Link>
        </nav>
        <SystemStatus />
      </aside>
      <main className="workspace">{children}</main>
    </div>
  );
}
