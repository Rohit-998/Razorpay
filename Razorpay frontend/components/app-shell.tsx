import Link from "next/link";
import { Mark, Pulse, SettingsGlyph } from "./icons";

type AppShellProps = {
  children: React.ReactNode;
  active?: "feed" | "settings";
};

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
          <Link href="/settings" className={active === "settings" ? "nav-item active" : "nav-item"}>
            <SettingsGlyph /> <span>Model controls</span>
          </Link>
        </nav>
        <div className="rail-bottom">
          <span className="status-dot" />
          <span>System status<strong>Operational</strong></span>
        </div>
      </aside>
      <main className="workspace">{children}</main>
    </div>
  );
}
