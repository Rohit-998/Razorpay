/**
 * Number formatting, in one place.
 *
 * Two reasons this is a module rather than a helper inside each page. Money arrives from the
 * API in two units — the eval endpoints report rupees, the live endpoints report paise — and a
 * page that divides by 100 in the wrong place is off by two orders of magnitude in a way that
 * looks plausible on a dashboard. Naming the unit in the function makes the mismatch a type
 * error rather than a rendering bug.
 *
 * The second reason is that this file is the only arithmetic the browser is allowed to do.
 * Every rate, share, interval and total on these pages is computed by the eval harness and
 * served; the frontend scales for display and nothing else. The dashboard this replaced
 * divided recovered payments by total payments to draw a "recovery rate" — a number the
 * report spends a page arguing against, invented client-side after the API stopped serving it.
 */

const CRORE = 10_000_000;
const LAKH = 100_000;

/** Rupees in Indian units — ₹75.06 cr, ₹17.52 L, ₹8,400. */
export function rupees(value: number, opts: { sign?: boolean } = {}): string {
  if (!Number.isFinite(value)) return "—";
  const sign = value < 0 ? "−" : opts.sign && value > 0 ? "+" : "";
  const n = Math.abs(value);
  if (n === 0) return "₹0";
  if (n >= CRORE) return `${sign}₹${(n / CRORE).toFixed(2)} cr`;
  if (n >= LAKH) return `${sign}₹${(n / LAKH).toFixed(2)} L`;
  if (n >= 1000) return `${sign}₹${Math.round(n).toLocaleString("en-IN")}`;
  return `${sign}₹${n.toFixed(0)}`;
}

/** Paise, as the live endpoints report them. Converts, then formats as rupees. */
export function paise(value: number, opts: { sign?: boolean } = {}): string {
  if (!Number.isFinite(value)) return "—";
  return rupees(value / 100, opts);
}

/** A fraction as a percentage. `share(0.6833)` → "68.3%". */
export function share(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/** A count against its denominator, which is the only way a count can be checked. */
export function outOf(count: number, of: number): string {
  if (!Number.isFinite(of) || of === 0) return `${count.toLocaleString("en-IN")}`;
  return `${count.toLocaleString("en-IN")} of ${of.toLocaleString("en-IN")}`;
}

export function count(value: number): string {
  return Number.isFinite(value) ? value.toLocaleString("en-IN") : "—";
}

/** A 95% interval, rendered as the bounds rather than as ± half a width.
 *
 * The bootstrap distribution of a rupee lift is not symmetric — the amount distribution is
 * heavy-tailed enough that one payment can be several percent of a batch — so "mean ± x"
 * would be a claim the resampling does not make. */
export function interval(low: number, high: number): string {
  return `${rupees(low)} – ${rupees(high)}`;
}

/** Timestamps, in the reader's locale, with the date spelled out.
 *
 * A report is only reproducible if you can tell which run you are looking at, and "2 hours
 * ago" cannot be compared against the timestamp printed in REPORT.md. */
export function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** `stress_dead_instruments` → `Stress dead instruments`. */
export function humanise(key: string): string {
  const s = key.replace(/_/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** `BANK_DOWNTIME` → `Bank downtime`, for prose. The raw key stays available for tables. */
export function cause(key: string): string {
  return humanise(key.toLowerCase());
}
