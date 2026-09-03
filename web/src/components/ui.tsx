/* Small shared pieces: stat tiles, status chips, loading and error states. */

import type { ReactNode } from "react";
import { UNKNOWN } from "../format";

export function Stat({
  value,
  label,
  sub,
  tone,
}: {
  value: string;
  label: string;
  sub?: ReactNode;
  tone?: "good" | "warn" | "bad";
}) {
  const unknown = value === UNKNOWN;
  const color =
    tone && !unknown
      ? { good: "var(--good)", warn: "var(--accent-ink)", bad: "var(--critical)" }[tone]
      : undefined;
  return (
    <div className="card stat">
      <span className={unknown ? "n unknown" : "n"} style={color ? { color } : undefined}>
        {value}
      </span>
      <span className="l">{label}</span>
      {sub ? <span className="sub">{sub}</span> : null}
    </div>
  );
}

/* Status is never carried by colour alone: every chip pairs its colour with a
 * glyph and a word. The dark palette's CVD separation sits in the 6-8 band,
 * where secondary encoding is required rather than optional. */
const GLYPH = { good: "✓", warn: "▲", bad: "✕", plain: "·" } as const;

export function Chip({
  tone = "plain",
  children,
  title,
}: {
  tone?: "good" | "warn" | "bad" | "plain";
  children: ReactNode;
  title?: string;
}) {
  return (
    <span className={`chip ${tone}`} title={title}>
      <span aria-hidden="true">{GLYPH[tone]}</span>
      {children}
    </span>
  );
}

export function Loading({ what }: { what: string }) {
  return <p className="spinner">Loading {what}…</p>;
}

export function ErrorBox({ error }: { error: Error }) {
  return (
    <div className="err">
      <b>Could not load.</b> {error.message}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

/** Renders coverage honestly, including when it is not known. */
export function CoverageNote({
  coverage,
}: {
  coverage: { complete: boolean; observed_hours: number; runs: number; runs_missing_uptime: number };
}) {
  if (coverage.runs === 0) {
    return (
      <div className="callout bad">
        <p>
          <b>No observed coverage.</b> Nothing has run against this store, so every count
          below is zero because nothing was watched — not because nothing happened.
        </p>
      </div>
    );
  }
  if (!coverage.complete) {
    return (
      <div className="callout bad">
        <p>
          <b>Coverage is incomplete.</b> {coverage.runs_missing_uptime} of {coverage.runs} runs
          did not record how long they observed, so rates per camera-hour cannot be computed
          and are shown as unknown.
        </p>
        <p>
          Counts are still shown, but a count without a denominator cannot distinguish a quiet
          site from an unwatched one.
        </p>
      </div>
    );
  }
  return null;
}
