/* Formatting helpers.
 *
 * The rule these enforce: a value the store cannot supply is rendered as a word
 * ("unknown", "no coverage"), never as a zero or a dash that reads like one.
 * The architecture is explicit that an offline camera showing "0 events" is the
 * most dangerous output this system can produce, and the same trap is one
 * `?? 0` away in any of these views.
 */

export const UNKNOWN = "unknown";

export function num(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return UNKNOWN;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function pct(fraction: number | null | undefined, digits = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return UNKNOWN;
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Compact counts for axis ticks and dense tables: 52,488,000 -> 52.5M */
export function compact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return UNKNOWN;
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `${(value / 1e3).toFixed(0)}k`;
  return value.toLocaleString();
}

export function hours(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return UNKNOWN;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(seconds < 36000 ? 1 : 0)} h`;
}

export function seconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return UNKNOWN;
  return `${value.toFixed(1)}s`;
}

/** UTC clock time. The store has no site timezone, so nothing pretends to local. */
export function clockUTC(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.slice(11, 19);
}

export function dateUTC(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.slice(0, 10);
}

export function stampUTC(iso: string | null | undefined): string {
  if (!iso) return "—";
  return `${iso.slice(0, 10)} ${iso.slice(11, 16)}Z`;
}

export function shortId(id: string): string {
  return id.slice(0, 10);
}

/** "3 nights ago", for freshness on camera rows. */
export function agoUTC(iso: string | null | undefined): string {
  if (!iso) return "never";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "—";
  const days = Math.floor((Date.now() - then) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  return `${days} days ago`;
}
