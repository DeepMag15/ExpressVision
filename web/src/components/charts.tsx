/* Charts, hand-rolled as inline SVG.
 *
 * No chart library: every form here is a bar, a column, a sector or a dot, and
 * the mark specs matter more than the breadth — capped bar thickness, 4px
 * rounded data-ends, 2px surface gaps between touching marks, hairline
 * gridlines, values at the tip rather than on every mark, and text in ink
 * tokens rather than the series colour.
 *
 * Everything is a single series, so no chart carries a legend: the title says
 * what is plotted and one colour cannot be mistaken for identity. The only
 * multi-colour mark is the verdict bar, where each segment is directly labelled
 * and separated by a surface gap.
 */

import { useId, useState } from "react";
import type { ReactNode } from "react";
import { compact } from "../format";

const SERIES = "var(--accent)";
const BAR_MAX = 24;
const GAP = 2;

function Frame({
  title,
  note,
  children,
  tip,
}: {
  title: string;
  note?: ReactNode;
  children: ReactNode;
  tip?: { x: number; y: number; content: ReactNode } | null;
}) {
  return (
    <div className="chart" style={{ position: "relative" }}>
      <p className="chart-title">{title}</p>
      {note ? <p className="chart-note">{note}</p> : null}
      {children}
      {tip ? (
        <div
          role="tooltip"
          style={{
            position: "absolute",
            left: `${tip.x}%`,
            top: tip.y,
            transform: "translate(-50%, -100%)",
            marginTop: -8,
            pointerEvents: "none",
            background: "var(--surface)",
            border: "1px solid var(--line-strong)",
            borderRadius: 4,
            padding: "5px 9px",
            fontSize: 11.5,
            color: "var(--ink)",
            whiteSpace: "nowrap",
            boxShadow: "var(--shadow)",
            zIndex: 5,
          }}
        >
          {tip.content}
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ bars */

export interface BarDatum {
  label: string;
  value: number;
  /** Optional trailing annotation, e.g. a percentage share. */
  note?: string;
  tone?: "series" | "critical" | "good";
}

/** Horizontal bars. The form for ranked magnitude with wordy category names.
 *
 *  `scale="log"` is for the cascade, where the stages span five orders of
 *  magnitude and a linear axis renders the last three as a sub-pixel sliver.
 *  The exact counts sit at the bar tips, so the axis distorts the picture
 *  without distorting the values — the same choice, and the same caveat, as
 *  Figure 2 of the architecture. */
export function Bars({
  title,
  note,
  data,
  format = compact,
  labelWidth = 150,
  scale = "linear",
}: {
  title: string;
  note?: ReactNode;
  data: BarDatum[];
  format?: (n: number) => string;
  labelWidth?: number;
  scale?: "linear" | "log";
}) {
  const [hover, setHover] = useState<number | null>(null);
  const id = useId();

  if (data.length === 0) return <Frame title={title} note={note}>{null}</Frame>;

  const max = Math.max(...data.map((d) => d.value), 1);
  const row = Math.min(BAR_MAX + 14, 34);
  const height = data.length * row;
  const width = 640;

  // Fixed gutters for the value and the optional note, so neither can collide
  // with the other when a bar runs the full width of the plot.
  const hasNotes = data.some((d) => d.note);
  const valueGutter = 66;
  const noteGutter = hasNotes ? 66 : 0;
  const plot = width - labelWidth - valueGutter - noteGutter;
  const bar = Math.min(BAR_MAX, row - 12);

  const extent = (value: number) => {
    if (scale === "log") {
      // Anchor the floor an order of magnitude below the smallest positive
      // value so the shortest bar is still a visible bar, not a hairline.
      const smallest = Math.min(...data.filter((d) => d.value > 0).map((d) => d.value), 1);
      const floor = Math.log10(Math.max(smallest, 1)) - 1;
      const top = Math.log10(Math.max(max, 10));
      const at = Math.log10(Math.max(value, 1));
      return Math.max(0, (at - floor) / (top - floor));
    }
    return value / max;
  };

  const tone = (d: BarDatum) =>
    d.tone === "critical" ? "var(--critical)" : d.tone === "good" ? "var(--good)" : SERIES;

  return (
    <Frame
      title={title}
      note={note}
      tip={
        hover !== null && data[hover]
          ? {
              x: 50,
              y: hover * row + 6,
              content: (
                <>
                  <b>{data[hover]!.label}</b> · {format(data[hover]!.value)}
                  {data[hover]!.note ? ` · ${data[hover]!.note}` : ""}
                </>
              ),
            }
          : null
      }
    >
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={id}>
        <title id={id}>{title}</title>
        {data.map((d, i) => {
          const w = Math.max(3, extent(d.value) * plot);
          const y = i * row + (row - bar) / 2;
          return (
            <g
              key={d.label}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              {/* Hit target spans the row, not just the bar — a 6px bar is
                  almost impossible to hover deliberately. */}
              <rect x={0} y={i * row} width={width} height={row} fill="transparent" />
              <text x={0} y={i * row + row / 2 + 4} className="bar-label">
                {d.label}
              </text>
              <rect
                x={labelWidth}
                y={y}
                width={w}
                height={bar}
                rx={4}
                fill={tone(d)}
                opacity={hover === null || hover === i ? 1 : 0.55}
              />
              {/* Square off the baseline end: rx rounds all four corners. */}
              <rect x={labelWidth} y={y} width={Math.min(4, w)} height={bar} fill={tone(d)} />
              <text x={labelWidth + w + 8} y={i * row + row / 2 + 4} className="bar-value">
                {format(d.value)}
              </text>
              {d.note ? (
                <text x={width} y={i * row + row / 2 + 4} className="tick" textAnchor="end">
                  {d.note}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </Frame>
  );
}

/* --------------------------------------------------------------- columns */

/** Vertical columns over a fixed ordered domain — hours of the day. */
export function Columns({
  title,
  note,
  values,
  tickEvery = 3,
  tickLabel = (i: number) => String(i),
  hoverLabel = (i: number, v: number) => `${tickLabel(i)}: ${v}`,
}: {
  title: string;
  note?: ReactNode;
  values: number[];
  tickEvery?: number;
  tickLabel?: (i: number) => string;
  hoverLabel?: (i: number, v: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const id = useId();

  const max = Math.max(...values, 1);
  const width = 640;
  const height = 190;
  const padBottom = 26;
  const padTop = 14;
  // Right gutter for the gridline ticks. Without it the tick label sits on top
  // of the last column, which is exactly where a nocturnal peak wraps around.
  const padRight = 34;
  const plot = height - padBottom - padTop;
  const slot = (width - padRight) / values.length;
  const bar = Math.min(BAR_MAX, slot - GAP * 2);

  // Two gridlines only — enough to read magnitude, quiet enough to recede.
  const ticks = [max / 2, max];

  return (
    <Frame
      title={title}
      note={note}
      tip={
        hover !== null && values[hover] !== undefined
          ? {
              x: ((hover + 0.5) / values.length) * 100,
              y: padTop + plot - (values[hover]! / max) * plot + 24,
              content: hoverLabel(hover, values[hover]!),
            }
          : null
      }
    >
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={id}>
        <title id={id}>{title}</title>

        {ticks.map((t) => (
          <g key={t}>
            <line
              className="gridline"
              x1={0}
              x2={width}
              y1={padTop + plot - (t / max) * plot}
              y2={padTop + plot - (t / max) * plot}
            />
            <text
              className="tick"
              x={width}
              y={padTop + plot - (t / max) * plot - 4}
              textAnchor="end"
            >
              {Math.round(t)}
            </text>
          </g>
        ))}

        {values.map((v, i) => {
          const h = v > 0 ? Math.max(2, (v / max) * plot) : 0;
          const x = i * slot + (slot - bar) / 2;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={i * slot} y={0} width={slot} height={height - padBottom} fill="transparent" />
              {h > 0 ? (
                <>
                  <rect
                    x={x}
                    y={padTop + plot - h}
                    width={bar}
                    height={h}
                    rx={4}
                    fill={SERIES}
                    opacity={hover === null || hover === i ? 1 : 0.55}
                  />
                  {/* Square the baseline end. */}
                  <rect
                    x={x}
                    y={padTop + plot - Math.min(4, h)}
                    width={bar}
                    height={Math.min(4, h)}
                    fill={SERIES}
                    opacity={hover === null || hover === i ? 1 : 0.55}
                  />
                </>
              ) : null}
              {i % tickEvery === 0 ? (
                <text className="tick" x={i * slot + slot / 2} y={height - 8} textAnchor="middle">
                  {tickLabel(i)}
                </text>
              ) : null}
            </g>
          );
        })}
        <line className="baseline" x1={0} x2={width} y1={padTop + plot} y2={padTop + plot} />
      </svg>
    </Frame>
  );
}

/* ------------------------------------------------------------------ rose */

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

/** Eight-sector rose for direction of travel. Radius is sqrt-scaled so area,
 *  not radius, carries the count — a linear radius triples the apparent weight
 *  of a sector that is merely twice as large. */
export function Rose({
  title,
  note,
  sectors,
}: {
  title: string;
  note?: ReactNode;
  sectors: number[];
}) {
  const [hover, setHover] = useState<number | null>(null);
  const id = useId();
  const size = 240;
  const c = size / 2;
  const rMax = c - 30;
  const max = Math.max(...sectors, 1);

  const arc = (i: number, r: number) => {
    // Sector i is centred on its compass point, spanning 45 degrees.
    const a0 = ((i * 45 - 22.5 - 90) * Math.PI) / 180;
    const a1 = ((i * 45 + 22.5 - 90) * Math.PI) / 180;
    const x0 = c + r * Math.cos(a0);
    const y0 = c + r * Math.sin(a0);
    const x1 = c + r * Math.cos(a1);
    const y1 = c + r * Math.sin(a1);
    return `M ${c} ${c} L ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1} Z`;
  };

  return (
    <Frame
      title={title}
      note={note}
      tip={
        hover !== null && sectors[hover] !== undefined
          ? {
              x: 50,
              y: 20,
              content: `${COMPASS[hover]} — ${sectors[hover]} tracks`,
            }
          : null
      }
    >
      <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-labelledby={id} style={{ maxWidth: 260 }}>
        <title id={id}>{title}</title>
        <circle cx={c} cy={c} r={rMax} className="gridline" fill="none" />
        <circle cx={c} cy={c} r={rMax / 2} className="gridline" fill="none" />

        {sectors.map((v, i) => {
          const r = v > 0 ? Math.max(4, Math.sqrt(v / max) * rMax) : 0;
          if (r === 0) return null;
          return (
            <path
              key={i}
              d={arc(i, r)}
              fill={SERIES}
              opacity={hover === null || hover === i ? 0.9 : 0.45}
              stroke="var(--surface)"
              strokeWidth={GAP}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          );
        })}

        {COMPASS.map((name, i) => {
          const a = ((i * 45 - 90) * Math.PI) / 180;
          return (
            <text
              key={name}
              className="tick"
              x={c + (rMax + 14) * Math.cos(a)}
              y={c + (rMax + 14) * Math.sin(a) + 3}
              textAnchor="middle"
            >
              {name}
            </text>
          );
        })}
      </svg>
    </Frame>
  );
}

/* --------------------------------------------------------------- scatter */

export interface ScatterPoint {
  x: number;
  y: number;
  tx: number | null;
  ty: number | null;
  label: string;
  verdict: string | null;
}

/** Track origins in image coordinates, with the path to each terminus.
 *
 *  Origins are drawn on top of the trajectories and larger, because the origin
 *  is the datum: §7 of the architecture infers entry points from where tracks
 *  *begin*, and a plot that weights the path equally buries the signal. */
export function Origins({
  title,
  note,
  points,
  extent,
}: {
  title: string;
  note?: ReactNode;
  points: ScatterPoint[];
  extent: { width: number; height: number };
}) {
  const [hover, setHover] = useState<number | null>(null);
  const id = useId();
  const w = 640;
  const h = Math.round((extent.height / extent.width) * w);
  const sx = (x: number) => (x / extent.width) * w;
  const sy = (y: number) => (y / extent.height) * h;

  return (
    <Frame
      title={title}
      note={note}
      tip={
        hover !== null && points[hover]
          ? {
              x: (sx(points[hover]!.x) / w) * 100,
              y: sy(points[hover]!.y),
              content: (
                <>
                  <b>{points[hover]!.label}</b> · origin{" "}
                  {Math.round(points[hover]!.x)}, {Math.round(points[hover]!.y)}
                  {points[hover]!.verdict ? ` · ${points[hover]!.verdict}` : " · unreviewed"}
                </>
              ),
            }
          : null
      }
    >
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-labelledby={id}>
        <title id={id}>{title}</title>
        <rect x={0} y={0} width={w} height={h} fill="var(--surface-2)" />

        {points.map((p, i) =>
          p.tx !== null && p.ty !== null ? (
            <line
              key={`t${i}`}
              x1={sx(p.x)}
              y1={sy(p.y)}
              x2={sx(p.tx)}
              y2={sy(p.ty)}
              stroke="var(--muted)"
              strokeWidth={1}
              opacity={hover === i ? 0.9 : 0.22}
            />
          ) : null,
        )}

        {points.map((p, i) => (
          <circle
            key={`o${i}`}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={hover === i ? 6 : 4}
            fill={SERIES}
            stroke="var(--surface)"
            strokeWidth={GAP}
            opacity={hover === null || hover === i ? 0.85 : 0.5}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>
    </Frame>
  );
}

/* ------------------------------------------------------- verdict progress */

export interface Segment {
  label: string;
  value: number;
  color: string;
}

/** One stacked bar of review state. Segments are separated by a 2px surface gap
 *  and every one is named in the row beneath, so the colours are a convenience
 *  rather than the only way to read it. */
export function StackedBar({
  title,
  note,
  segments,
}: {
  title: string;
  note?: ReactNode;
  segments: Segment[];
}) {
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  const id = useId();
  if (total === 0) return <Frame title={title} note={note}>{null}</Frame>;

  const w = 640;
  const barH = 22;
  let x = 0;

  return (
    <Frame title={title} note={note}>
      <svg viewBox={`0 0 ${w} ${barH}`} role="img" aria-labelledby={id} height={barH}>
        <title id={id}>{title}</title>
        {segments.map((s) => {
          if (s.value === 0) return null;
          const raw = (s.value / total) * w;
          const width = Math.max(2, raw - GAP);
          const rect = <rect key={s.label} x={x} y={0} width={width} height={barH} rx={2} fill={s.color} />;
          x += raw;
          return rect;
        })}
      </svg>
      <div className="legend">
        {segments.map((s) => (
          <span key={s.label}>
            <i className="swatch" style={{ background: s.color }} aria-hidden="true" />
            {s.label} — {s.value.toLocaleString()}
          </span>
        ))}
      </div>
    </Frame>
  );
}
