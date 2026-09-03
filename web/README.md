# Operator console

React + TypeScript + Vite, talking to the FastAPI service in
[`src/expressvision/api/`](../src/expressvision/api/).

```bash
uv pip install -e ".[web]"    # from the repo root
npm install
npm run build                 # then: uv run exv serve
```

For front-end work, run the two separately so Vite can hot-reload:

```bash
uv run exv serve --dev --reload   # terminal 1: API on :8000
npm run dev                       # terminal 2: console on :5173, proxying /api
```

`npm run build` typechecks first (`tsc --noEmit && vite build`), so a type error
fails the build rather than shipping.

---

## Why the API boundary exists

The console never touches SQLite. It talks HTTP to a documented API, and the API
reads the store. That is the seam the architecture draws between L3 and L4,
brought forward to what exists today: when the store behind it becomes Postgres
and the events arrive from real edge nodes, this front end does not change.

---

## Three rules the code enforces

**A rate with no denominator is `null`, and renders as the word "unknown".**
Never `0`. The types make this hard to get wrong on purpose — `number | null`
rather than `number` — because a zero here is indistinguishable from a genuinely
quiet camera, and telling a food plant it has no pest activity when nobody was
watching is the most dangerous thing this system can do. If you find yourself
writing `?? 0` on a rate, that is the bug.

**Times are UTC and say so.** The store carries no site timezone; the
architecture puts it on the `Site` entity, which a single-camera collector has no
table for. A peak-activity chart quietly 5½ hours out would send a technician on
the wrong shift, so nothing is rendered as local until there is a timezone to
render it in.

**Analytics stop where the schema stops.** Heat maps, zone rollups, repeat-visit
clustering, period comparison and entry-point *ranking* are named on the
Analytics page with what each is blocked on, rather than approximated from what
happens to be available. Track origins are plotted raw and captioned as not-yet
entry points: ranking them needs floor-plane coordinates and frame edges marked
physical or open, and without that a cluster on an open frame edge is a
field-of-view artefact rather than a hole in the building.

---

## Colour

The surface palette is lifted from `architecture.html` so the console and the
documents read as one product. The **chart colours are not**: the document's
muted ok/info greens measure ΔE 6.3 apart in normal vision, which is too close to
carry meaning in a chart, so the three semantic hues were re-stepped and
validated against these exact surfaces.

| Role | Light | Dark |
|---|---|---|
| confirmed / good | `#17936A` | `#46A37D` |
| reclassified / attention / series | `#B8790C` | `#BD8B12` |
| rejected / alert | `#A2382A` | `#C74F58` |

That is the whole vocabulary. Anything else — pending, unknown coverage — is
muted ink, not a colour. Light mode passes every check clean; dark sits at CVD
ΔE 6.3, which is legal **only** with secondary encoding, so every status chip
pairs its colour with a glyph and a word, and stacked segments carry a 2px
surface gap plus a named legend row. Colour is never the only channel.

No chart carries a legend for a single series: the title names what is plotted.

---

## Layout

```
src/
  api.ts              typed client; null-able fields are load-bearing
  format.ts           formatters that render absence as a word, not a zero
  useAsync.ts         fetch + reload, guarded against the stale-response race
  App.tsx             shell, hash routing, demo banner
  components/
    ui.tsx            stat tiles, status chips, coverage notice
    charts.tsx        bars, columns, rose, origins, stacked bar — inline SVG
  views/
    Overview.tsx      counts, coverage, review progress
    Review.tsx        the verdict queue — the return path in Figure 5
    Pipeline.tsx      the cascade as measured, against its design targets
    Cameras.tsx       per-camera health, uptime-normalised
    Analytics.tsx     when / which way / from where / what
    Runs.tsx          one row per pass of the collector
```

Charts are hand-rolled inline SVG rather than a charting library: every form here
is a bar, a column, a sector or a dot, and the mark specs matter more than the
breadth.

---

## Not built yet

- **Authentication.** There is none. `exv serve` binds to localhost; exposing it
  would publish evidence clips to the network. §10 of the architecture wants RBAC
  plus a log of who viewed which clip. **Required before this goes anywhere but a
  laptop.**
- **Live wall.** Needs MediaMTX and a real RTSP source.
- **Survey findings on the camera page.** `exv survey` produces them; the
  collector store has nowhere to put them. Needs a `camera` table.
