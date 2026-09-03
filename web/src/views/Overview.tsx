/* Site overview — what the store holds, and whether it can be trusted.
 *
 * The lead figure is deliberately the review backlog rather than the event
 * count. An event count is a measure of how noisy the site is; the backlog is
 * the measure of whether the learning loop is turning, and a queue that grows
 * without bound means the system is frozen at whatever accuracy it shipped with
 * however good the detection numbers look.
 */

import { api } from "../api";
import { hours, num, pct, stampUTC } from "../format";
import { useAsync } from "../useAsync";
import { StackedBar } from "../components/charts";
import { CoverageNote, ErrorBox, Loading, Stat } from "../components/ui";

export function Overview() {
  const { data, error, loading } = useAsync(() => api.overview(), []);

  if (loading) return <div className="view"><Loading what="overview" /></div>;
  if (error) return <div className="view"><ErrorBox error={error} /></div>;
  if (!data) return null;

  const { counts, coverage } = data;

  return (
    <div className="view">
      <h1>Overview</h1>
      <p className="lede">
        {counts.events.toLocaleString()} events across {data.cameras} camera
        {data.cameras === 1 ? "" : "s"}, from {hours(coverage.observed_seconds)} of observed
        footage. Every rate below is divided by observed camera-hours; where that denominator
        is missing the rate reads <i>unknown</i> rather than zero.
      </p>

      <CoverageNote coverage={coverage} />

      <div className="grid g4">
        <Stat
          value={num(counts.unverified)}
          label="Awaiting a verdict"
          sub={`${pct(data.review_progress)} of all events reviewed`}
          tone={counts.unverified > 0 ? "warn" : "good"}
        />
        <Stat
          value={
            data.events_per_camera_hour === null
              ? "unknown"
              : data.events_per_camera_hour.toFixed(3)
          }
          label="Events per camera-hour"
          sub="uptime-normalised"
        />
        <Stat
          value={num(counts.discarded_tracks)}
          label="Discarded tracks retained"
          sub="hard negatives, with features"
        />
        <Stat value={hours(coverage.observed_seconds)} label="Observed footage" sub={`${coverage.runs} runs`} />
      </div>

      <h2>Review progress</h2>
      <div className="card">
        <StackedBar
          title="Operator verdicts"
          note="Each segment is named below; colour is a convenience, not the only way to read it."
          segments={[
            { label: "Confirmed", value: counts.confirmed, color: "var(--good)" },
            { label: "Rejected", value: counts.rejected, color: "var(--critical)" },
            { label: "Reclassified", value: counts.reclassified, color: "var(--accent)" },
            { label: "Awaiting review", value: counts.unverified, color: "var(--line-strong)" },
          ]}
        />
      </div>

      <h2>Span</h2>
      <div className="tw">
        <table>
          <tbody>
            <tr>
              <td>First event</td>
              <td className="mono">{stampUTC(data.first_event_at)}</td>
            </tr>
            <tr>
              <td>Last event</td>
              <td className="mono">{stampUTC(data.last_event_at)}</td>
            </tr>
            <tr>
              <td>Collector runs</td>
              <td className="mono">{num(counts.runs)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p className="chart-note" style={{ marginTop: 16 }}>
        Times are UTC. The store has no site timezone — the architecture puts it on the Site
        entity, which a single-camera collector has no table for — so nothing here is rendered
        as local time.
      </p>
    </div>
  );
}
