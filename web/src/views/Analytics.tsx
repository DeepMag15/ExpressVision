/* Analytics — only what the current schema can honestly answer.
 *
 * The architecture lists thirteen analytics. Most of them need zones, incidents
 * or a site timezone, none of which a single-camera collector store has, and
 * computing them from what is here would mean inventing the missing half. What
 * remains genuinely derives from stored event fields: when activity happens,
 * which way things travel, where tracks begin, and what they were called.
 *
 * The missing ones are named at the foot of the page rather than omitted
 * silently, so the gap is visible to whoever picks this up next.
 */

import { useState } from "react";
import { api } from "../api";
import { num, seconds } from "../format";
import { useAsync } from "../useAsync";
import { Bars, Columns, Origins, Rose } from "../components/charts";
import { CoverageNote, Empty, ErrorBox, Loading, Stat } from "../components/ui";

const NOT_YET = [
  ["Heat maps", "needs zone polygons and, for a site-wide view, camera homography"],
  ["Zone rollups", "needs zones — 'camera 4' is not an actionable location, 'dock threshold' is"],
  ["Repeat-visit clustering", "needs zones plus enough nights per zone to cluster over"],
  ["Period comparison", "needs uptime per period and temperature as a covariate, so a cold snap does not read as a treatment failure"],
  ["Entry-point ranking", "needs floor-plane coordinates and frame edges marked physical or open; without that a cluster on an open edge is a field-of-view artefact, not a hole in the building"],
  ["Cross-camera incidents", "stage G is not built — one animal crossing three views is still three events"],
];

export function Analytics() {
  const [camera, setCamera] = useState("");
  const cameras = useAsync(() => api.cameras(), []);
  const { data, error, loading } = useAsync(() => api.analytics(camera || undefined), [camera]);

  if (loading) return <div className="view"><Loading what="analytics" /></div>;
  if (error) return <div className="view"><ErrorBox error={error} /></div>;
  if (!data) return null;

  const peak = data.hour_of_day.bins.reduce(
    (best, v, i) => (v > (data.hour_of_day.bins[best] ?? 0) ? i : best),
    0,
  );

  return (
    <div className="view">
      <h1>Analytics</h1>
      <p className="lede">
        Computed from stored track geometry. Times are <b>UTC</b>, not site-local — the store
        carries no site timezone, and a peak-activity chart quietly five and a half hours out
        would send a technician on the wrong shift.
      </p>

      <div className="toolbar">
        <label>
          Camera
          <select value={camera} onChange={(e) => setCamera(e.target.value)}>
            <option value="">All cameras</option>
            {(cameras.data ?? []).map((c) => (
              <option key={c.camera_id} value={c.camera_id}>
                {c.camera_id}
              </option>
            ))}
          </select>
        </label>
      </div>

      <CoverageNote coverage={data.coverage} />

      {data.n === 0 ? (
        <Empty>No events for this selection.</Empty>
      ) : (
        <>
          <div className="grid g4">
            <Stat value={num(data.n)} label="Events analysed" />
            <Stat value={`${String(peak).padStart(2, "0")}:00 UTC`} label="Busiest hour" />
            <Stat
              value={data.durations ? seconds(data.durations.p50) : "unknown"}
              label="Median track duration"
            />
            <Stat value={num(Object.keys(data.labels).length)} label="Distinct classes" />
          </div>

          <h2>When</h2>
          <div className="card">
            <Columns
              title="Events by hour of day (UTC)"
              note="Bin counts over the whole store. Rodent activity should mass well after the building goes quiet."
              values={data.hour_of_day.bins}
              tickEvery={2}
              tickLabel={(i) => String(i).padStart(2, "0")}
              hoverLabel={(i, v) => `${String(i).padStart(2, "0")}:00 UTC — ${v} events`}
            />
          </div>

          <h2>Which way, and from where</h2>
          <div className="grid g2">
            <div className="card">
              <Rose
                title="Direction of travel"
                note="First-to-last displacement, eight sectors. Sector area carries the count."
                sectors={data.headings}
              />
            </div>
            <div className="card">
              <Origins
                title="Track origins and paths"
                note="Where each track was first seen, with a line to where it was last seen. Dense persistent clusters are entry-point candidates."
                points={data.origins}
                extent={data.extent}
              />
            </div>
          </div>

          <div className="callout">
            <p>
              <b>These origins are not yet ranked entry points.</b> Ranking needs two things the
              store does not have: floor-plane coordinates, so clusters from different cameras
              are comparable, and frame edges marked as physical or open — a cluster sitting on
              an edge the operator marked "open view" is a field-of-view artefact, not a gap in
              the building. Plotted raw, they still show whether origins pile up anywhere.
            </p>
          </div>

          <h2>What</h2>
          <div className="card">
            <Bars
              title="Events by class"
              note="Corrected label where an operator reclassified, otherwise the detector's own call."
              data={Object.entries(data.labels).map(([label, value]) => ({ label, value }))}
              labelWidth={120}
            />
          </div>
        </>
      )}

      <h2>Not computed, and why</h2>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>Analytic</th>
              <th>Blocked on</th>
            </tr>
          </thead>
          <tbody>
            {NOT_YET.map(([name, why]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{why}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
