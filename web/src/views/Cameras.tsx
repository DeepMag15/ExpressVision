/* Per-camera health.
 *
 * The column that matters most is coverage, and the reason is the failure this
 * whole system has to avoid: a camera offline for three nights produces zero
 * events, and an un-normalised dashboard renders that as zero pest activity —
 * the most dangerous possible false reassurance in a food plant. So the table
 * leads with observed hours, reports every rate per camera-hour, and flags a
 * camera whose coverage has fallen behind the fleet.
 *
 * Tamper is given equal weight for the same reason: a fouled lens also produces
 * quiet, and quiet is indistinguishable from clean unless something says so.
 */

import { api } from "../api";
import type { CameraRow } from "../api";
import { agoUTC, hours, num, pct } from "../format";
import { useAsync } from "../useAsync";
import { Bars } from "../components/charts";
import { Chip, Empty, ErrorBox, Loading } from "../components/ui";

/** A camera is flagged when its own numbers say it may not be watching. */
function assess(camera: CameraRow, medianHours: number): {
  tone: "good" | "warn" | "bad";
  text: string;
  why: string;
} {
  if (camera.tamper_frames > 1000) {
    return {
      tone: "bad",
      text: "lens fouled",
      why: "Sustained Laplacian-variance collapse. Quiet from this camera means nothing until the lens is cleaned.",
    };
  }
  if (camera.coverage.observed_hours < medianHours * 0.85) {
    return {
      tone: "bad",
      text: "coverage short",
      why: `Observed ${hours(camera.coverage.observed_seconds)} against a fleet median of ${hours(medianHours * 3600)}. Missing events here are missing coverage, not absence of pests.`,
    };
  }
  if (camera.gate_pass_rate !== null && camera.gate_pass_rate > 0.15) {
    return {
      tone: "warn",
      text: "gate noisy",
      why: "The gate is passing far more than a quiet overnight scene should. Something in view moves constantly — tune this camera before trusting its tile budget.",
    };
  }
  return { tone: "good", text: "watching", why: "Coverage and gate rate are within the expected band." };
}

export function Cameras() {
  const { data, error, loading } = useAsync(() => api.cameras(), []);

  if (loading) return <div className="view"><Loading what="cameras" /></div>;
  if (error) return <div className="view"><ErrorBox error={error} /></div>;
  if (!data || data.length === 0)
    return (
      <div className="view">
        <h1>Cameras</h1>
        <Empty>No runs in this store yet. Run the collector against a source first.</Empty>
      </div>
    );

  const hoursList = [...data.map((c) => c.coverage.observed_hours)].sort((a, b) => a - b);
  const medianHours = hoursList[Math.floor(hoursList.length / 2)] ?? 0;
  const flagged = data.filter((c) => assess(c, medianHours).tone !== "good");

  const normalised = data.every((c) => c.events_per_camera_hour !== null);

  return (
    <div className="view">
      <h1>Cameras</h1>
      <p className="lede">
        Rates are per observed camera-hour, so a camera that watched half as long does not look
        half as quiet. Counts alone would make an offline camera the calmest site on the estate.
      </p>

      {flagged.length > 0 ? (
        <div className="callout bad">
          <p>
            <b>
              {flagged.length} camera{flagged.length === 1 ? "" : "s"} may not be watching
              properly.
            </b>{" "}
            Their event counts cannot be read as pest activity until this is resolved.
          </p>
          {flagged.map((c) => (
            <p key={c.camera_id}>
              <b>{c.camera_id}</b> — {assess(c, medianHours).why}
            </p>
          ))}
        </div>
      ) : null}

      {normalised ? (
        <div className="card" style={{ marginBottom: 18 }}>
          <Bars
            title="Events per observed camera-hour"
            note="Normalised by uptime. A camera with less coverage is not thereby quieter."
            data={[...data]
              .sort((a, b) => (b.events_per_camera_hour ?? 0) - (a.events_per_camera_hour ?? 0))
              .map((c) => ({
                label: c.camera_id,
                value: c.events_per_camera_hour ?? 0,
                note: `${c.events} events`,
                tone: assess(c, medianHours).tone === "bad" ? ("critical" as const) : ("series" as const),
              }))}
            format={(n) => n.toFixed(3)}
            labelWidth={130}
          />
        </div>
      ) : null}

      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>Camera</th>
              <th>State</th>
              <th className="num">Observed</th>
              <th className="num">Events</th>
              <th className="num">Per cam-hour</th>
              <th className="num">Pending</th>
              <th className="num">Gate</th>
              <th className="num">Tiles/frame</th>
              <th className="num">Tamper</th>
              <th>Last event</th>
            </tr>
          </thead>
          <tbody>
            {data.map((c) => {
              const state = assess(c, medianHours);
              return (
                <tr key={c.camera_id}>
                  <td>{c.camera_id}</td>
                  <td>
                    <Chip tone={state.tone} title={state.why}>
                      {state.text}
                    </Chip>
                  </td>
                  <td className="num mono">
                    {hours(c.coverage.observed_seconds)}
                    <br />
                    <span style={{ color: "var(--muted)" }}>{c.coverage.runs} runs</span>
                  </td>
                  <td className="num mono">{num(c.events)}</td>
                  <td className="num mono">
                    {c.events_per_camera_hour === null
                      ? "unknown"
                      : c.events_per_camera_hour.toFixed(3)}
                  </td>
                  <td className="num mono">{num(c.pending)}</td>
                  <td className="num mono">{pct(c.gate_pass_rate)}</td>
                  <td className="num mono">
                    {c.tiles_per_gated_frame === null ? "—" : c.tiles_per_gated_frame.toFixed(2)}
                  </td>
                  <td className="num mono">{num(c.tamper_frames)}</td>
                  <td className="mono">{agoUTC(c.last_event_at)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="chart-note" style={{ marginTop: 12 }}>
        Survey findings — lens, mount height, night shutter, usable detection range — are not
        shown here because the collector store does not carry them. They come from{" "}
        <code>exv survey</code>, which writes its own report per camera, and belong on this page
        once the two are joined.
      </p>
    </div>
  );
}
