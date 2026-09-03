/* Collector runs — one row per pass over a source.
 *
 * A run that recorded no observed_seconds is called out rather than shown as a
 * blank cell: it is the reason a rate elsewhere reads "unknown", and tracing
 * that back to a specific run is otherwise guesswork.
 */

import { api } from "../api";
import { compact, hours, num, pct, stampUTC } from "../format";
import { useAsync } from "../useAsync";
import { Chip, Empty, ErrorBox, Loading } from "../components/ui";

export function Runs() {
  const { data, error, loading } = useAsync(() => api.runs(), []);

  if (loading) return <div className="view"><Loading what="runs" /></div>;
  if (error) return <div className="view"><ErrorBox error={error} /></div>;
  if (!data || data.length === 0)
    return (
      <div className="view">
        <h1>Runs</h1>
        <Empty>No runs recorded yet.</Empty>
      </div>
    );

  const missing = data.filter((r) => r.observed_seconds === null);

  return (
    <div className="view">
      <h1>Runs</h1>
      <p className="lede">
        Each row is one pass of the cascade over one source. Observed time is stream time, not
        wall-clock: a run over a file observed the length of the file however long the decode
        took, and a run that lost its stream observed less than the clock suggests.
      </p>

      {missing.length > 0 ? (
        <div className="callout bad">
          <p>
            <b>
              {missing.length} run{missing.length === 1 ? "" : "s"} did not record observed time.
            </b>{" "}
            These predate uptime recording or are still in flight, and they are why coverage is
            reported as incomplete. Re-running them fills the gap; until then rates that depend
            on them read as unknown.
          </p>
        </div>
      ) : null}

      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>Started (UTC)</th>
              <th>Camera</th>
              <th className="num">Observed</th>
              <th className="num">FPS</th>
              <th className="num">Frames</th>
              <th className="num">Gated</th>
              <th className="num">Tiles</th>
              <th className="num">Events</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {data.map((r) => (
              <tr key={r.id}>
                <td className="mono">{stampUTC(r.started_at)}</td>
                <td className="mono">{r.camera_id}</td>
                <td className="num mono">
                  {r.observed_seconds === null ? (
                    <Chip tone="bad">not recorded</Chip>
                  ) : (
                    hours(r.observed_seconds)
                  )}
                </td>
                <td className="num mono">{r.fps === null ? "—" : r.fps.toFixed(1)}</td>
                <td className="num mono">{compact(r.frames_processed)}</td>
                <td className="num mono">
                  {compact(r.frames_gated)}
                  {r.frames_processed && r.frames_gated !== null ? (
                    <>
                      <br />
                      <span style={{ color: "var(--muted)" }}>
                        {pct(r.frames_gated / r.frames_processed)}
                      </span>
                    </>
                  ) : null}
                </td>
                <td className="num mono">{compact(r.tiles)}</td>
                <td className="num mono">{num(r.events)}</td>
                <td className="mono" style={{ maxWidth: 260, wordBreak: "break-all" }}>
                  {r.source}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
