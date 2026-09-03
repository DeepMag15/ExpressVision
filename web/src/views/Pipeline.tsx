/* The cascade, as measured rather than as designed.
 *
 * The architecture's sizing rests on two ratios — roughly a 96% gate discard
 * and about 1.5 tiles per gated frame — and they decide how many cameras fit on
 * one edge node. They are assumptions until real footage confirms them, so this
 * view puts the measured figures next to the designed ones and lets the gap
 * speak.
 *
 * The rejection panel exists because the discarded tracks are not waste. They
 * are the hard negatives the plausibility classifier will be trained on, and
 * showing the feature distribution behind each reason is how a threshold gets
 * judged against what it is actually cutting instead of tuned blind.
 */

import { api } from "../api";
import { compact, num, pct } from "../format";
import { useAsync } from "../useAsync";
import { Bars } from "../components/charts";
import { Chip, ErrorBox, Loading, Stat } from "../components/ui";

// The design targets from the architecture, for comparison against measurement.
const DESIGN_GATE_RATE = 0.04;
const DESIGN_TILES_PER_GATED = 1.5;

export function Pipeline() {
  const funnel = useAsync(() => api.funnel(), []);
  const rejections = useAsync(() => api.rejections(), []);

  if (funnel.loading) return <div className="view"><Loading what="the cascade" /></div>;
  if (funnel.error) return <div className="view"><ErrorBox error={funnel.error} /></div>;
  if (!funnel.data) return null;

  const f = funnel.data;
  const stages = f.stages.filter((s) => s.count > 0);
  const gateOff = f.gate_pass_rate !== null && f.gate_pass_rate > DESIGN_GATE_RATE * 3;

  return (
    <div className="view">
      <h1>Cascade</h1>
      <p className="lede">
        Seven stages, each of which exists to throw work away. The compounding discard rate is
        what turns continuous inference on every camera into something an edge box can carry,
        so these are the numbers that decide hardware sizing.
      </p>

      <div className="grid g4">
        <Stat
          value={pct(f.gate_pass_rate)}
          label="Gate pass rate"
          sub={`design target ~${pct(DESIGN_GATE_RATE, 0)} overnight`}
          tone={gateOff ? "warn" : "good"}
        />
        <Stat
          value={f.tiles_per_gated_frame === null ? "unknown" : f.tiles_per_gated_frame.toFixed(2)}
          label="Tiles per gated frame"
          sub={`design target ~${DESIGN_TILES_PER_GATED}`}
        />
        <Stat
          value={f.inference_saving === null ? "unknown" : `${f.inference_saving.toFixed(1)}×`}
          label="Inference saved by gating"
          sub="vs running the detector on every frame"
        />
        <Stat
          value={num(f.tamper_frames)}
          label="Tamper frames"
          sub="fouled or covered lens"
          tone={f.tamper_frames > 0 ? "bad" : "good"}
        />
      </div>

      {gateOff ? (
        <div className="callout">
          <p>
            <b>The gate is passing {pct(f.gate_pass_rate)} of frames.</b> Real overnight
            footage gates in the single digits; a much higher rate means the camera is seeing
            something constant — vegetation in the view, a flickering lamp, a running conveyor —
            and needs per-camera tuning before the tile budget is meaningful.
          </p>
          <p>
            On the synthetic fixture this is expected and not a defect: the swaying plant moves
            in nearly every frame by design.
          </p>
        </div>
      ) : null}

      <h2>Where the work goes</h2>
      <div className="card">
        <Bars
          title="Stages of the cascade"
          note="Bar lengths are log-scaled — the stages span five orders of magnitude, and on a linear axis the last three are a sliver. Exact counts sit at each tip, and the share of the previous stage at the right."
          data={stages.map((s) => ({
            label: s.label,
            value: s.count,
            note: s.share_of_previous === null ? "" : pct(s.share_of_previous),
            tone: s.key === "events" ? ("good" as const) : ("series" as const),
          }))}
          format={compact}
          labelWidth={150}
          scale="log"
        />
      </div>

      <h2>What the validator discarded</h2>
      <p className="lede">
        {rejections.data
          ? `${rejections.data.total.toLocaleString()} tracks were discarded and kept, with the
             feature vector that got each one discarded.`
          : ""}{" "}
        These are the training set for the learned validator that replaces the hand-tuned gates
        — which is why they are stored rather than counted and dropped.
      </p>

      {rejections.error ? <ErrorBox error={rejections.error} /> : null}
      {rejections.loading ? <Loading what="rejections" /> : null}

      {rejections.data && rejections.data.reasons.length > 0 ? (
        <>
          <div className="card">
            <Bars
              title="Rejection reasons"
              note="Each reason is a gate in the validator. A reason that dominates is the threshold worth revisiting first."
              data={rejections.data.reasons.map((r) => ({
                label: r.reason.replace(/_/g, " "),
                value: r.count,
              }))}
              labelWidth={160}
            />
          </div>

          <h2>The distributions behind the thresholds</h2>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>Reason</th>
                  <th className="num">Tracks</th>
                  <th className="num">Extent frac (min / p50 / max)</th>
                  <th className="num">Straightness p50</th>
                  <th className="num">Detections p50</th>
                </tr>
              </thead>
              <tbody>
                {rejections.data.reasons.map((r) => {
                  const e = r.features.extent_frac;
                  const s = r.features.straightness;
                  const n = r.features.n_detections;
                  return (
                    <tr key={r.reason}>
                      <td>{r.reason.replace(/_/g, " ")}</td>
                      <td className="num">{r.count.toLocaleString()}</td>
                      <td className="num mono">
                        {e ? `${e.min.toFixed(3)} / ${e.p50.toFixed(3)} / ${e.max.toFixed(3)}` : "—"}
                      </td>
                      <td className="num mono">{s ? s.p50.toFixed(3) : "—"}</td>
                      <td className="num mono">{n ? n.p50.toFixed(0) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="chart-note" style={{ marginTop: 10 }}>
            Extent — the diagonal of the box containing the whole trajectory — is the primary
            spatial gate, not net displacement. Vegetation swaying about a fixed point cannot
            clear it however long it is watched, and an animal that doubles back still can.
          </p>
        </>
      ) : null}

      <h2>Guards</h2>
      <div className="grid g2">
        <div className="card">
          <p className="kicker">Global change</p>
          <p style={{ margin: "0 0 8px" }}>
            <Chip tone={f.global_change_frames > 0 ? "warn" : "good"}>
              {num(f.global_change_frames)} frames
            </Chip>
          </p>
          <p className="chart-note" style={{ margin: 0 }}>
            A light switch, an IR-cut toggle at dawn, or a bumped camera. The gate resets its
            background model instead of emitting thousands of candidates.
          </p>
        </div>
        <div className="card">
          <p className="kicker">Tamper / defocus</p>
          <p style={{ margin: "0 0 8px" }}>
            <Chip tone={f.tamper_frames > 0 ? "bad" : "good"}>{num(f.tamper_frames)} frames</Chip>
          </p>
          <p className="chart-note" style={{ margin: 0 }}>
            Laplacian variance collapse — a spider web across the lens. This kills more outdoor
            analytics than any other single cause, and it must surface as maintenance rather
            than as silence that reads like "no pests".
          </p>
        </div>
      </div>
    </div>
  );
}
