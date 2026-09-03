/* The review queue — the return path in Figure 5.
 *
 * This is the console's reason to exist. Every other view reports on the
 * system; this one is the mechanism by which the system improves, because each
 * verdict becomes a labelled training example and each rejection becomes a hard
 * negative. It therefore has to be fast enough that someone actually does it
 * for an hour: the keyboard drives everything, and a verdict posts optimistically
 * so the queue never stalls on a round trip.
 *
 * The keys match the offline HTML review sheet deliberately — J/K, C, R, U,
 * 1-7 — so muscle memory transfers between the two and neither has to be
 * relearned.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, LABEL_KEYS, LABELS } from "../api";
import type { EventRow, Label, Verdict } from "../api";
import { clockUTC, dateUTC, seconds, shortId } from "../format";
import { useAsync } from "../useAsync";
import { Chip, Empty, ErrorBox, Loading } from "../components/ui";

const PAGE = 60;

type Filter = "pending" | "confirmed" | "rejected" | "reclassified" | "";

export function Review({ onVerdict }: { onVerdict: () => void }) {
  const [filter, setFilter] = useState<Filter>("pending");
  const [camera, setCamera] = useState("");
  const [cursor, setCursor] = useState(0);
  const [pending, setPending] = useState<Record<string, Verdict>>({});
  const [failed, setFailed] = useState<string | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  const cameras = useAsync(() => api.cameras(), []);
  const page = useAsync(
    () => api.events({ verdict: filter || undefined, camera_id: camera || undefined, limit: PAGE }),
    [filter, camera],
  );

  const events = page.data?.events ?? [];

  useEffect(() => {
    setCursor(0);
    setPending({});
  }, [filter, camera]);

  /* A verdict is applied to the local row immediately and posted in the
   * background. Waiting for the server before advancing would put a round trip
   * between every keystroke, which is the difference between reviewing 300
   * events in a sitting and giving up after 20. */
  const judge = useCallback(
    (event: EventRow | undefined, verdict: Verdict, label?: Label) => {
      if (!event) return;
      setPending((p) => ({ ...p, [event.id]: verdict }));
      setFailed(null);
      api.setVerdict(event.id, verdict, label).then(onVerdict).catch((err: Error) => {
        // Roll the row back rather than leaving a verdict on screen that the
        // store never received — a silent divergence here corrupts the
        // training set, which is the one thing this view must not do.
        setPending((p) => {
          const next = { ...p };
          delete next[event.id];
          return next;
        });
        setFailed(`Could not record verdict for ${shortId(event.id)}: ${err.message}`);
      });
      setCursor((c) => Math.min(events.length - 1, c + 1));
    },
    [events.length, onVerdict],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement;
      if (target.tagName === "SELECT" || target.tagName === "INPUT") return;

      const key = e.key.toLowerCase();
      const current = events[cursor];

      if (key === "j" || key === "arrowdown") {
        setCursor((c) => Math.min(events.length - 1, c + 1));
      } else if (key === "k" || key === "arrowup") {
        setCursor((c) => Math.max(0, c - 1));
      } else if (key === "c") {
        judge(current, "confirmed");
      } else if (key === "r") {
        judge(current, "rejected");
      } else if (key === "u" && current) {
        setPending((p) => {
          const next = { ...p };
          delete next[current.id];
          return next;
        });
        return;
      } else if (LABEL_KEYS[key]) {
        judge(current, "reclassified", LABEL_KEYS[key]);
      } else {
        return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cursor, events, judge]);

  useEffect(() => {
    gridRef.current
      ?.querySelector<HTMLElement>(`[data-index="${cursor}"]`)
      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [cursor]);

  const judged = Object.keys(pending).length;

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Review queue</h1>
          <p className="lede">
            Every verdict here becomes a labelled training example, and every rejection becomes
            a hard negative — which is worth more per example than a positive when the
            plausibility classifier is trained. This is the loop that makes the system improve
            after delivery.
          </p>
        </div>
      </div>

      <div className="toolbar">
        <label>
          Show
          <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)}>
            <option value="pending">Awaiting a verdict</option>
            <option value="confirmed">Confirmed</option>
            <option value="rejected">Rejected</option>
            <option value="reclassified">Reclassified</option>
            <option value="">All events</option>
          </select>
        </label>
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
        <span className="keys">
          <kbd>J</kbd>/<kbd>K</kbd> move · <kbd>C</kbd> confirm · <kbd>R</kbd> reject ·{" "}
          <kbd>U</kbd> undo · <kbd>1</kbd>–<kbd>7</kbd> reclassify
        </span>
        <span className="keys" style={{ marginLeft: "auto" }}>
          {judged > 0 ? `${judged} judged this session · ` : ""}
          {page.data ? `${page.data.total.toLocaleString()} match` : ""}
        </span>
      </div>

      {failed ? <div className="err">{failed}</div> : null}
      {page.error ? <ErrorBox error={page.error} /> : null}
      {page.loading ? <Loading what="events" /> : null}

      {!page.loading && !page.error && events.length === 0 ? (
        <Empty>
          {filter === "pending"
            ? "Nothing awaiting a verdict. The queue is clear."
            : "No events match this filter."}
        </Empty>
      ) : null}

      <div className="review-grid" ref={gridRef}>
        {events.map((event, i) => (
          <Card
            key={event.id}
            event={event}
            index={i}
            selected={i === cursor}
            verdict={pending[event.id] ?? event.verdict}
            onSelect={() => setCursor(i)}
            onJudge={(v, l) => judge(event, v, l)}
          />
        ))}
      </div>

      {page.data && page.data.total > events.length ? (
        <p className="chart-note" style={{ marginTop: 14 }}>
          Showing the first {events.length} of {page.data.total.toLocaleString()}. Work the queue
          down and reload for the next batch.
        </p>
      ) : null}
    </div>
  );
}

const BADGE: Record<Verdict, string> = {
  confirmed: "var(--good)",
  rejected: "var(--critical)",
  reclassified: "var(--accent)",
};

function Card({
  event,
  index,
  selected,
  verdict,
  onSelect,
  onJudge,
}: {
  event: EventRow;
  index: number;
  selected: boolean;
  verdict: Verdict | null;
  onSelect: () => void;
  onJudge: (verdict: Verdict, label?: Label) => void;
}) {
  const [showLabels, setShowLabels] = useState(false);
  const shown = event.corrected_label ?? event.label;

  return (
    <article
      className={`evt${selected ? " sel" : ""}`}
      data-verdict={verdict ?? ""}
      data-index={index}
      onClick={onSelect}
    >
      <div className="shot">
        {event.has_keyframe ? (
          <img loading="lazy" src={api.keyframeUrl(event.id)} alt={`Keyframe for ${shown}`} />
        ) : (
          <div className="noshot">no keyframe — run was --no-clips</div>
        )}
        {verdict ? (
          <span className="verdict-badge" style={{ background: BADGE[verdict] }}>
            {verdict}
          </span>
        ) : null}
      </div>

      <div className="meta">
        <div>
          <b>{event.camera_id}</b>{" "}
          <span className="line">
            {dateUTC(event.started_at)} {clockUTC(event.started_at)}Z
          </span>
        </div>
        <div className="line">
          {shown} · {seconds(event.duration_s)} · {event.n_detections} det · straight{" "}
          {event.straightness.toFixed(2)}
        </div>
        <div className="line">{shortId(event.id)}</div>
      </div>

      <div className="acts">
        <button
          className="btn good"
          onClick={(e) => {
            e.stopPropagation();
            onJudge("confirmed");
          }}
        >
          Confirm
        </button>
        <button
          className="btn bad"
          onClick={(e) => {
            e.stopPropagation();
            onJudge("rejected");
          }}
        >
          Reject
        </button>
        <button
          className="btn"
          onClick={(e) => {
            e.stopPropagation();
            setShowLabels((s) => !s);
          }}
          aria-expanded={showLabels}
        >
          Reclassify
        </button>
        {event.has_clip ? (
          <a
            className="btn"
            href={api.clipUrl(event.id)}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => e.stopPropagation()}
          >
            Clip
          </a>
        ) : null}

        {showLabels
          ? LABELS.map((label, i) => (
              <button
                key={label}
                className="btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onJudge("reclassified", label);
                  setShowLabels(false);
                }}
              >
                <kbd>{i + 1}</kbd> {label}
              </button>
            ))
          : null}

        {event.verified_by && !verdict ? (
          <Chip tone="plain">reviewed by {event.verified_by}</Chip>
        ) : null}
      </div>
    </article>
  );
}
