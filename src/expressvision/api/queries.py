"""Read queries and analytics over the collector store.

Everything the dashboard shows is computed here, so the rules about *how* a
number may be presented live in one place rather than being re-decided in each
view.

Two of those rules are load-bearing:

* **A rate is never emitted without its denominator.** Counts divided by
  observed camera-hours, or nothing. Where a store predates uptime recording the
  rate is ``None`` and the view says "coverage unknown" — which is a different
  statement from "no activity", and the difference is the whole point.
* **Times are reported in the timezone they were stored in.** The store has no
  site timezone (the architecture puts it on ``Site``, which a single-camera
  collector has no table for), so time-of-day histograms are labelled UTC rather
  than silently rendered as if they were local. A peak-activity chart that is
  quietly five and a half hours out is worse than one that admits its frame.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

from ..store import Store

VERDICTS = ("confirmed", "rejected", "reclassified")

# Mirrors review.REVIEW_LABELS. Duplicated as a tuple rather than imported
# because that module pulls in cv2, and the API process has no reason to load a
# computer-vision stack to render a dropdown.
LABELS = ("rodent", "bird", "reptile", "carnivore", "insect", "human", "other")


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cur]


def _loads(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return fallback


# --------------------------------------------------------------------------
# coverage — the denominator behind every rate on the dashboard
# --------------------------------------------------------------------------


def coverage(store: Store, camera_id: str | None = None) -> dict[str, Any]:
    """Observed stream-seconds, and whether that figure can be trusted.

    ``complete`` is false when any run is missing ``observed_seconds`` — a run
    still in flight, or one written before the column existed. The total is then
    a floor, not a measurement, and rates derived from it would overstate.
    """
    sql = (
        "SELECT COUNT(*) n,"
        "       SUM(COALESCE(observed_seconds, 0)) secs,"
        "       SUM(CASE WHEN observed_seconds IS NULL THEN 1 ELSE 0 END) unknown"
        "  FROM run"
    )
    params: tuple[Any, ...] = ()
    if camera_id:
        sql += " WHERE camera_id = ?"
        params = (camera_id,)

    row = store.conn.execute(sql, params).fetchone()
    runs = int(row["n"] or 0)
    unknown = int(row["unknown"] or 0)
    seconds = float(row["secs"] or 0.0)

    return {
        "runs": runs,
        "runs_missing_uptime": unknown,
        "observed_seconds": seconds,
        "observed_hours": seconds / 3600.0,
        "complete": runs > 0 and unknown == 0,
    }


def _per_hour(count: int, cov: dict[str, Any]) -> float | None:
    """Count per observed camera-hour, or None when that cannot be said.

    Returning None rather than 0.0 is deliberate: the view renders it as an
    explicit "coverage unknown", and a zero would be indistinguishable from a
    genuinely quiet camera.
    """
    if not cov["complete"] or cov["observed_hours"] <= 0:
        return None
    return count / cov["observed_hours"]


# --------------------------------------------------------------------------
# overview
# --------------------------------------------------------------------------


def overview(store: Store) -> dict[str, Any]:
    counts = store.counts()
    cov = coverage(store)

    cameras = int(
        store.conn.execute("SELECT COUNT(DISTINCT camera_id) n FROM run").fetchone()["n"] or 0
    )
    reviewed = counts["confirmed"] + counts["rejected"]
    reclassified = int(
        store.conn.execute(
            "SELECT COUNT(*) n FROM event WHERE verdict = 'reclassified'"
        ).fetchone()["n"]
        or 0
    )
    judged = reviewed + reclassified

    span = store.conn.execute(
        "SELECT MIN(started_at) first, MAX(started_at) last FROM event"
    ).fetchone()

    return {
        "demo": store.is_demo,
        "cameras": cameras,
        "counts": counts | {"reclassified": reclassified, "judged": judged},
        "coverage": cov,
        "events_per_camera_hour": _per_hour(counts["events"], cov),
        # The share of events a human has ruled on. This is the health metric
        # for the return path in Figure 5 — a system whose verdict backlog grows
        # without bound has stopped learning, whatever its detection numbers say.
        "review_progress": (judged / counts["events"]) if counts["events"] else 0.0,
        "first_event_at": span["first"] if span else None,
        "last_event_at": span["last"] if span else None,
    }


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------

EVENT_COLUMNS = (
    "id, run_id, camera_id, site_id, track_id, label, started_at,"
    " start_t_s, end_t_s, duration_s, n_detections,"
    " displacement_px, straightness, heading_deg,"
    " origin_x, origin_y, terminus_x, terminus_y,"
    " clip_path, keyframe_path, clip_sha256, model_version,"
    " verdict, corrected_label, verified_by, verified_at"
)


def list_events(
    store: Store,
    camera_id: str | None = None,
    verdict: str | None = None,
    label: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 50,
    offset: int = 0,
    order: str = "asc",
) -> dict[str, Any]:
    """A page of the event feed.

    ``verdict="pending"`` selects the unjudged ones, which is the review queue
    and the default the console opens on.
    """
    where: list[str] = []
    params: list[Any] = []

    if camera_id:
        where.append("camera_id = ?")
        params.append(camera_id)
    if verdict == "pending":
        where.append("verdict IS NULL")
    elif verdict in VERDICTS:
        where.append("verdict = ?")
        params.append(verdict)
    if label:
        where.append("COALESCE(corrected_label, label) = ?")
        params.append(label)
    if since:
        where.append("started_at >= ?")
        params.append(since)
    if until:
        where.append("started_at <= ?")
        params.append(until)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    total = int(
        store.conn.execute(f"SELECT COUNT(*) n FROM event{clause}", params).fetchone()["n"]
    )

    direction = "DESC" if order.lower() == "desc" else "ASC"
    rows = _rows(
        store.conn.execute(
            f"SELECT {EVENT_COLUMNS} FROM event{clause}"
            f" ORDER BY started_at {direction}, id {direction} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
    )
    for row in rows:
        row["has_keyframe"] = bool(row.pop("keyframe_path"))
        row["has_clip"] = bool(row.pop("clip_path"))

    return {"total": total, "limit": limit, "offset": offset, "events": rows}


def get_event(store: Store, event_id: str) -> dict[str, Any] | None:
    row = store.conn.execute(
        "SELECT * FROM event WHERE id = ? OR id LIKE ?", (event_id, f"{event_id}%")
    ).fetchone()
    if row is None:
        return None

    event = dict(row)
    event["polyline"] = _loads(event.pop("polyline_json", None), [])
    event["features"] = _loads(event.pop("features_json", None), {})
    event["has_keyframe"] = bool(event.pop("keyframe_path"))
    event["has_clip"] = bool(event.pop("clip_path"))
    return event


def media_path(store: Store, event_id: str, kind: str) -> str | None:
    """The on-disk path for an event's keyframe or clip.

    Returned as the raw stored string; the caller is responsible for confining
    it to the media root before opening it. Keeping that check at the HTTP
    boundary rather than here means there is exactly one place to audit.
    """
    column = "keyframe_path" if kind == "keyframe" else "clip_path"
    row = store.conn.execute(
        f"SELECT {column} p FROM event WHERE id = ? OR id LIKE ?",
        (event_id, f"{event_id}%"),
    ).fetchone()
    return row["p"] if row and row["p"] else None


def set_verdict(
    store: Store,
    event_id: str,
    verdict: str,
    corrected_label: str | None,
    user: str,
) -> bool:
    row = store.conn.execute(
        "SELECT id FROM event WHERE id = ? OR id LIKE ?", (event_id, f"{event_id}%")
    ).fetchone()
    if row is None:
        return False
    return store.set_verdict(row["id"], verdict, corrected_label, user)


# --------------------------------------------------------------------------
# cameras and runs
# --------------------------------------------------------------------------


def cameras(store: Store) -> list[dict[str, Any]]:
    """Per-camera rollup, every rate normalised by that camera's own uptime."""
    ids = [
        row["camera_id"]
        for row in store.conn.execute(
            "SELECT camera_id FROM run"
            " UNION SELECT camera_id FROM event"
            " ORDER BY camera_id"
        )
    ]

    out: list[dict[str, Any]] = []
    for camera_id in ids:
        cov = coverage(store, camera_id)
        stats = store.conn.execute(
            "SELECT COUNT(*) n,"
            "       SUM(CASE WHEN verdict IS NULL THEN 1 ELSE 0 END) pending,"
            "       SUM(CASE WHEN verdict = 'confirmed' THEN 1 ELSE 0 END) confirmed,"
            "       SUM(CASE WHEN verdict = 'rejected' THEN 1 ELSE 0 END) rejected,"
            "       MAX(started_at) last_event"
            "  FROM event WHERE camera_id = ?",
            (camera_id,),
        ).fetchone()

        funnel_row = store.conn.execute(
            "SELECT SUM(frames_processed) processed, SUM(frames_gated) gated,"
            "       SUM(tiles) tiles, SUM(tamper_frames) tamper,"
            "       SUM(global_change_frames) global_change"
            "  FROM funnel WHERE camera_id = ?",
            (camera_id,),
        ).fetchone()

        processed = int(funnel_row["processed"] or 0)
        gated = int(funnel_row["gated"] or 0)
        events = int(stats["n"] or 0)

        out.append(
            {
                "camera_id": camera_id,
                "events": events,
                "pending": int(stats["pending"] or 0),
                "confirmed": int(stats["confirmed"] or 0),
                "rejected": int(stats["rejected"] or 0),
                "last_event_at": stats["last_event"],
                "last_run_at": store.conn.execute(
                    "SELECT MAX(started_at) t FROM run WHERE camera_id = ?", (camera_id,)
                ).fetchone()["t"],
                "coverage": cov,
                "events_per_camera_hour": _per_hour(events, cov),
                "gate_pass_rate": (gated / processed) if processed else None,
                "tiles_per_gated_frame": (
                    int(funnel_row["tiles"] or 0) / gated if gated else None
                ),
                # Surfaced per camera because a fouled lens must read as a
                # maintenance alert, not as a quiet night.
                "tamper_frames": int(funnel_row["tamper"] or 0),
                "global_change_frames": int(funnel_row["global_change"] or 0),
            }
        )
    return out


def runs(store: Store, limit: int = 100) -> list[dict[str, Any]]:
    rows = _rows(
        store.conn.execute(
            "SELECT r.id, r.started_at, r.finished_at, r.camera_id, r.source,"
            "       r.fps, r.observed_seconds,"
            "       f.frames_read, f.frames_processed, f.frames_gated,"
            "       f.global_change_frames, f.tamper_frames, f.rois, f.tiles,"
            "       f.detections, f.tracks_created, f.tracks_confirmed, f.events,"
            "       f.rejected_json"
            "  FROM run r LEFT JOIN funnel f ON f.run_id = r.id"
            " ORDER BY r.started_at DESC LIMIT ?",
            (limit,),
        )
    )
    for row in rows:
        row["rejected"] = _loads(row.pop("rejected_json", None), {})
    return rows


def funnel(store: Store, run_id: str | None = None) -> dict[str, Any]:
    """The cascade from Figure 2, as measured rather than as designed.

    Each stage carries its share of the previous one, because the discard rates
    are the sizing assumption the architecture rests on and the only way to know
    whether they hold on a given site is to read them off real footage.
    """
    sql = (
        "SELECT SUM(frames_read) frames_read, SUM(frames_processed) frames_processed,"
        "       SUM(frames_gated) frames_gated, SUM(rois) rois, SUM(tiles) tiles,"
        "       SUM(detections) detections, SUM(tracks_created) tracks_created,"
        "       SUM(tracks_confirmed) tracks_confirmed, SUM(events) events,"
        "       SUM(global_change_frames) global_change_frames,"
        "       SUM(tamper_frames) tamper_frames"
        "  FROM funnel"
    )
    params: tuple[Any, ...] = ()
    if run_id:
        sql += " WHERE run_id = ?"
        params = (run_id,)

    row = store.conn.execute(sql, params).fetchone()
    value = {key: int(row[key] or 0) for key in row.keys()}  # noqa: SIM118 — sqlite3.Row

    order = [
        ("frames_read", "Frames read"),
        ("frames_processed", "Frames processed"),
        ("frames_gated", "Passed motion gate"),
        ("rois", "ROIs"),
        ("tiles", "Tiles inferred"),
        ("detections", "Detections"),
        ("tracks_created", "Tracks created"),
        ("tracks_confirmed", "Tracks confirmed"),
        ("events", "Events"),
    ]

    stages = []
    previous: int | None = None
    for key, label in order:
        count = value[key]
        stages.append(
            {
                "key": key,
                "label": label,
                "count": count,
                "share_of_previous": (count / previous) if previous else None,
            }
        )
        previous = count

    processed, gated, tiles = (
        value["frames_processed"],
        value["frames_gated"],
        value["tiles"],
    )
    return {
        "stages": stages,
        "gate_pass_rate": (gated / processed) if processed else None,
        "tiles_per_gated_frame": (tiles / gated) if gated else None,
        # What gating actually bought, against inferring on every frame. The
        # architecture assumes ~26x; this is the measured figure.
        "inference_saving": (processed / tiles) if tiles else None,
        "global_change_frames": value["global_change_frames"],
        "tamper_frames": value["tamper_frames"],
    }


# --------------------------------------------------------------------------
# rejections — the hard negatives
# --------------------------------------------------------------------------

# The features the validator gates on. Shown per rejection reason so a threshold
# can be judged against the distribution it is actually cutting, rather than
# tuned blind.
GATE_FEATURES = ("extent_frac", "straightness", "area_cv", "n_detections", "duration_s")


def rejections(store: Store) -> dict[str, Any]:
    reasons = _rows(
        store.conn.execute(
            "SELECT reason, COUNT(*) count FROM rejection GROUP BY reason ORDER BY count DESC"
        )
    )

    for reason in reasons:
        vectors = [
            _loads(row["features_json"], {})
            for row in store.conn.execute(
                "SELECT features_json FROM rejection WHERE reason = ?", (reason["reason"],)
            )
        ]
        reason["features"] = {
            name: _summarise([v[name] for v in vectors if isinstance(v.get(name), int | float)])
            for name in GATE_FEATURES
        }

    return {
        "total": int(store.conn.execute("SELECT COUNT(*) n FROM rejection").fetchone()["n"]),
        "reasons": reasons,
    }


def _summarise(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


# --------------------------------------------------------------------------
# analytics
# --------------------------------------------------------------------------


def analytics(store: Store, camera_id: str | None = None) -> dict[str, Any]:
    """The analytics the current event schema genuinely supports.

    Deliberately short of the list in the architecture: heat maps, zone rollups,
    repeat-visit clustering and period comparison all need zones, incidents or a
    site timezone, none of which a single-camera collector store has. Computing
    them from what is here would mean inventing the missing half.
    """
    clause, params = ("", ())
    if camera_id:
        clause, params = (" WHERE camera_id = ?", (camera_id,))

    rows = _rows(
        store.conn.execute(
            "SELECT started_at, duration_s, heading_deg, straightness, n_detections,"
            "       origin_x, origin_y, terminus_x, terminus_y, verdict,"
            "       COALESCE(corrected_label, label) label"
            f"  FROM event{clause}",
            params,
        )
    )

    cov = coverage(store, camera_id)

    return {
        "coverage": cov,
        "hour_of_day": _hour_histogram(rows, cov),
        "headings": _heading_rose(rows),
        "origins": _origins(rows),
        "labels": _label_counts(rows),
        "durations": _summarise([r["duration_s"] for r in rows if r["duration_s"] is not None]),
        "n": len(rows),
    }


def _hour_histogram(rows: list[dict[str, Any]], cov: dict[str, Any]) -> dict[str, Any]:
    """Events per hour-of-day bin, in UTC.

    The timezone is carried in the payload rather than assumed by the view. A
    site in IST reading a UTC histogram would place its 02:00 peak at 20:30 and
    send a technician on the wrong shift.
    """
    bins = [0] * 24
    for row in rows:
        stamp = row["started_at"] or ""
        if len(stamp) >= 13 and stamp[11:13].isdigit():
            bins[int(stamp[11:13])] += 1
    return {"timezone": "UTC", "bins": bins, "normalised": cov["complete"]}


def _heading_rose(rows: list[dict[str, Any]]) -> list[int]:
    """Direction of travel in 8 compass sectors, 0 = up/north, clockwise."""
    sectors = [0] * 8
    for row in rows:
        heading = row["heading_deg"]
        if heading is None:
            continue
        sectors[int(((heading % 360) + 22.5) // 45) % 8] += 1
    return sectors


def _origins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """First and last observed positions of every track.

    The raw input to the entry-point clustering in §7 of the architecture.
    Clustering itself is deliberately not done here: it is only trustworthy once
    origins are in floor-plane coordinates and frame edges are annotated as
    physical or open, and neither exists yet. Plotted raw, an operator can still
    see whether the origins pile up anywhere.
    """
    out = []
    for row in rows:
        if row["origin_x"] is None or row["origin_y"] is None:
            continue
        out.append(
            {
                "x": row["origin_x"],
                "y": row["origin_y"],
                "tx": row["terminus_x"],
                "ty": row["terminus_y"],
                "label": row["label"],
                "verdict": row["verdict"],
            }
        )
    return out


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = row["label"] or "unlabelled"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def frame_extent(store: Store) -> dict[str, float]:
    """Bounds for plotting track origins, inferred from the tracks themselves.

    The store does not record frame dimensions, so the plot is scaled to the
    data it has. Padded by 5% so points on a frame edge — which is exactly where
    an entry point appears — are not clipped out of the picture.
    """
    row = store.conn.execute(
        "SELECT MIN(origin_x) x0, MAX(origin_x) x1, MIN(origin_y) y0, MAX(origin_y) y1,"
        "       MIN(terminus_x) tx0, MAX(terminus_x) tx1,"
        "       MIN(terminus_y) ty0, MAX(terminus_y) ty1 FROM event"
    ).fetchone()
    if row is None or row["x0"] is None:
        return {"width": 1280.0, "height": 720.0}

    width = max(row["x1"] or 0, row["tx1"] or 0)
    height = max(row["y1"] or 0, row["ty1"] or 0)
    return {
        "width": math.ceil(max(width, 1.0) * 1.05),
        "height": math.ceil(max(height, 1.0) * 1.05),
    }
