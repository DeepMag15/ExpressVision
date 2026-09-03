"""SQLite store for the collector.

A subset of the full data model — the tables a single-site collector needs. Two
things here are worth defending:

* **Rejections are persisted, with their features.** Discarded tracks are the
  hard negatives, and hard negatives are worth more per example than positives
  when training the plausibility classifier. Throwing them away means paying for
  a second collection round later.
* **Events carry a verdict column from day one.** That is the return path in the
  architecture — the operator's Confirm / Reject / Reclassify is the training
  pipeline, so the column exists before the review UI does.
* **Runs record how long they actually observed.** Every rate the dashboard
  shows is divided by observed camera-hours, and a store that does not carry the
  denominator can only show raw counts. Raw counts are how these systems lie: a
  camera offline for three nights produces zero events, which an un-normalised
  view renders as *zero pest activity*. ``observed_seconds`` is that denominator.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from .assembler import Event
from .types import FunnelStats

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS run (
    id           TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    camera_id    TEXT NOT NULL,
    source       TEXT NOT NULL,
    config_json  TEXT,
    fps              REAL,   -- effective rate after frame_stride
    observed_seconds REAL    -- stream time actually watched; the rate denominator
);

CREATE TABLE IF NOT EXISTS event (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    site_id         TEXT NOT NULL,
    track_id        INTEGER NOT NULL,
    label           TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    start_t_s       REAL, end_t_s REAL, duration_s REAL,
    n_detections    INTEGER,
    displacement_px REAL, straightness REAL, heading_deg REAL,
    origin_x        REAL, origin_y REAL,
    terminus_x      REAL, terminus_y REAL,
    polyline_json   TEXT,
    features_json   TEXT,
    clip_path       TEXT, keyframe_path TEXT, clip_sha256 TEXT,
    model_version   TEXT,
    verdict         TEXT,          -- NULL | confirmed | rejected | reclassified
    corrected_label TEXT,
    verified_by     TEXT,
    verified_at     TEXT,
    FOREIGN KEY (run_id) REFERENCES run(id)
);

CREATE TABLE IF NOT EXISTS rejection (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    camera_id     TEXT NOT NULL,
    track_id      INTEGER NOT NULL,
    reason        TEXT NOT NULL,
    start_t_s     REAL, end_t_s REAL,
    n_detections  INTEGER,
    features_json TEXT
);

CREATE TABLE IF NOT EXISTS funnel (
    run_id               TEXT PRIMARY KEY,
    camera_id            TEXT NOT NULL,
    frames_read          INTEGER, frames_processed INTEGER, frames_gated INTEGER,
    global_change_frames INTEGER, tamper_frames INTEGER,
    rois                 INTEGER, tiles INTEGER, detections INTEGER,
    tracks_created       INTEGER, tracks_confirmed INTEGER, events INTEGER,
    rejected_json        TEXT
);

CREATE INDEX IF NOT EXISTS idx_event_camera_time ON event(camera_id, started_at);
CREATE INDEX IF NOT EXISTS idx_event_verdict     ON event(verdict);
CREATE INDEX IF NOT EXISTS idx_rejection_reason  ON rejection(reason);
"""


class Store:
    def __init__(self, path: str | Path, check_same_thread: bool = True) -> None:
        """Open (or create) a store.

        ``check_same_thread=False`` is for the API, where a request's connection
        is created on one threadpool worker and used on another. That is safe
        here because the connection belongs to a single request and is never
        touched by two threads at once — FastAPI completes dependency setup
        before the endpoint runs and teardown after it returns. It is *not* safe
        to share one Store across concurrent requests, which is why the API
        opens a fresh one per request rather than caching it.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns that older stores predate.

        Stores are written on edge nodes and read back weeks later, so a schema
        change must not orphan a database that already holds a night of
        evidence. Missing columns read as NULL, which the API reports as unknown
        coverage rather than as zero — an important difference, since zero
        coverage and unknown coverage justify very different conclusions.
        """
        have = {row["name"] for row in self.conn.execute("PRAGMA table_info(run)")}
        for column, decl in (("fps", "REAL"), ("observed_seconds", "REAL")):
            if column not in have:
                self.conn.execute(f"ALTER TABLE run ADD COLUMN {column} {decl}")

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value)
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    @property
    def is_demo(self) -> bool:
        """Whether this store holds synthesised data rather than real footage.

        The dashboard reads this to banner every view. A screenshot of invented
        pest activity that is not marked as invented becomes a claim about a
        system that has never seen a real rat.
        """
        return self.get_meta("demo") == "1"

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start_run(self, run_id: str, camera_id: str, source: str, config_json: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO run (id, started_at, camera_id, source, config_json)"
            " VALUES (?,?,?,?,?)",
            (run_id, datetime.now(UTC).isoformat(), camera_id, source, config_json),
        )
        self.conn.commit()

    def finish_run(
        self,
        run_id: str,
        fps: float | None = None,
        observed_seconds: float | None = None,
    ) -> None:
        """Close a run, recording how much footage it actually watched.

        ``observed_seconds`` is stream time, not wall-clock: a run over a file
        observed the length of the file however long the decode took, and a run
        that lost an RTSP link for an hour observed an hour less than the clock
        suggests. Rates are only honest against the former.
        """
        self.conn.execute(
            "UPDATE run SET finished_at = ?, fps = ?, observed_seconds = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), fps, observed_seconds, run_id),
        )
        self.conn.commit()

    def add_event(self, run_id: str, event: Event) -> None:
        ox, oy = event.origin_xy or (None, None)
        tx, ty = event.terminus_xy or (None, None)
        self.conn.execute(
            """INSERT OR REPLACE INTO event (
                id, run_id, camera_id, site_id, track_id, label, started_at,
                start_t_s, end_t_s, duration_s, n_detections,
                displacement_px, straightness, heading_deg,
                origin_x, origin_y, terminus_x, terminus_y,
                polyline_json, features_json,
                clip_path, keyframe_path, clip_sha256, model_version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.id, run_id, event.camera_id, event.site_id, event.track_id,
                event.label, event.started_at,
                event.start_t_s, event.end_t_s, event.duration_s, event.n_detections,
                event.displacement_px, event.straightness, event.heading_deg,
                ox, oy, tx, ty,
                json.dumps(event.polyline), json.dumps(event.features),
                event.clip_path, event.keyframe_path, event.clip_sha256,
                event.model_version,
            ),
        )
        self.conn.commit()

    def add_rejection(
        self,
        run_id: str,
        camera_id: str,
        track_id: int,
        reason: str,
        start_t_s: float,
        end_t_s: float,
        n_detections: int,
        features: dict[str, float],
    ) -> None:
        self.conn.execute(
            """INSERT INTO rejection
               (run_id, camera_id, track_id, reason, start_t_s, end_t_s,
                n_detections, features_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                run_id, camera_id, track_id, reason,
                start_t_s, end_t_s, n_detections, json.dumps(features),
            ),
        )

    def add_funnel(self, run_id: str, camera_id: str, stats: FunnelStats) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO funnel (
                run_id, camera_id, frames_read, frames_processed, frames_gated,
                global_change_frames, tamper_frames, rois, tiles, detections,
                tracks_created, tracks_confirmed, events, rejected_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, camera_id, stats.frames_read, stats.frames_processed,
                stats.frames_gated, stats.global_change_frames, stats.tamper_frames,
                stats.rois, stats.tiles, stats.detections, stats.tracks_created,
                stats.tracks_confirmed, stats.events, json.dumps(stats.rejected),
            ),
        )
        self.conn.commit()

    def set_verdict(
        self,
        event_id: str,
        verdict: str,
        corrected_label: str | None = None,
        user: str = "local",
    ) -> bool:
        """Record an operator's judgement — the return path in Figure 5."""
        cur = self.conn.execute(
            """UPDATE event
               SET verdict = ?, corrected_label = ?, verified_by = ?, verified_at = ?
               WHERE id = ?""",
            (verdict, corrected_label, user, datetime.now(UTC).isoformat(), event_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def pending_events(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM event WHERE verdict IS NULL"
                " ORDER BY started_at LIMIT ?",
                (limit,),
            )
        )

    def counts(self) -> dict[str, int]:
        def one(sql: str) -> int:
            row = self.conn.execute(sql).fetchone()
            return int(row[0]) if row else 0

        return {
            "runs": one("SELECT COUNT(*) FROM run"),
            "events": one("SELECT COUNT(*) FROM event"),
            "unverified": one("SELECT COUNT(*) FROM event WHERE verdict IS NULL"),
            "confirmed": one("SELECT COUNT(*) FROM event WHERE verdict = 'confirmed'"),
            "rejected": one("SELECT COUNT(*) FROM event WHERE verdict = 'rejected'"),
            "discarded_tracks": one("SELECT COUNT(*) FROM rejection"),
        }
