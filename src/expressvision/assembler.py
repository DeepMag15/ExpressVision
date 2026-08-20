"""Event assembly — stage F of the cascade.

Turns a validated track into the evidence package the architecture specifies:
clip with pre- and post-roll, annotated keyframe, simplified trajectory,
timings, heading, and a content hash.

The non-obvious part is timing. A track is only known to have *ended* once it
has gone unmatched for ``max_age`` frames — about two seconds at 15 fps — so at
the moment the tracker closes it, the requested five seconds of post-roll have
not been recorded yet. Cutting immediately would silently truncate every clip.
Events are therefore parked in a pending queue and cut once enough footage has
accumulated behind them.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .config import EventConfig
from .ringbuffer import RingBuffer
from .types import Track


@dataclass
class Event:
    """One evidence package. The unit the review UI and the store work in."""

    id: str
    camera_id: str
    site_id: str
    track_id: int
    label: str
    started_at: str          # ISO-8601 UTC
    start_t_s: float
    end_t_s: float
    duration_s: float
    n_detections: int
    displacement_px: float
    straightness: float
    heading_deg: float | None
    origin_xy: tuple[float, float] | None
    terminus_xy: tuple[float, float] | None
    polyline: list[tuple[float, float]]
    features: dict[str, float] = field(default_factory=dict)
    clip_path: str | None = None
    keyframe_path: str | None = None
    clip_sha256: str | None = None
    model_version: str = "motion-only-v0"


@dataclass
class _Pending:
    track: Track
    features: dict[str, float]
    cut_after_t: float


class EventAssembler:
    def __init__(
        self,
        cfg: EventConfig,
        buffer: RingBuffer,
        out_dir: Path,
        camera_id: str,
        site_id: str,
        fps: float,
        wall_clock_start: datetime | None = None,
    ) -> None:
        self.cfg = cfg
        self.buffer = buffer
        self.camera_id = camera_id
        self.site_id = site_id
        self.fps = max(1.0, fps)
        self.wall_clock_start = wall_clock_start or datetime.now(UTC)

        self.clips_dir = out_dir / "clips" / camera_id
        self.keyframes_dir = out_dir / "keyframes" / camera_id
        if cfg.write_clips:
            self.clips_dir.mkdir(parents=True, exist_ok=True)
        if cfg.write_keyframes:
            self.keyframes_dir.mkdir(parents=True, exist_ok=True)

        self._pending: list[_Pending] = []

    def submit(self, track: Track, features: dict[str, float]) -> None:
        self._pending.append(
            _Pending(track, features, track.end_t_s + self.cfg.post_roll_s)
        )

    def tick(self, now_t_s: float) -> list[Event]:
        """Cut any event whose post-roll window has now been recorded."""
        ready = [p for p in self._pending if now_t_s >= p.cut_after_t]
        if not ready:
            return []
        self._pending = [p for p in self._pending if now_t_s < p.cut_after_t]
        return [self._build(p) for p in ready]

    def flush(self) -> list[Event]:
        """Cut everything still pending — call at end of stream."""
        ready, self._pending = self._pending, []
        return [self._build(p) for p in ready]

    def _build(self, pending: _Pending) -> Event:
        track = pending.track
        event_id = uuid.uuid4().hex[:16]

        # Clamp the track portion so a long dwell cannot demand more history
        # than the ring buffer holds. Evidence keeps the entry, which is the
        # part a reviewer needs, and truncates the loitering.
        track_end = min(track.end_t_s, track.start_t_s + self.cfg.max_clip_s)
        start_t = track.start_t_s - self.cfg.pre_roll_s
        end_t = track_end + self.cfg.post_roll_s
        frames = self.buffer.window(start_t, end_t)

        polyline = simplify([(p.cx, p.cy) for p in track.points], epsilon=2.0)

        event = Event(
            id=event_id,
            camera_id=self.camera_id,
            site_id=self.site_id,
            track_id=track.id,
            label=track.label,
            started_at=self._to_wall_clock(track.start_t_s),
            start_t_s=track.start_t_s,
            end_t_s=track.end_t_s,
            duration_s=track.duration_s,
            n_detections=len(track.points),
            displacement_px=track.net_displacement(),
            straightness=track.straightness(),
            heading_deg=track.heading_deg(),
            origin_xy=track.origin(),
            terminus_xy=track.terminus(),
            polyline=polyline,
            features=pending.features,
        )

        if self.cfg.write_clips and frames:
            event.clip_path, event.clip_sha256 = self._write_clip(event_id, frames)
        if self.cfg.write_keyframes and frames:
            event.keyframe_path = self._write_keyframe(event_id, track, frames)

        return event

    def _to_wall_clock(self, t_s: float) -> str:
        """Stream-relative seconds to absolute UTC.

        For a file this anchors on when the run started; a real deployment
        anchors on the camera's own timestamp so evidence carries true site time.
        """
        return datetime.fromtimestamp(
            self.wall_clock_start.timestamp() + t_s, tz=UTC
        ).isoformat()

    def _write_clip(self, event_id: str, frames: list) -> tuple[str, str]:
        path = self.clips_dir / f"{event_id}.mp4"
        first = frames[0].decode()
        h, w = first.shape[:2]

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
        )
        try:
            for f in frames:
                writer.write(f.decode())
        finally:
            writer.release()

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return str(path), digest

    def _write_keyframe(self, event_id: str, track: Track, frames: list) -> str:
        """Annotated still at the track's largest observation.

        Overlays are also stored as sidecar JSON: a burned-in box is an
        assertion the client cannot check, whereas a clean frame plus separately
        recorded geometry is evidence they can.
        """
        peak = max(track.points, key=lambda p: p.area)
        chosen = min(frames, key=lambda f: abs(f.t_s - peak.t_s))
        img = chosen.decode()

        pts = [(int(p.cx), int(p.cy)) for p in track.points]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (40, 190, 255), 2)
        if pts:
            cv2.circle(img, pts[0], 5, (90, 220, 130), -1)   # origin
            cv2.circle(img, pts[-1], 5, (60, 60, 235), -1)   # terminus

        # The box from the frame we chose, not the track's final position. An
        # animal usually exits the frame, so the last box is clipped to a sliver
        # at the edge — drawing that over a mid-track image would put a wrong
        # box on the picture, and the sidecar geometry feeds pre-annotations
        # that a human is meant to trust as a starting point.
        x1, y1, x2, y2 = (peak.box or track.box).as_int()
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 190, 255), 2)

        # Keep the caption inside the frame — tracks routinely end at an edge,
        # which is where the label is most needed and most easily lost.
        caption = f"{track.label} | {track.duration_s:.1f}s | {len(track.points)} det"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (text_w, text_h), _ = cv2.getTextSize(caption, font, scale, thickness)
        height, width = img.shape[:2]
        text_x = max(4, min(x1, width - text_w - 4))
        text_y = y1 - 8 if y1 - 8 > text_h else min(height - 4, y2 + text_h + 8)
        cv2.putText(
            img, caption, (text_x, text_y), font, scale, (40, 190, 255), thickness,
            cv2.LINE_AA,
        )

        path = self.keyframes_dir / f"{event_id}.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "event_id": event_id,
                    "box": [x1, y1, x2, y2],
                    "polyline": pts,
                    "label": track.label,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(path)


def simplify(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker polyline simplification.

    A 300-point trajectory and its 12-point simplification are visually
    identical and the latter is what gets stored and replayed.
    """
    if len(points) < 3:
        return list(points)

    start, end = points[0], points[-1]
    max_dist, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _perpendicular_distance(points[i], start, end)
        if d > max_dist:
            max_dist, index = d, i

    if max_dist <= epsilon:
        return [start, end]

    left = simplify(points[: index + 1], epsilon)
    right = simplify(points[index:], epsilon)
    return left[:-1] + right


def _perpendicular_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    (px, py), (x1, y1), (x2, y2) = point, line_start, line_end
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return math.dist(point, line_start)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / math.hypot(dx, dy)
