"""Shared value types that move between pipeline stages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in full-resolution frame coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def scaled(self, factor: float) -> Box:
        return Box(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor)

    def clipped(self, w: float, h: float) -> Box:
        return Box(
            max(0.0, min(self.x1, w)),
            max(0.0, min(self.y1, h)),
            max(0.0, min(self.x2, w)),
            max(0.0, min(self.y2, h)),
        )

    def iou(self, other: Box) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def gap_to(self, other: Box) -> float:
        """Edge-to-edge distance; 0 when the boxes touch or overlap."""
        dx = max(0.0, max(self.x1 - other.x2, other.x1 - self.x2))
        dy = max(0.0, max(self.y1 - other.y2, other.y1 - self.y2))
        return math.hypot(dx, dy)

    def merged(self, other: Box) -> Box:
        return Box(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )

    def as_int(self) -> tuple[int, int, int, int]:
        return round(self.x1), round(self.y1), round(self.x2), round(self.y2)


@dataclass(frozen=True)
class Detection:
    """One detection in full-resolution frame coordinates.

    Milestone 1 emits motion candidates with ``label="motion"``. When a detector
    is added at stage C these carry real classes and confidences instead, and
    nothing downstream changes shape.
    """

    box: Box
    score: float = 1.0
    label: str = "motion"
    frame_idx: int = 0


@dataclass
class TrackPoint:
    frame_idx: int
    t_s: float
    cx: float
    cy: float
    area: float
    score: float
    # The detection box at this instant. Needed because evidence and training
    # data are drawn from one *chosen* frame of the track — usually the clearest
    # view of the animal — and the box shown must be the box from that frame,
    # not wherever the track happened to end.
    box: Box | None = None


@dataclass
class Track:
    """A tentative object identity, accumulated across frames."""

    id: int
    camera_id: str
    box: Box
    label: str = "motion"
    points: list[TrackPoint] = field(default_factory=list)
    hits: int = 0
    age: int = 0                 # frames since last successful match
    start_frame: int = 0
    end_frame: int = 0
    start_t_s: float = 0.0
    end_t_s: float = 0.0
    confirmed: bool = False

    # Kalman-style constant-velocity estimate, in pixels/frame.
    vx: float = 0.0
    vy: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_t_s - self.start_t_s)

    @property
    def frame_span(self) -> int:
        return self.end_frame - self.start_frame + 1

    def path_length(self) -> float:
        return sum(
            math.dist((a.cx, a.cy), (b.cx, b.cy))
            for a, b in zip(self.points, self.points[1:])
        )

    def net_displacement(self) -> float:
        if len(self.points) < 2:
            return 0.0
        a, b = self.points[0], self.points[-1]
        return math.dist((a.cx, a.cy), (b.cx, b.cy))

    def extent(self) -> float:
        """Diagonal of the bounding box containing the whole trajectory.

        The primary test of whether an object actually went anywhere, and the
        one feature that separates a pest from swaying vegetation robustly.
        Net displacement cannot do this job: an animal that enters, works along
        a wall and leaves the way it came has near-zero net displacement but
        large extent, while a plant fragment drifting one way for half a second
        has small extent but enough net displacement to look purposeful.
        """
        if len(self.points) < 2:
            return 0.0
        xs = [p.cx for p in self.points]
        ys = [p.cy for p in self.points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def straightness(self) -> float:
        """Net displacement over total path length.

        Near 1.0 is a purposeful traverse; near 0.0 is something oscillating in
        place, which is what wind-blown vegetation looks like to a motion gate.
        """
        path = self.path_length()
        return self.net_displacement() / path if path > 1e-6 else 0.0

    def heading_deg(self) -> float | None:
        """Compass-style heading of travel, 0 = up/north, clockwise."""
        if len(self.points) < 2:
            return None
        a, b = self.points[0], self.points[-1]
        dx, dy = b.cx - a.cx, b.cy - a.cy
        if math.hypot(dx, dy) < 1e-6:
            return None
        return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0

    def area_cv(self) -> float:
        """Coefficient of variation of box area across the track."""
        areas = [p.area for p in self.points if p.area > 0]
        if len(areas) < 2:
            return 0.0
        mean = sum(areas) / len(areas)
        if mean <= 0:
            return 0.0
        var = sum((a - mean) ** 2 for a in areas) / len(areas)
        return math.sqrt(var) / mean

    def origin(self) -> tuple[float, float] | None:
        """First observed position — the input to entry-point clustering."""
        return (self.points[0].cx, self.points[0].cy) if self.points else None

    def terminus(self) -> tuple[float, float] | None:
        return (self.points[-1].cx, self.points[-1].cy) if self.points else None


@dataclass
class GateResult:
    """What the motion gate decided about one frame."""

    rois: list[Box] = field(default_factory=list)
    changed_frac: float = 0.0
    global_change: bool = False
    tampered: bool = False
    warming_up: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.rois)


@dataclass
class Tile:
    """A native-resolution crop handed to the detector, plus the offset needed
    to map results back into full-frame coordinates."""

    x: int
    y: int
    size: int
    roi_indices: list[int] = field(default_factory=list)


@dataclass
class FunnelStats:
    """Per-camera counters mirroring the cascade in the architecture doc.

    Printed at the end of every run so the gate ratio and tile load can be
    checked against real footage instead of assumed.
    """

    frames_read: int = 0
    frames_processed: int = 0
    frames_gated: int = 0
    global_change_frames: int = 0
    tamper_frames: int = 0
    rois: int = 0
    tiles: int = 0
    detections: int = 0
    tracks_created: int = 0
    tracks_confirmed: int = 0
    events: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
