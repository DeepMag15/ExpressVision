"""A synthetic store, for showing the console before real footage exists.

The client's footage is the blocker on everything model-related, and it is not
recoverable by working harder on the software. Meanwhile the console has to be
demonstrable — to Express Pesticides, and to us, so that layout and analytics
decisions are made against a realistic shape of data rather than against five
fixture events.

**Everything this writes is invented.** The store is stamped ``demo=1``, the API
reports that flag, the console banners every view with it, and each rendered
keyframe carries the words burned into the image. That matters more than it
might seem: a screenshot of pest activity that is not
visibly marked as synthetic is a claim about a system that has never seen a real
rat, and this project's whole discipline is about not letting the software say
things the evidence does not support.

The numbers are chosen to match the architecture's own figures — a single-digit
overnight gate rate, ~1.4 tiles per gated frame, a nocturnal activity peak — so
that what the console renders is the shape a real site should produce. Two
deliberate defects are seeded because a demo that shows only the happy path
teaches the wrong thing:

* **A camera that goes offline for three nights.** Its raw event count drops to
  zero, and the console must show that as lost coverage rather than as an
  absence of pests. That failure mode is the most dangerous one in a food plant.
* **A camera whose lens fouls part way through.** Tamper frames climb, and the
  console must surface it as maintenance rather than as a quiet corner.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .store import Store
from .types import FunnelStats

MODEL_VERSION = "demo-detector-v1 (SYNTHETIC — no model was trained)"

# Twelve-hour night at 15 fps, the sizing case in the architecture.
NIGHT_SECONDS = 12 * 3600
FPS = 15.0


@dataclass(frozen=True)
class DemoCamera:
    """One camera's character: how busy it is, and how it misbehaves."""

    id: str
    site_id: str
    name: str
    events_per_night: tuple[int, int]
    # Share of events by L2 class. Kitchens see insects, docks see rodents.
    mix: tuple[tuple[str, float], ...]
    gate_rate: float
    # Nights, counted from the start, on which this camera recorded nothing.
    offline_nights: tuple[int, ...] = ()
    # Night on which the lens fouls; tamper frames climb from here.
    fouls_on_night: int | None = None
    # Origins cluster tightly here, as they do at a real gap under a shutter.
    entry_xy: tuple[float, float] | None = None


CAMERAS = (
    DemoCamera(
        id="cam-dock-01",
        site_id="site-041",
        name="Dock threshold",
        events_per_night=(3, 7),
        mix=(("rodent", 0.82), ("human", 0.12), ("other", 0.06)),
        gate_rate=0.052,
        # The Figure 4 story: a gap under the roller shutter that the origins
        # pile up against, night after night.
        entry_xy=(166.0, 604.0),
    ),
    DemoCamera(
        id="cam-aisle-03",
        site_id="site-041",
        name="Dry store aisle 3",
        events_per_night=(1, 4),
        mix=(("rodent", 0.70), ("other", 0.18), ("human", 0.12)),
        gate_rate=0.031,
    ),
    DemoCamera(
        id="cam-waste-02",
        site_id="site-041",
        name="Waste bay",
        events_per_night=(4, 9),
        mix=(("rodent", 0.55), ("bird", 0.20), ("carnivore", 0.15), ("other", 0.10)),
        gate_rate=0.084,
    ),
    DemoCamera(
        id="cam-prep-01",
        site_id="site-088",
        name="Prep counter",
        events_per_night=(0, 3),
        mix=(("insect", 0.58), ("rodent", 0.24), ("human", 0.18)),
        gate_rate=0.024,
    ),
    DemoCamera(
        id="cam-store-04",
        site_id="site-088",
        name="Back store",
        events_per_night=(0, 2),
        mix=(("rodent", 0.60), ("other", 0.40)),
        gate_rate=0.019,
        # A spider web across the lens on night 9. The single largest cause of
        # silently dead outdoor analytics.
        fouls_on_night=9,
    ),
    DemoCamera(
        id="cam-loft-05",
        site_id="site-088",
        name="Roof void",
        events_per_night=(1, 3),
        mix=(("rodent", 0.78), ("bird", 0.22)),
        gate_rate=0.027,
        # Switch knocked out on a Friday, noticed the following Monday.
        offline_nights=(6, 7, 8),
    ),
)

# Reasons the validator discards a track, with the feature profile that earns
# each one. Ranges mirror what the real validator produced on the fixture.
REJECTIONS = (
    ("confined", 0.42, {"extent_frac": (0.01, 0.07), "straightness": (0.02, 0.14)}),
    ("too_few_detections", 0.24, {"extent_frac": (0.02, 0.30), "straightness": (0.4, 0.98)}),
    ("too_brief", 0.19, {"extent_frac": (0.01, 0.12), "straightness": (0.3, 0.9)}),
    ("oscillating", 0.10, {"extent_frac": (0.03, 0.11), "straightness": (0.02, 0.12)}),
    ("area_unstable", 0.05, {"extent_frac": (0.05, 0.35), "straightness": (0.2, 0.8)}),
)


def seed_demo(
    db_path: Path,
    out_dir: Path,
    nights: int = 14,
    write_keyframes: bool = True,
    seed: int = 41,
    end_date: datetime | None = None,
) -> dict[str, int]:
    """Build a demo store. Returns what it wrote."""
    rng = np.random.default_rng(seed)
    last_night = (end_date or datetime.now(UTC)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    store = Store(db_path)
    store.set_meta("demo", "1")
    store.set_meta("demo_generated_at", datetime.now(UTC).isoformat())
    store.set_meta("demo_seed", str(seed))

    written = {"runs": 0, "events": 0, "rejections": 0, "keyframes": 0}

    try:
        for night in range(nights):
            # Night 0 is the oldest, so the newest data is the most recent date.
            date = last_night - timedelta(days=nights - 1 - night)
            for camera in CAMERAS:
                counts = _seed_night(
                    store, camera, night, date, rng, out_dir, write_keyframes
                )
                for key, value in counts.items():
                    written[key] += value
    finally:
        store.close()

    return written


def _seed_night(
    store: Store,
    camera: DemoCamera,
    night: int,
    date: datetime,
    rng: np.random.Generator,
    out_dir: Path,
    write_keyframes: bool,
) -> dict[str, int]:
    written = {"runs": 0, "events": 0, "rejections": 0, "keyframes": 0}

    if night in camera.offline_nights:
        # No run row at all. This is the honest representation: the camera did
        # not observe, so there is no coverage to divide by, and the console
        # must not read the missing events as a quiet night.
        return written

    run_id = uuid.uuid4().hex[:16]
    # Nights run 19:00 to 07:00 site time; stored UTC, as the real collector does.
    started = date.replace(hour=19, minute=0, tzinfo=UTC) - timedelta(days=1)

    store.start_run(
        run_id,
        camera.id,
        f"rtsp://demo/{camera.id}/main",
        json.dumps({"demo": True, "camera": camera.name}),
    )
    written["runs"] = 1

    low, high = camera.events_per_night
    n_events = int(rng.integers(low, high + 1))

    events = []
    for _ in range(n_events):
        event = _make_event(store, run_id, camera, started, rng, out_dir, write_keyframes)
        events.append(event)
        written["events"] += 1
        if event["keyframe"]:
            written["keyframes"] += 1

    written["rejections"] = _seed_rejections(store, run_id, camera, rng)
    store.add_funnel(
        run_id, camera.id, _funnel_for(camera, night, n_events, written["rejections"], rng)
    )
    store.finish_run(run_id, fps=FPS, observed_seconds=float(NIGHT_SECONDS))
    return written


def _make_event(
    store: Store,
    run_id: str,
    camera: DemoCamera,
    night_start: datetime,
    rng: np.random.Generator,
    out_dir: Path,
    write_keyframes: bool,
) -> dict[str, object]:
    from .assembler import Event

    event_id = uuid.uuid4().hex[:16]
    label = _pick(camera.mix, rng)
    offset_s = _nocturnal_offset(rng)
    started_at = night_start + timedelta(seconds=offset_s)

    origin, terminus, polyline = _trajectory(camera, label, rng)
    duration = float(rng.uniform(1.4, 9.5) if label != "human" else rng.uniform(4.0, 22.0))
    n_detections = max(5, int(duration * FPS * rng.uniform(0.55, 0.95)))

    dx, dy = terminus[0] - origin[0], terminus[1] - origin[1]
    displacement = math.hypot(dx, dy)
    path_length = displacement * float(rng.uniform(1.05, 1.9))
    heading = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0

    xs = [p[0] for p in polyline]
    ys = [p[1] for p in polyline]
    extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    diagonal = math.hypot(1280.0, 720.0)

    features = {
        "n_detections": float(n_detections),
        "duration_s": duration,
        "displacement_px": displacement,
        "extent_px": extent,
        "extent_frac": extent / diagonal,
        "path_length_px": path_length,
        "straightness": displacement / path_length if path_length else 0.0,
        "area_cv": float(rng.uniform(0.12, 0.55)),
        "mean_area_px": float(rng.uniform(280, 1400)),
        "mean_speed_px_s": path_length / duration if duration else 0.0,
        "mean_score": float(rng.uniform(0.42, 0.93)),
    }

    keyframe = None
    if write_keyframes:
        keyframe = _render_keyframe(out_dir, camera, event_id, label, polyline, duration,
                                    n_detections)

    event = Event(
        id=event_id,
        camera_id=camera.id,
        site_id=camera.site_id,
        track_id=int(rng.integers(1, 400)),
        label=label,
        started_at=started_at.isoformat(),
        start_t_s=offset_s,
        end_t_s=offset_s + duration,
        duration_s=duration,
        n_detections=n_detections,
        displacement_px=displacement,
        straightness=features["straightness"],
        heading_deg=heading,
        origin_xy=origin,
        terminus_xy=terminus,
        polyline=polyline,
        features=features,
        keyframe_path=str(keyframe) if keyframe else None,
        model_version=MODEL_VERSION,
    )
    store.add_event(run_id, event)

    # Roughly three in five have been through a human. A demo showing an empty
    # verdict queue would hide the review backlog, which is the metric that says
    # whether the learning loop is actually turning.
    if rng.random() < 0.6:
        verdict, corrected = _verdict_for(label, rng)
        store.set_verdict(event_id, verdict, corrected, "demo-operator")

    return {"id": event_id, "keyframe": keyframe}


def _verdict_for(label: str, rng: np.random.Generator) -> tuple[str, str | None]:
    """What an operator concluded.

    Confirmation is likelier for the classes the pixels support. Insects are the
    least reliable call at room distance, which §6 of the architecture is candid
    about, so they are reclassified and rejected more often.
    """
    roll = rng.random()
    if label in {"rodent", "human"}:
        if roll < 0.78:
            return "confirmed", None
        if roll < 0.92:
            return "rejected", None
        return "reclassified", "other"
    if roll < 0.52:
        return "confirmed", None
    if roll < 0.80:
        return "rejected", None
    return "reclassified", str(rng.choice(["rodent", "other", "bird"]))


def _pick(mix: tuple[tuple[str, float], ...], rng: np.random.Generator) -> str:
    roll = rng.random()
    cumulative = 0.0
    for name, share in mix:
        cumulative += share
        if roll <= cumulative:
            return name
    return mix[-1][0]


def _nocturnal_offset(rng: np.random.Generator) -> float:
    """Seconds into a 19:00-07:00 night, weighted to the small hours.

    Rodent activity peaks well after the building goes quiet. Centring the
    distribution around 06:30 into the night puts the mode near 01:30 local,
    which is where the architecture's peak-activity example sits.
    """
    hours = float(rng.normal(6.5, 2.1))
    return float(np.clip(hours, 0.05, 11.9) * 3600.0)


def _trajectory(
    camera: DemoCamera, label: str, rng: np.random.Generator
) -> tuple[tuple[float, float], tuple[float, float], list[tuple[float, float]]]:
    """A plausible path across a 1280x720 frame.

    Rodents hug the wall-floor junction along the bottom of the view and, on a
    camera with a known gap, start at it. That is what makes the origins plot
    show a cluster rather than a uniform scatter — the effect §7 relies on.
    """
    if camera.entry_xy and label == "rodent" and rng.random() < 0.62:
        origin = (
            camera.entry_xy[0] + float(rng.normal(0, 11)),
            camera.entry_xy[1] + float(rng.normal(0, 8)),
        )
    elif label == "rodent":
        origin = (float(rng.uniform(10, 1270)), float(rng.uniform(430, 700)))
    else:
        origin = (float(rng.uniform(10, 1270)), float(rng.uniform(120, 700)))

    reach = float(rng.uniform(160, 900))
    angle = float(rng.uniform(0, 2 * math.pi))
    terminus = (
        float(np.clip(origin[0] + reach * math.cos(angle), 2, 1278)),
        float(np.clip(origin[1] + reach * math.sin(angle) * 0.45, 2, 718)),
    )

    steps = int(rng.integers(6, 14))
    polyline = []
    for i in range(steps + 1):
        t = i / steps
        wobble = math.sin(t * math.pi * float(rng.uniform(1.4, 3.2))) * float(rng.uniform(8, 46))
        polyline.append(
            (
                origin[0] + (terminus[0] - origin[0]) * t,
                float(np.clip(origin[1] + (terminus[1] - origin[1]) * t + wobble, 2, 718)),
            )
        )
    return origin, terminus, polyline


def _seed_rejections(
    store: Store, run_id: str, camera: DemoCamera, rng: np.random.Generator
) -> int:
    """Discarded tracks, with the feature vectors that got them discarded.

    Seeded because they are not noise: hard negatives are worth more per example
    than positives when the plausibility classifier is trained, and the console
    has a view whose whole job is showing what the thresholds are cutting.
    """
    total = int(rng.integers(18, 70))
    diagonal = math.hypot(1280.0, 720.0)

    for _ in range(total):
        reason, _, ranges = REJECTIONS[
            int(rng.choice(len(REJECTIONS), p=[r[1] for r in REJECTIONS]))
        ]
        extent_frac = float(rng.uniform(*ranges["extent_frac"]))
        straightness = float(rng.uniform(*ranges["straightness"]))
        n_detections = (
            int(rng.integers(2, 5))
            if reason == "too_few_detections"
            else int(rng.integers(6, 90))
        )
        duration = (
            float(rng.uniform(0.1, 0.5))
            if reason == "too_brief"
            else float(rng.uniform(0.9, 14.0))
        )
        extent_px = extent_frac * diagonal
        path_length = extent_px * float(rng.uniform(2.0, 14.0))

        start = float(rng.uniform(0, NIGHT_SECONDS))
        store.add_rejection(
            run_id,
            camera.id,
            int(rng.integers(1, 900)),
            reason,
            start,
            start + duration,
            n_detections,
            {
                "n_detections": float(n_detections),
                "duration_s": duration,
                "extent_px": extent_px,
                "extent_frac": extent_frac,
                "path_length_px": path_length,
                "straightness": straightness,
                "area_cv": float(
                    rng.uniform(1.25, 2.4) if reason == "area_unstable" else rng.uniform(0.2, 1.1)
                ),
                "mean_area_px": float(rng.uniform(60, 2600)),
                "mean_score": float(rng.uniform(0.3, 1.0)),
            },
        )
    store.conn.commit()
    return total


def _funnel_for(
    camera: DemoCamera,
    night: int,
    n_events: int,
    n_rejections: int,
    rng: np.random.Generator,
) -> FunnelStats:
    """Cascade counters consistent with the architecture's Figure 2.

    Derived from the camera's gate rate rather than invented independently, so
    the ratios the console displays — gate pass rate, tiles per gated frame,
    inference saving — hold together instead of contradicting each other.
    """
    processed = int(NIGHT_SECONDS * FPS)
    gate_rate = camera.gate_rate * float(rng.uniform(0.8, 1.25))

    fouled = camera.fouls_on_night is not None and night >= camera.fouls_on_night
    if fouled:
        # A fouled lens blurs the scene: less passes the gate, which is exactly
        # why it reads as "no pests" unless tamper detection speaks up.
        gate_rate *= 0.28

    gated = int(processed * gate_rate)
    rois = int(gated * rng.uniform(1.15, 1.5))
    tiles = int(gated * rng.uniform(1.25, 1.55))
    detections = int(rois * rng.uniform(0.75, 0.98))
    tracks_created = n_events + n_rejections + int(rng.integers(0, 6))

    stats = FunnelStats(
        frames_read=processed,
        frames_processed=processed,
        frames_gated=gated,
        global_change_frames=int(rng.integers(20, 90)),
        # Dawn IR-cut toggles trip global change on every camera; a fouled lens
        # trips tamper for most of the night.
        tamper_frames=int(rng.integers(9000, 21000)) if fouled else int(rng.integers(0, 40)),
        rois=rois,
        tiles=tiles,
        detections=detections,
        tracks_created=tracks_created,
        tracks_confirmed=n_events + n_rejections,
        events=n_events,
    )
    for reason, share, _ in REJECTIONS:
        stats.rejected[reason] = int(n_rejections * share)
    return stats


def _render_keyframe(
    out_dir: Path,
    camera: DemoCamera,
    event_id: str,
    label: str,
    polyline: list[tuple[float, float]],
    duration: float,
    n_detections: int,
) -> Path:
    """An annotated still, drawn the way the real assembler draws one.

    Reuses the fixture's scene generator so the demo looks like the product
    rather than like a mock: same dim IR floor, same racking, same overlay
    conventions — trajectory, origin dot, terminus dot, box, caption.
    """
    import cv2

    from .synth import _background, _draw_plant, plan_fixture

    plan = plan_fixture(width=1280, height=720)
    # Seeded from the event id so a given event always renders the same frame —
    # a demo whose pictures shuffle on every reseed is hard to talk about.
    rng = np.random.default_rng(int(event_id[:8], 16))

    frame = _background(plan)
    noise = rng.normal(0, 5.0, (plan.height, plan.width, 1)).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _draw_plant(frame, int(rng.integers(0, 200)), plan)

    # The body sits at the track's midpoint, which is where the real assembler
    # picks its keyframe: the track's largest, clearest observation.
    mid = polyline[len(polyline) // 2]
    half_w, half_h = 22, 13
    x1, y1 = int(mid[0] - half_w), int(mid[1] - half_h)
    x2, y2 = int(mid[0] + half_w), int(mid[1] + half_h)
    cv2.ellipse(frame, (int(mid[0]), int(mid[1])), (16, 8), 0, 0, 360, (128, 128, 128), -1)

    points = np.array([(int(x), int(y)) for x, y in polyline], dtype=np.int32)
    cv2.polylines(frame, [points], False, (40, 190, 255), 2)
    cv2.circle(frame, tuple(points[0]), 5, (90, 220, 130), -1)
    cv2.circle(frame, tuple(points[-1]), 5, (60, 60, 235), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 190, 255), 2)

    caption = f"{label} | {duration:.1f}s | {n_detections} det"
    cv2.putText(
        frame, caption, (max(4, min(x1, 1100)), max(16, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 190, 255), 1, cv2.LINE_AA,
    )
    # Unmissable on any screenshot that escapes the console.
    cv2.putText(
        frame, "SYNTHETIC DEMO FRAME - NOT REAL FOOTAGE", (14, 700),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 235), 2, cv2.LINE_AA,
    )

    directory = Path(out_dir) / "keyframes" / camera.id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{event_id}.jpg"
    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])

    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "event_id": event_id,
                "box": [x1, y1, x2, y2],
                "polyline": [[int(x), int(y)] for x, y in polyline],
                "label": label,
                "synthetic": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
