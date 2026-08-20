"""Track validation — stage E, the false-positive firewall.

Nothing reaches a person on the strength of a single frame. A track that has
been seen enough times, has actually gone somewhere, and moved in a way a living
thing plausibly moves becomes an event; everything else is discarded silently.

On representative footage this stage removes ~93% of tracks. Without it a client
receives several hundred alerts a night and stops reading them inside a week,
which is the most common way systems like this fail in the field — not by
missing pests, but by being muted.

The heuristics here are the v1 hand-built version. Phase 2 replaces the scoring
with a small gradient-boosted classifier trained on operator verdicts; the
features it will use are exactly the ones computed below, so the swap is local.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ValidatorConfig
from .types import Track


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


class TrackValidator:
    def __init__(self, cfg: ValidatorConfig, frame_width: int, frame_height: int) -> None:
        self.cfg = cfg
        self.diagonal = math.hypot(frame_width, frame_height)

    def judge(self, track: Track) -> Verdict:
        cfg = self.cfg

        if len(track.points) < cfg.min_detections:
            return Verdict(False, "too_few_detections")

        if track.frame_span < cfg.min_frames_span:
            return Verdict(False, "too_brief")

        extent = track.extent()
        if extent < cfg.min_extent_frac * self.diagonal:
            # Stayed put. Vegetation swaying in a doorway draft, a flag, a
            # reflection flickering on a wet floor, a fan blade.
            return Verdict(False, "confined")

        if (
            extent < cfg.straightness_below_extent_frac * self.diagonal
            and track.straightness() < cfg.min_straightness
        ):
            # Covered a moderate area, but only by oscillating within it.
            # Not applied above that extent: an animal working along a wall and
            # doubling back is a real event with poor straightness.
            return Verdict(False, "oscillating")

        if track.area_cv() > cfg.max_area_cv:
            # Apparent size swinging wildly: rain, an insect crossing close to
            # the lens, or the detector latching onto a growing shadow.
            return Verdict(False, "unstable_size")

        return Verdict(True, "ok")

    def features(self, track: Track) -> dict[str, float]:
        """The feature vector for this track.

        Persisted alongside every event and every rejection so that Phase 2 has a
        labelled training set the moment operators start marking verdicts —
        rather than needing a fresh data collection round.
        """
        duration = max(1e-6, track.duration_s)
        displacement = track.net_displacement()
        extent = track.extent()
        speeds = [
            math.dist((a.cx, a.cy), (b.cx, b.cy)) / max(1e-6, b.t_s - a.t_s)
            for a, b in zip(track.points, track.points[1:])
            if b.t_s > a.t_s
        ]
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        peak_speed = max(speeds) if speeds else 0.0

        return {
            "n_detections": float(len(track.points)),
            "frame_span": float(track.frame_span),
            "duration_s": track.duration_s,
            "displacement_px": displacement,
            "displacement_frac": displacement / self.diagonal,
            "extent_px": extent,
            "extent_frac": extent / self.diagonal,
            "path_length_px": track.path_length(),
            "straightness": track.straightness(),
            "area_cv": track.area_cv(),
            "mean_area_px": (
                sum(p.area for p in track.points) / len(track.points)
                if track.points
                else 0.0
            ),
            "mean_speed_px_s": mean_speed,
            "peak_speed_px_s": peak_speed,
            "speed_ratio": peak_speed / mean_speed if mean_speed > 1e-6 else 0.0,
            "displacement_per_s": displacement / duration,
            "mean_score": (
                sum(p.score for p in track.points) / len(track.points)
                if track.points
                else 0.0
            ),
        }
