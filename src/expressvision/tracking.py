"""Multi-object tracking — stage D of the cascade.

IoU association with a constant-velocity prediction and Hungarian assignment.
The design choice that matters for this domain is that association happens
against *predicted* positions and tolerates a low IoU: a rat crossing behind a
pallet leg vanishes for several frames and reappears displaced, and a tracker
that drops it there turns one visit into three.

This is deliberately the same shape as ByteTrack's association step so that
swapping in the real thing at Phase 2 — once detections carry confidences worth
splitting on — is a local change.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackConfig
from .types import Box, Detection, Track, TrackPoint


class Tracker:
    def __init__(self, cfg: TrackConfig, camera_id: str) -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self.tracks: list[Track] = []
        self._next_id = 1
        self.created = 0

    def update(
        self, detections: list[Detection], frame_idx: int, t_s: float
    ) -> list[Track]:
        """Advance every track by one frame.

        Returns the tracks that ended on this frame — those are what the
        validator judges and the assembler turns into events.
        """
        for track in self.tracks:
            track.age += 1

        matches, unmatched_dets, _ = self._associate(detections)

        for track_i, det_i in matches:
            self._apply(self.tracks[track_i], detections[det_i], frame_idx, t_s)

        for det_i in unmatched_dets:
            self._spawn(detections[det_i], frame_idx, t_s)

        finished = [t for t in self.tracks if t.age > self.cfg.max_age]
        self.tracks = [t for t in self.tracks if t.age <= self.cfg.max_age]
        return [t for t in finished if t.confirmed]

    def flush(self) -> list[Track]:
        """Close every open track — call at end of stream."""
        finished = [t for t in self.tracks if t.confirmed]
        self.tracks = []
        return finished

    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        score = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for ti, track in enumerate(self.tracks):
            predicted = self._predict(track)
            for di, det in enumerate(detections):
                score[ti, di] = self._affinity(predicted, det.box)

        track_idx, det_idx = linear_sum_assignment(-score)

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for ti, di in zip(track_idx, det_idx):
            if score[ti, di] < self.cfg.iou_threshold:
                continue
            matches.append((int(ti), int(di)))
            matched_tracks.add(int(ti))
            matched_dets.add(int(di))

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        return matches, unmatched_dets, unmatched_tracks

    def _affinity(self, predicted: Box, candidate: Box) -> float:
        """How strongly a predicted track position matches a detection.

        Overlap first. When there is none — which for a small fast animal is the
        normal case, not an edge case — fall back to how close the centres are,
        measured against how far the object could plausibly have travelled.

        Without this, a rat whose per-frame displacement exceeds its own body
        length never forms a track at all, and the failure is silent: the gate
        fires, tiles are inferred, detections are produced, and nothing reaches
        the validator.
        """
        overlap = predicted.iou(candidate)
        if overlap >= self.cfg.iou_threshold:
            return overlap

        reach = self.cfg.max_travel_factor * max(
            predicted.w, predicted.h, candidate.w, candidate.h
        )
        if reach <= 0:
            return overlap

        distance = math.dist((predicted.cx, predicted.cy), (candidate.cx, candidate.cy))
        if distance >= reach:
            return overlap

        proximity = (1.0 - distance / reach) * self.cfg.proximity_weight
        return max(overlap, proximity)

    @staticmethod
    def _predict(track: Track) -> Box:
        """Where the track's box should be now, given its velocity.

        Extrapolation is capped at the max_age horizon so a long gap does not
        fling the predicted box off-frame and lose an otherwise good match.
        """
        steps = min(track.age, 10)
        dx, dy = track.vx * steps, track.vy * steps
        return Box(
            track.box.x1 + dx, track.box.y1 + dy, track.box.x2 + dx, track.box.y2 + dy
        )

    def _apply(self, track: Track, det: Detection, frame_idx: int, t_s: float) -> None:
        prev_cx, prev_cy = track.box.cx, track.box.cy
        gap = max(1, track.age)

        track.box = det.box
        track.label = det.label
        track.hits += 1
        track.age = 0
        track.end_frame = frame_idx
        track.end_t_s = t_s

        # Velocity as an EMA of per-frame centroid delta, so one noisy box does
        # not throw the prediction off.
        inst_vx = (det.box.cx - prev_cx) / gap
        inst_vy = (det.box.cy - prev_cy) / gap
        track.vx = 0.5 * track.vx + 0.5 * inst_vx
        track.vy = 0.5 * track.vy + 0.5 * inst_vy

        track.points.append(
            TrackPoint(
                frame_idx=frame_idx,
                t_s=t_s,
                cx=det.box.cx,
                cy=det.box.cy,
                area=det.box.area,
                score=det.score,
                box=det.box,
            )
        )

        if not track.confirmed and track.hits >= self.cfg.min_hits:
            track.confirmed = True

    def _spawn(self, det: Detection, frame_idx: int, t_s: float) -> None:
        track = Track(
            id=self._next_id,
            camera_id=self.camera_id,
            box=det.box,
            label=det.label,
            start_frame=frame_idx,
            end_frame=frame_idx,
            start_t_s=t_s,
            end_t_s=t_s,
            hits=1,
        )
        track.points.append(
            TrackPoint(
                frame_idx=frame_idx,
                t_s=t_s,
                cx=det.box.cx,
                cy=det.box.cy,
                area=det.box.area,
                score=det.score,
                box=det.box,
            )
        )
        self._next_id += 1
        self.created += 1
        self.tracks.append(track)
