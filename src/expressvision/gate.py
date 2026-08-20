"""Motion gate — stage B of the cascade.

Runs on a downscaled greyscale copy of every frame and answers one question:
*is there anywhere in this frame worth spending GPU on?* On representative
warehouse footage it discards ~96% of frames, which is what makes native-
resolution tiling affordable at stage C.

Two guards matter as much as the subtraction itself, because both produce
thousands of spurious candidates in the field:

* **Global change** — someone hits the lights, the IR-cut filter toggles at
  dawn, or a forklift nudges the camera. Every pixel changes at once. Without
  the guard this looks like maximal motion; with it the model resets and emits
  nothing until it settles.
* **Tamper / defocus** — a spider builds a web across the lens, or a housekeeper
  sprays it. Image detail collapses. This kills more outdoor analytics than any
  other single cause, and it must surface as a maintenance alert rather than as
  silence that reads like "no pests".
"""

from __future__ import annotations

import statistics
from collections import deque

import cv2
import numpy as np

from .config import GateConfig
from .types import Box, GateResult


class MotionGate:
    def __init__(self, cfg: GateConfig, frame_width: int, frame_height: int) -> None:
        self.cfg = cfg
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Downscale factor from full resolution into gate work space.
        self.scale = min(1.0, cfg.work_width / max(1, frame_width))
        self.work_w = max(16, round(frame_width * self.scale))
        self.work_h = max(16, round(frame_height * self.scale))
        self.work_area = float(self.work_w * self.work_h)

        self._bg = self._new_subtractor()
        self._open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel)
        )
        self._close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel)
        )

        self._frames_seen = 0
        self._suppress_until = cfg.warmup_frames

        # Tamper detection state.
        self._sharpness: deque[float] = deque(maxlen=300)
        self._low_sharp_run = 0
        self._tampered = False

    def _new_subtractor(self) -> cv2.BackgroundSubtractorMOG2:
        return cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=True,
        )

    def reset(self) -> None:
        """Drop the background model and re-enter warmup."""
        self._bg = self._new_subtractor()
        self._suppress_until = self._frames_seen + self.cfg.global_change_reset_frames

    def process(self, frame_bgr: np.ndarray) -> GateResult:
        self._frames_seen += 1

        small = cv2.resize(
            frame_bgr, (self.work_w, self.work_h), interpolation=cv2.INTER_AREA
        )
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        tampered = self._check_tamper(grey)

        mask = self._bg.apply(grey, learningRate=self.cfg.learning_rate)
        # MOG2 marks shadows as 127; only hard foreground counts as motion.
        fg = cv2.inRange(mask, 255, 255)
        changed_frac = float(np.count_nonzero(fg)) / self.work_area

        if changed_frac > self.cfg.global_change_frac:
            self.reset()
            return GateResult(
                changed_frac=changed_frac, global_change=True, tampered=tampered
            )

        if self._frames_seen <= self._suppress_until:
            return GateResult(
                changed_frac=changed_frac, warming_up=True, tampered=tampered
            )

        if tampered:
            # A fouled lens produces meaningless candidates; stop emitting until
            # someone cleans it, but keep reporting the condition every frame.
            return GateResult(changed_frac=changed_frac, tampered=True)

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._open_k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._close_k)

        rois = self._contours_to_rois(fg)
        return GateResult(rois=rois, changed_frac=changed_frac, tampered=False)

    def _check_tamper(self, grey: np.ndarray) -> bool:
        cfg = self.cfg
        if not cfg.tamper_enabled:
            return False
        if self._frames_seen % cfg.tamper_sample_every != 0:
            return self._tampered

        sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        self._sharpness.append(sharpness)

        # Need a baseline before the comparison means anything.
        if len(self._sharpness) < 30:
            return False

        baseline = statistics.median(self._sharpness)
        if baseline > 0 and sharpness < baseline * cfg.tamper_ratio:
            self._low_sharp_run += cfg.tamper_sample_every
        else:
            self._low_sharp_run = 0

        self._tampered = self._low_sharp_run >= cfg.tamper_frames
        return self._tampered

    def _contours_to_rois(self, fg: np.ndarray) -> list[Box]:
        cfg = self.cfg
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        min_area = cfg.min_area_frac * self.work_area
        max_area = cfg.max_area_frac * self.work_area

        boxes: list[Box] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = float(w * h)
            if area < min_area or area > max_area:
                continue
            # Reject slivers: a 40:1 bounding box is a lighting seam or a cable
            # shadow, not an animal.
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect > 12.0:
                continue
            boxes.append(Box(float(x), float(y), float(x + w), float(y + h)))

        boxes = merge_nearby(boxes, cfg.merge_distance)

        # Scale work-space boxes back into full-resolution coordinates.
        inv = 1.0 / self.scale if self.scale > 0 else 1.0
        return [
            b.scaled(inv).clipped(self.frame_width, self.frame_height) for b in boxes
        ]


def merge_nearby(boxes: list[Box], max_gap: float) -> list[Box]:
    """Union boxes whose edges are within ``max_gap`` of each other.

    One animal routinely fragments into several contours — a rat's body and tail
    separated by a dark patch of floor. Merging first means the tiler cuts one
    tile instead of three, and the tracker sees one object instead of three.
    """
    if len(boxes) <= 1:
        return list(boxes)

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        while merged:
            head = merged.pop()
            keep: list[Box] = []
            for other in merged:
                if head.gap_to(other) <= max_gap:
                    head = head.merged(other)
                    changed = True
                else:
                    keep.append(other)
            merged = keep
            out.append(head)
        merged = out
    return merged
