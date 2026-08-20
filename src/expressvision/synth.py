"""Synthetic night-camera footage for pipeline verification.

Real overnight footage from client sites is weeks away — it needs site access,
a survey and nights of accumulation. This generator exists so the cascade can be
exercised end to end today, and so every guard has a deterministic test case:

* **A rodent-sized target** crossing the frame, at a size and speed matched to a
  real rat (~25 cm, 1.5 m/s) as seen by a 2 MP camera at ~6 m. Each crossing
  should produce exactly one event.
* **A swaying plant** oscillating about a fixed point for the whole clip.
  Should be rejected as ``confined`` — this is the single most common false
  positive in real deployments.
* **A lighting change**, as when someone hits the lights or the IR-cut filter
  toggles at dawn. Should trip the global-change guard, not flood the pipeline.

The fixture carries its own ground truth: :func:`plan_fixture` returns exactly
where the crossings are, so a regression test asserts against the plan rather
than against a magic number that nobody can check.

It is a test fixture, not training data. No model should ever be trained on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REFERENCE_WIDTH = 1280  # geometry below is authored at this width and scales


@dataclass(frozen=True)
class FixturePlan:
    """Ground truth for one generated clip."""

    seconds: int
    fps: int
    width: int
    height: int
    crossings: list[tuple[int, int]] = field(default_factory=list)  # frame ranges
    flash: tuple[int, int] = (0, 0)
    plant_xy: tuple[int, int] = (0, 0)
    seed: int = 7

    @property
    def total_frames(self) -> int:
        return self.seconds * self.fps

    @property
    def scale(self) -> float:
        return self.width / REFERENCE_WIDTH

    def crossing_times_s(self) -> list[tuple[float, float]]:
        return [(a / self.fps, b / self.fps) for a, b in self.crossings]

    def describe(self) -> str:
        times = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in self.crossing_times_s())
        return (
            f"Expected on this fixture:\n"
            f"  events                {len(self.crossings)}"
            f"    (one per rodent crossing, and nothing else)\n"
            f"  crossings at          {times}\n"
            f"  rejected: confined    >0   (the swaying plant)\n"
            f"  global_change_frames  >0   (lighting change at "
            f"{self.flash[0] / self.fps:.1f}s)\n\n"
            f"Note the gate pass rate will be high (~70%), which is correct here and\n"
            f"not a defect: the plant sways in every single frame, so there is genuine\n"
            f"motion almost continuously. Real overnight footage is mostly static and\n"
            f"gates in the single digits. This fixture is deliberately the harder case."
        )


def plan_fixture(
    seconds: int = 60,
    fps: int = 15,
    width: int = 1280,
    height: int = 720,
    seed: int = 7,
) -> FixturePlan:
    """Lay out crossings and the lighting change for a clip of any length.

    The first crossing starts well after the gate's warmup so it is never
    swallowed by an unsettled background model.
    """
    first_start_s = 8.0
    crossing_s = 3.0
    gap_s = 8.0

    crossings: list[tuple[int, int]] = []
    t = first_start_s
    while t + crossing_s <= seconds - 1.0:
        crossings.append((int(t * fps), int((t + crossing_s) * fps)))
        t += crossing_s + gap_s

    # Put the lighting change in a gap, never on top of a crossing.
    flash_start = (
        crossings[0][1] + int(4.0 * fps) if crossings else int(seconds * fps * 0.5)
    )
    flash = (flash_start, flash_start + int(2.0 * fps))

    return FixturePlan(
        seconds=seconds,
        fps=fps,
        width=width,
        height=height,
        crossings=crossings,
        flash=flash,
        plant_xy=(int(width * 0.85), int(height * 0.76)),
        seed=seed,
    )


def make_test_video(path: str | Path, plan: FixturePlan | None = None, **kwargs) -> FixturePlan:
    """Render a clip. Returns the plan so callers know the ground truth."""
    plan = plan or plan_fixture(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(plan.seed)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), plan.fps, (plan.width, plan.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out}")

    base = _background(plan)
    try:
        for i in range(plan.total_frames):
            frame = base.copy()

            # Sensor noise — without it MOG2 behaves unrealistically well.
            noise = rng.normal(0, 4.5, (plan.height, plan.width, 1)).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            _draw_plant(frame, i, plan)
            for start, end in plan.crossings:
                if start <= i < end:
                    _draw_rodent(frame, i, start, end, plan)

            if plan.flash[0] <= i < plan.flash[1]:
                frame = np.clip(frame.astype(np.int16) + 85, 0, 255).astype(np.uint8)

            writer.write(frame)
    finally:
        writer.release()

    return plan


def _background(plan: FixturePlan) -> np.ndarray:
    """A dim IR-lit floor with racking along the back wall."""
    s = plan.scale
    w, h = plan.width, plan.height
    base = np.full((h, w, 3), 38, dtype=np.uint8)

    wall_y = int(210 * s)
    cv2.rectangle(base, (0, 0), (w, wall_y), (28, 28, 28), -1)
    for x in range(int(90 * s), w - int(60 * s), int(240 * s)):
        cv2.rectangle(
            base, (x, int(96 * s)), (x + int(150 * s), wall_y), (52, 52, 52), -1
        )
        cv2.rectangle(
            base, (x, int(96 * s)), (x + int(150 * s), wall_y), (66, 66, 66), max(1, int(2 * s))
        )
    cv2.line(base, (0, wall_y + 2), (w, wall_y + 2), (70, 70, 70), max(1, int(2 * s)))
    return base


def _draw_rodent(frame: np.ndarray, i: int, start: int, end: int, plan: FixturePlan) -> None:
    """A 32x16 px body (at reference width) crossing left to right ~19 px/frame.

    That is a 25 cm animal at 1.5 m/s on a 2 MP camera at roughly 6 m — right at
    the size the architecture's range table calls detectable but not identifiable.
    """
    s = plan.scale
    progress = (i - start) / max(1, end - start)
    x = int(-40 * s + progress * (plan.width + 80 * s))
    y = int((430 + 55 * math.sin(progress * math.pi * 2.4)) * s)

    if not (-40 * s < x < plan.width + 40 * s):
        return

    body = (max(2, int(16 * s)), max(1, int(8 * s)))
    cv2.ellipse(frame, (x, y), body, 0, 0, 360, (128, 128, 128), -1)
    cv2.ellipse(
        frame,
        (x + int(12 * s), y - int(2 * s)),
        (max(1, int(5 * s)), max(1, int(4 * s))),
        0, 0, 360, (140, 140, 140), -1,
    )
    tail_x = x - int(26 * s) + int(4 * s * math.sin(i * 0.7))
    cv2.line(
        frame, (x - int(14 * s), y), (tail_x, y + int(5 * s)),
        (104, 104, 104), max(1, int(2 * s)),
    )


def _draw_plant(frame: np.ndarray, i: int, plan: FixturePlan) -> None:
    """Foliage swaying in a doorway draft — moves constantly, goes nowhere."""
    s = plan.scale
    cx = plan.plant_xy[0] + int(17 * s * math.sin(i * 0.19))
    cy = plan.plant_xy[1] + int(8 * s * math.cos(i * 0.23))
    for k in range(5):
        angle = k * 1.26 + 0.32 * math.sin(i * 0.19 + k)
        tip = (cx + int(46 * s * math.cos(angle)), cy - int(46 * s * math.sin(angle)))
        cv2.line(frame, (cx, cy), tip, (74, 82, 74), max(1, int(4 * s)))
    cv2.circle(frame, (cx, cy), max(2, int(9 * s)), (68, 76, 68), -1)
