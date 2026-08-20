"""Two follow-up experiments the headline results made necessary.

The first pass showed the model handles near-IR fine but collapses on small
subjects. Two questions decide what that means for us, and neither is answerable
from the first pass:

**1. Is it size, or is it my synthetic canvas?**
The small-subject images place the animal on a flat grey background. A miss
could be caused by the size *or* by the unnatural backdrop. A size sweep on the
identical canvas separates them: if a large subject on the same canvas is
detected and a small one is not, size is the cause. It also locates the
threshold, which is a number we can design against.

**2. Does our own tiling rescue it?**
Stage C of our cascade does not feed whole frames to the detector. It cuts a
640 px tile at native resolution around each motion region — so a rodent that is
3% of a 1280 px frame becomes a much larger fraction of a 640 px tile. If the
model works on the tile, the small-object failure is one our pipeline already
solves, and the assessment changes materially.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.roboflow_eval import testset
from experiments.roboflow_eval.client import RoboflowDetector
from expressvision.config import GateConfig, TileConfig
from expressvision.gate import MotionGate
from expressvision.tiling import Tiler
from expressvision.types import Box

MODEL_ID = "thermal-rodent-detection-xkfep/1"
FRACTIONS = [0.45, 0.30, 0.20, 0.14, 0.10, 0.07, 0.05, 0.035, 0.02]
CANVAS = 1280


def is_rodent(label: str) -> bool:
    return any(w in label.lower() for w in ("rat", "mouse", "mice", "rodent"))


def best_conf(result) -> float:
    hits = [p for p in result.predictions if is_rodent(p.label)]
    return max((p.confidence for p in hits), default=0.0)


def size_sweep(detector, subject: np.ndarray, out_dir: Path, name: str) -> list[dict]:
    """Same canvas, same subject, only the size changes."""
    rows = []
    print(f"\n  size sweep — {name}")
    print(f"  {'frac':>7}{'subject px':>12}{'conf':>9}  result")
    print("  " + "-" * 44)
    for fraction in FRACTIONS:
        img = testset.shrink_subject(subject, fraction, canvas=CANVAS)
        path = out_dir / f"sweep_{name}_{int(fraction * 1000):03d}.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        result = detector.infer(path)
        conf = best_conf(result)
        subject_px = int(CANVAS * fraction)
        rows.append(
            {
                "subject": name,
                "fraction": fraction,
                "subject_px": subject_px,
                "conf": round(conf, 4),
                "hit": conf > 0,
                "latency_ms": round(result.latency_ms, 1),
                "error": result.error,
            }
        )
        mark = "HIT " if conf > 0 else "miss"
        print(f"  {fraction:>7.3f}{subject_px:>12}{conf:>9.3f}  {mark}")
    return rows


def tiling_rescue(detector, subject: np.ndarray, out_dir: Path, name: str) -> list[dict]:
    """Does stage C's native-resolution tile recover a subject the whole frame loses?

    Runs the real gate and tiler over the same synthetic frame our pipeline
    would see, then submits the tile the pipeline would actually have produced.
    """
    rows = []
    print(f"\n  tiling rescue — {name}")
    print(f"  {'frac':>7}{'whole frame':>13}{'640px tile':>13}  verdict")
    print("  " + "-" * 50)

    tiler = Tiler(TileConfig(tile_size=640), CANVAS, int(CANVAS * 9 / 16))

    for fraction in (0.10, 0.07, 0.05, 0.035, 0.02):
        frame = testset.shrink_subject(subject, fraction, canvas=CANVAS)
        h, w = frame.shape[:2]

        whole_path = out_dir / f"rescue_{name}_{int(fraction * 1000):03d}_whole.jpg"
        cv2.imwrite(str(whole_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        whole_conf = best_conf(detector.infer(whole_path))

        # Where shrink_subject places the subject, as a Box, so the tiler sees
        # the same ROI the motion gate would hand it.
        target_w = max(8, int(CANVAS * fraction))
        scale = target_w / subject.shape[1]
        target_h = max(4, int(subject.shape[0] * scale))
        x = min(int(CANVAS * 0.35), w - target_w - 1)
        y = min(int(h * 0.62), h - target_h - 1)
        roi = Box(float(x), float(y), float(x + target_w), float(y + target_h))

        tiles = tiler.plan([roi])
        tile_conf = 0.0
        if tiles:
            patch = tiler.crop(frame, tiles[0])
            tile_path = out_dir / f"rescue_{name}_{int(fraction * 1000):03d}_tile.jpg"
            cv2.imwrite(str(tile_path), patch, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            tile_conf = best_conf(detector.infer(tile_path))

        verdict = (
            "RESCUED" if tile_conf > 0 and whole_conf == 0
            else "both hit" if tile_conf > 0
            else "both miss"
        )
        rows.append(
            {
                "subject": name,
                "fraction": fraction,
                "whole_conf": round(whole_conf, 4),
                "tile_conf": round(tile_conf, 4),
                "verdict": verdict,
            }
        )
        print(
            f"  {fraction:>7.3f}{whole_conf:>13.3f}{tile_conf:>13.3f}  {verdict}"
        )
    return rows


def gate_check(subject: np.ndarray) -> None:
    """Would our motion gate even produce a tile at these sizes?

    Worth knowing: if the gate cannot see the animal either, the detector's
    small-object weakness is moot — nothing would ever be submitted.
    """
    print("\n  our motion gate, same subjects")
    print(f"  {'frac':>7}{'subject px':>12}  gate emits ROI?")
    print("  " + "-" * 40)
    for fraction in (0.10, 0.05, 0.035, 0.02):
        frame = testset.shrink_subject(subject, fraction, canvas=CANVAS)
        h, w = frame.shape[:2]
        gate = MotionGate(GateConfig(warmup_frames=2), w, h)
        empty = np.full_like(frame, 62)

        # Prime the background on empty floor, then show the frame with the animal.
        for _ in range(12):
            gate.process(empty)
        result = gate.process(frame)
        print(
            f"  {fraction:>7.3f}{int(CANVAS * fraction):>12}  "
            f"{'yes, ' + str(len(result.rois)) + ' ROI(s)' if result.rois else 'no'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--out", type=Path, default=Path("out/roboflow_eval/followup"))
    args = parser.parse_args()

    if not os.environ.get("ROBOFLOW_API_KEY"):
        print("ROBOFLOW_API_KEY is not set.")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    detector = RoboflowDetector(args.model_id, confidence=0.05)

    print("fetching subjects...")
    subjects = {}
    for title, species in (("Brown rat", "rat"), ("House mouse", "mouse")):
        img = testset.fetch_wikipedia_image(title)
        if img is not None:
            subjects[species] = img
    if not subjects:
        print("could not fetch any subject image")
        return 1

    sweeps, rescues = [], []
    for name, img in subjects.items():
        sweeps += size_sweep(detector, img, args.out, name)
        rescues += tiling_rescue(detector, img, args.out, name)

    gate_check(next(iter(subjects.values())))

    # Where does it break?
    print("\n" + "=" * 66)
    print("DETECTION THRESHOLD")
    print("=" * 66)
    for name in subjects:
        hits = [r for r in sweeps if r["subject"] == name and r["hit"]]
        if hits:
            smallest = min(hits, key=lambda r: r["fraction"])
            print(
                f"  {name}: smallest detected = {smallest['subject_px']} px wide "
                f"({smallest['fraction']:.1%} of frame), conf {smallest['conf']:.3f}"
            )
        else:
            print(f"  {name}: never detected at any size on this canvas")

    rescued = [r for r in rescues if r["verdict"] == "RESCUED"]
    print(f"\n  tiling rescued {len(rescued)} of {len(rescues)} small-subject cases")

    (args.out / "followup.json").write_text(
        json.dumps({"sweeps": sweeps, "rescues": rescues}, indent=2), encoding="utf-8"
    )
    latencies = [r["latency_ms"] for r in sweeps if r.get("latency_ms")]
    if latencies:
        print(f"\n  latency this run: median {statistics.median(latencies):.0f} ms")
    print(f"  wrote {args.out / 'followup.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
