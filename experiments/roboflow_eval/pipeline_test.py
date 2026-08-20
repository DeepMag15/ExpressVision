"""Plug the Roboflow model into the existing cascade, without changing it.

Answers a narrower question than the image evaluation: what happens when this
detector sits at stage C of our real pipeline? The interesting output is not
accuracy — it is throughput, because every tile becomes an HTTPS round trip.

Deliberately capped to a small number of frames. A full camera-night through
this path would take days.

    $env:ROBOFLOW_API_KEY = "..."
    uv run python -m experiments.roboflow_eval.pipeline_test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.roboflow_eval.adapter import RoboflowTileDetector
from expressvision.config import PipelineConfig
from expressvision.pipeline import CameraPipeline

MODEL_ID = "thermal-rodent-detection-xkfep/1"
TILES_PER_CAMERA_NIGHT = 25_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/fixture.mp4"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--confidence", type=float, default=0.10)
    args = parser.parse_args()

    if not os.environ.get("ROBOFLOW_API_KEY"):
        print("ROBOFLOW_API_KEY is not set.")
        return 2
    if not args.source.exists():
        print(f"No such clip: {args.source}  (run: uv run exv make-fixture)")
        return 1

    cfg = PipelineConfig.for_source(str(args.source), "cam-roboflow")
    cfg.out_dir = Path("out/roboflow_pipeline")
    cfg.event.write_clips = False
    cfg.event.write_keyframes = False

    detector = RoboflowTileDetector(
        args.model_id, tile_size=cfg.tile.tile_size, confidence=args.confidence
    )
    print(f"model: {detector.name}")
    print(f"clip:  {args.source}  (first {args.max_frames} frames)\n")

    detector.warmup(1)
    started = time.perf_counter()
    pipeline = CameraPipeline(cfg, cfg.cameras[0], store=None, detector=detector)
    stats = pipeline.run(max_frames=args.max_frames)
    wall = time.perf_counter() - started

    mean_latency = (
        detector.total_latency_ms / detector.calls if detector.calls else 0.0
    )
    fps = stats.frames_processed / wall if wall else 0.0

    print("\n" + "=" * 70)
    print("PIPELINE INTEGRATION")
    print("=" * 70)
    print(f"frames processed      {stats.frames_processed}")
    print(f"passed motion gate    {stats.frames_gated}")
    print(f"tiles inferred        {stats.tiles}")
    print(f"API calls             {detector.calls}")
    print(f"detections            {stats.detections}")
    print(f"tracks confirmed      {stats.tracks_confirmed}")
    print(f"events                {stats.events}")
    if stats.rejected:
        for reason, count in sorted(stats.rejected.items(), key=lambda kv: -kv[1]):
            print(f"  rejected: {reason:<22} {count}")

    print(f"\nwall clock            {wall:.1f} s")
    print(f"throughput            {fps:.2f} fps")
    print(f"mean call latency     {mean_latency:.0f} ms")

    if detector.calls:
        night_hours = mean_latency * TILES_PER_CAMERA_NIGHT / 1000 / 3600
        print(
            f"\nAt ~{TILES_PER_CAMERA_NIGHT:,} tiles per camera-night, this detector "
            f"alone costs\n{night_hours:.1f} hours of round-trip latency per camera, "
            f"per night — before compute,\nand it scales linearly with cameras."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
