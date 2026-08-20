"""Throughput benchmark — how many CCTV streams can one GPU actually carry?

Measures the pipeline as it really runs, then converts that into the number the
capacity plan needs: streams per GPU.

Two properties make the answer trustworthy:

* **Stages are timed separately.** Decode, gate, tile, detect and track have very
  different costs, and only detection moves to the GPU. If detection stops being
  the bottleneck, adding GPUs buys nothing — and the per-stage split is what
  reveals that.
* **CUDA work is synchronised before every timestamp.** CUDA calls are
  asynchronous; timing without synchronisation measures how fast Python enqueues
  work, not how fast the GPU finishes it, and produces numbers that look
  wonderful and are worthless.

The stream estimate is deliberately conservative and states its assumptions,
because it is the number that decides hardware spend.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig
from .detect import Detector
from .gate import MotionGate
from .gpu import GpuSnapshot, reset_vram_peak, sample_gpu, synchronize, torch_vram_mb
from .sources import open_source
from .tiling import Tiler
from .tracking import Tracker
from .types import FunnelStats

# A camera-night: 12 hours of dark at the frame rate the pipeline processes.
NIGHT_HOURS = 12.0


@dataclass
class StageTimings:
    """Milliseconds per call, per stage."""

    decode: list[float] = field(default_factory=list)
    gate: list[float] = field(default_factory=list)
    tile: list[float] = field(default_factory=list)
    detect: list[float] = field(default_factory=list)
    track: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name in ("decode", "gate", "tile", "detect", "track"):
            samples = getattr(self, name)
            if not samples:
                continue
            ordered = sorted(samples)
            out[name] = {
                "calls": float(len(samples)),
                "mean_ms": statistics.fmean(samples),
                "median_ms": statistics.median(ordered),
                "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
                "max_ms": ordered[-1],
                "total_s": sum(samples) / 1000.0,
            }
        return out


@dataclass
class BenchmarkResult:
    label: str
    image_size: int | str
    tile_size: int
    device: str
    dtype: str
    frames: int
    wall_s: float
    fps: float
    tiles: int
    tiles_per_s: float
    gate_pass_rate: float
    tiles_per_gated_frame: float
    detect_ms_per_tile: float
    stages: dict[str, dict[str, float]]
    gpu_util_mean: float | None
    gpu_util_peak: float | None
    vram_peak_mb: float | None
    torch_vram_mb: float | None
    stats: FunnelStats
    runtime_info: dict[str, str] = field(default_factory=dict)

    def streams_supported(self, stream_fps: float = 15.0, headroom: float = 0.7) -> float:
        """How many live streams one instance of this pipeline could carry.

        ``headroom`` reserves capacity for bursts: a benchmark clip has an
        average motion load, but real sites have busy minutes — a dawn shift
        change, rain, a forklift — and a pipeline sized to its average will drop
        frames exactly when something is happening.
        """
        if self.fps <= 0:
            return 0.0
        return (self.fps / stream_fps) * headroom

    def camera_night_hours(self, stream_fps: float = 15.0) -> float:
        """Hours to process one 12-hour camera-night of recorded footage."""
        if self.fps <= 0:
            return float("inf")
        return (NIGHT_HOURS * 3600.0 * stream_fps) / self.fps / 3600.0


class _Sampler:
    """Polls GPU telemetry at a fixed interval without a background thread.

    Called from the frame loop: sampling is cheap via NVML but not free, so it
    is rate-limited rather than run per frame.
    """

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self._last = 0.0
        self.samples: list[GpuSnapshot] = []

    def maybe_sample(self) -> None:
        now = time.monotonic()
        if now - self._last < self.interval_s:
            return
        self._last = now
        snap = sample_gpu()
        if snap.utilisation_pct is not None or snap.memory_used_mb is not None:
            self.samples.append(snap)

    def summarise(self) -> tuple[float | None, float | None, float | None]:
        utils = [s.utilisation_pct for s in self.samples if s.utilisation_pct is not None]
        mems = [s.memory_used_mb for s in self.samples if s.memory_used_mb is not None]
        return (
            statistics.fmean(utils) if utils else None,
            max(utils) if utils else None,
            max(mems) if mems else None,
        )


def run_benchmark(
    source: str,
    detector: Detector,
    cfg: PipelineConfig | None = None,
    max_frames: int | None = None,
    label: str = "run",
    warmup_frames: int = 20,
    sample_gpu_telemetry: bool = True,
) -> BenchmarkResult:
    """Time the cascade stage by stage on one source.

    Tracking is exercised but events are not assembled: clip writing is disk-
    bound and would distort the compute measurement this exists to produce.
    """
    cfg = cfg or PipelineConfig()
    timings = StageTimings()
    stats = FunnelStats()

    if hasattr(detector, "warmup"):
        detector.warmup(3)
    reset_vram_peak()

    # Only report GPU telemetry when inference is actually on the GPU. On a CPU
    # run NVML still returns the card's idle utilisation and memory, and a
    # benchmark that prints "141 MB VRAM" for a CPU run invites the reader to
    # believe the GPU was involved.
    info_early = detector.runtime_info() if hasattr(detector, "runtime_info") else {}
    on_cuda = "cuda" in info_early.get("actual_device", "")
    sampler = _Sampler() if (sample_gpu_telemetry and on_cuda) else None

    src = open_source(source, frame_stride=1)
    frames = src.frames()

    gate: MotionGate | None = None
    tiler: Tiler | None = None
    tracker: Tracker | None = None

    counted_frames = 0
    started: float | None = None

    decode_start = time.perf_counter()
    for frame_idx, _t_s, frame in frames:
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        if gate is None:
            h, w = frame.shape[:2]
            gate = MotionGate(cfg.gate, w, h)
            tiler = Tiler(cfg.tile, w, h)
            tracker = Tracker(cfg.track, "bench")

        # Warmup frames prime the background model and CUDA kernels; timing them
        # would report a pipeline that is still settling.
        warm = stats.frames_read < warmup_frames
        stats.frames_read += 1

        if not warm and started is None:
            synchronize()
            started = time.perf_counter()

        if not warm:
            timings.decode.append(decode_ms)

        t0 = time.perf_counter()
        result = gate.process(frame)
        if not warm:
            timings.gate.append((time.perf_counter() - t0) * 1000.0)

        tiles = []
        if result.passed:
            t0 = time.perf_counter()
            tiles = tiler.plan(result.rois)
            if not warm:
                timings.tile.append((time.perf_counter() - t0) * 1000.0)

        detections = []
        if tiles:
            synchronize()
            t0 = time.perf_counter()
            detections = detector.detect(frame, tiles, result.rois, frame_idx)
            synchronize()
            if not warm:
                timings.detect.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        tracker.update(detections, frame_idx, frame_idx / 15.0)
        if not warm:
            timings.track.append((time.perf_counter() - t0) * 1000.0)

        if not warm:
            counted_frames += 1
            stats.frames_processed += 1
            if result.passed:
                stats.frames_gated += 1
                stats.rois += len(result.rois)
            stats.tiles += len(tiles)
            stats.detections += len(detections)
            if sampler:
                sampler.maybe_sample()

        if max_frames and stats.frames_read >= max_frames:
            break

        decode_start = time.perf_counter()

    synchronize()
    wall_s = (time.perf_counter() - started) if started else 0.0

    util_mean, util_peak, vram_peak = (
        sampler.summarise() if sampler else (None, None, None)
    )
    _, torch_peak = torch_vram_mb()

    detect_total_ms = sum(timings.detect)
    info = detector.runtime_info() if hasattr(detector, "runtime_info") else {}

    return BenchmarkResult(
        label=label,
        image_size=info.get("image_size", "n/a"),
        tile_size=cfg.tile.tile_size,
        device=info.get("actual_device", "cpu"),
        dtype=info.get("actual_dtype", "n/a"),
        frames=counted_frames,
        wall_s=wall_s,
        fps=counted_frames / wall_s if wall_s > 0 else 0.0,
        tiles=stats.tiles,
        tiles_per_s=stats.tiles / wall_s if wall_s > 0 else 0.0,
        gate_pass_rate=(
            stats.frames_gated / stats.frames_processed if stats.frames_processed else 0.0
        ),
        tiles_per_gated_frame=(
            stats.tiles / stats.frames_gated if stats.frames_gated else 0.0
        ),
        detect_ms_per_tile=detect_total_ms / stats.tiles if stats.tiles else 0.0,
        stages=timings.summary(),
        gpu_util_mean=util_mean,
        gpu_util_peak=util_peak,
        vram_peak_mb=vram_peak,
        torch_vram_mb=torch_peak,
        stats=stats,
        runtime_info=info,
    )


def bottleneck(result: BenchmarkResult) -> tuple[str, float]:
    """Which stage dominates, and what share of total time it takes.

    The decision this informs: if detection is not dominant, a faster or second
    GPU changes nothing, and the work belongs in decode or gating instead.
    """
    totals = {name: data["total_s"] for name, data in result.stages.items()}
    if not totals:
        return "unknown", 0.0
    worst = max(totals, key=lambda k: totals[k])
    grand = sum(totals.values()) or 1.0
    return worst, totals[worst] / grand


def results_to_dict(results: list[BenchmarkResult], env: dict) -> dict:
    """Serialisable record — commit these so runs are comparable over time."""
    return {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": env,
        "runs": [
            {
                "label": r.label,
                "image_size": r.image_size,
                "tile_size": r.tile_size,
                "device": r.device,
                "dtype": r.dtype,
                "frames": r.frames,
                "wall_s": round(r.wall_s, 3),
                "fps": round(r.fps, 2),
                "tiles": r.tiles,
                "tiles_per_s": round(r.tiles_per_s, 2),
                "detect_ms_per_tile": round(r.detect_ms_per_tile, 2),
                "gate_pass_rate": round(r.gate_pass_rate, 4),
                "tiles_per_gated_frame": round(r.tiles_per_gated_frame, 3),
                "gpu_util_mean": r.gpu_util_mean,
                "gpu_util_peak": r.gpu_util_peak,
                "vram_peak_mb": r.vram_peak_mb,
                "torch_vram_peak_mb": r.torch_vram_mb,
                "streams_at_15fps": round(r.streams_supported(), 2),
                "hours_per_camera_night": round(r.camera_night_hours(), 2),
                "bottleneck": bottleneck(r)[0],
                "stages": r.stages,
                "runtime_info": r.runtime_info,
            }
            for r in results
        ],
    }


def default_bench_clip(path: Path, seconds: int = 30) -> Path:
    """Generate a benchmark clip if none was supplied.

    Same synthetic fixture as the regression tests, so benchmark numbers and
    correctness numbers describe the same input.
    """
    from .synth import make_test_video, plan_fixture

    if path.exists():
        return path
    make_test_video(path, plan_fixture(seconds=seconds))
    return path
