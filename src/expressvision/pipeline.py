"""The collector pipeline — stages A through F wired together.

One instance per camera. Reads a source, runs the gated cascade, and writes
events, rejections and funnel counters to the store.

The funnel counters are not instrumentation for its own sake: the architecture's
sizing assumptions (a ~96% gate discard rate, ~1.5 tiles per gated frame) decide
how many cameras fit on one edge node, and they need checking against the
client's real footage rather than being assumed.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .assembler import Event, EventAssembler
from .config import CameraConfig, PipelineConfig
from .detect import Detector, MotionPassthrough
from .gate import MotionGate
from .ringbuffer import RingBuffer
from .sources import open_source
from .store import Store
from .tiling import Tiler
from .tracking import Tracker
from .types import FunnelStats
from .validator import TrackValidator

ProgressFn = Callable[[int, FunnelStats], None]


class CameraPipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        camera: CameraConfig,
        store: Store | None = None,
        detector: Detector | None = None,
        run_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.camera = camera
        self.store = store
        self.detector = detector or MotionPassthrough()
        self.run_id = run_id or uuid.uuid4().hex[:16]
        self.stats = FunnelStats()
        self.events: list[Event] = []
        # Effective frame rate after frame_stride, known once the first frame
        # reveals the source's real properties. Recorded on the run so every
        # rate the dashboard computes has a denominator.
        self.fps: float | None = None

        # Built on the first frame, once real dimensions are known.
        self.gate: MotionGate | None = None
        self.tiler: Tiler | None = None
        self.validator: TrackValidator | None = None
        self.buffer: RingBuffer | None = None
        self.assembler: EventAssembler | None = None
        self.tracker = Tracker(cfg.track, camera.id)

    def run(
        self,
        max_frames: int | None = None,
        on_progress: ProgressFn | None = None,
        progress_every: int = 200,
    ) -> FunnelStats:
        source = open_source(self.camera.source, self.camera.frame_stride)

        if self.store:
            self.store.start_run(
                self.run_id,
                self.camera.id,
                self.camera.source,
                self.cfg.model_dump_json(),
            )

        t_s = 0.0
        for frame_idx, t_s, frame in source.frames():
            self.stats.frames_read += 1

            if self.gate is None:
                self._init_stages(source, frame)

            self._process_frame(frame_idx, t_s, frame)

            if on_progress and self.stats.frames_processed % progress_every == 0:
                on_progress(frame_idx, self.stats)

            if max_frames and self.stats.frames_processed >= max_frames:
                break

        self._finish(t_s)
        return self.stats

    def _init_stages(self, source, frame) -> None:
        h, w = frame.shape[:2]
        info = source.info
        fps = (info.fps if info else 15.0) / max(1, self.camera.frame_stride)
        self.fps = fps

        self.gate = MotionGate(self.cfg.gate, w, h)
        self.tiler = Tiler(self.cfg.tile, w, h)
        self.validator = TrackValidator(self.cfg.validator, w, h)
        event_cfg = self.cfg.event
        self.buffer = RingBuffer(
            seconds=(
                event_cfg.pre_roll_s
                + event_cfg.max_clip_s
                + event_cfg.post_roll_s
                + 2.0  # headroom for a variable frame rate
            ),
            fps=fps,
            jpeg_quality=event_cfg.jpeg_quality,
        )
        self.assembler = EventAssembler(
            cfg=self.cfg.event,
            buffer=self.buffer,
            out_dir=Path(self.cfg.out_dir),
            camera_id=self.camera.id,
            site_id=self.camera.site_id,
            fps=fps,
            wall_clock_start=datetime.now(UTC),
        )

    def _process_frame(self, frame_idx: int, t_s: float, frame) -> None:
        # Identity checks, not truthiness: RingBuffer defines __len__, so an
        # empty buffer is falsy and a truthy check fails on the first frame.
        assert self.gate is not None
        assert self.tiler is not None
        assert self.buffer is not None
        assert self.assembler is not None

        self.stats.frames_processed += 1
        self.buffer.push(frame_idx, t_s, frame)

        result = self.gate.process(frame)

        if result.global_change:
            self.stats.global_change_frames += 1
        if result.tampered:
            self.stats.tamper_frames += 1

        detections = []
        if result.passed:
            self.stats.frames_gated += 1
            self.stats.rois += len(result.rois)

            tiles = self.tiler.plan(result.rois)
            self.stats.tiles += len(tiles)

            detections = self.detector.detect(frame, tiles, result.rois, frame_idx)
            self.stats.detections += len(detections)

        for track in self.tracker.update(detections, frame_idx, t_s):
            self._judge(track)

        for event in self.assembler.tick(t_s):
            self._record(event)

    def _judge(self, track) -> None:
        assert self.validator is not None
        assert self.assembler is not None

        self.stats.tracks_confirmed += 1
        features = self.validator.features(track)
        verdict = self.validator.judge(track)

        if verdict.passed:
            self.assembler.submit(track, features)
            return

        self.stats.reject(verdict.reason)
        if self.store:
            # Discarded tracks are the hard negatives. Keeping them now is what
            # lets the plausibility classifier be trained without a second
            # collection round later.
            self.store.add_rejection(
                self.run_id,
                self.camera.id,
                track.id,
                verdict.reason,
                track.start_t_s,
                track.end_t_s,
                len(track.points),
                features,
            )

    def _record(self, event: Event) -> None:
        event.model_version = self.detector.name
        self.events.append(event)
        self.stats.events += 1
        if self.store:
            self.store.add_event(self.run_id, event)

    def _finish(self, last_t_s: float) -> None:
        for track in self.tracker.flush():
            self._judge(track)

        if self.assembler is not None:
            for event in self.assembler.flush():
                self._record(event)

        self.stats.tracks_created = self.tracker.created

        if self.store:
            self.store.add_funnel(self.run_id, self.camera.id, self.stats)
            # Stream time watched, not wall-clock elapsed. A source that never
            # yielded a frame observed nothing, which must read as no coverage
            # rather than as a quiet night.
            self.store.finish_run(self.run_id, fps=self.fps, observed_seconds=last_t_s)
