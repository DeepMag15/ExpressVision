"""Video sources.

A source yields ``(frame_index, timestamp_seconds, frame_bgr)``. File sources
run as fast as they decode; RTSP sources run in real time and reconnect on drop,
because a camera that reboots at 03:00 must not end the night's collection.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SourceInfo:
    width: int
    height: int
    fps: float
    frame_count: int | None  # None for live streams


class VideoSource:
    """Base class. Subclasses implement :meth:`frames`."""

    def __init__(self, uri: str, frame_stride: int = 1) -> None:
        self.uri = uri
        self.frame_stride = max(1, frame_stride)
        self.info: SourceInfo | None = None

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        raise NotImplementedError

    @staticmethod
    def _probe(cap: cv2.VideoCapture, live: bool) -> SourceInfo:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240:
            fps = 15.0  # NVR sub-streams frequently report nonsense
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return SourceInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            frame_count=None if live or count <= 0 else count,
        )


class FileSource(VideoSource):
    """Decodes a video file as fast as possible. Timestamps come from the
    container, so analytics on recorded footage carry true wall-clock offsets."""

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        path = Path(self.uri)
        if not path.exists():
            raise FileNotFoundError(f"video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self.info = self._probe(cap, live=False)

        try:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self.frame_stride == 0:
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    t_s = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else idx / self.info.fps
                    yield idx, t_s, frame
                idx += 1
        finally:
            cap.release()


class RtspSource(VideoSource):
    """Live RTSP with bounded-backoff reconnect.

    Uses the main stream deliberately — sub-streams are typically D1 and destroy
    small targets before the pipeline ever sees them.
    """

    def __init__(
        self,
        uri: str,
        frame_stride: int = 1,
        max_reconnects: int = 0,      # 0 = retry forever
        backoff_start_s: float = 1.0,
        backoff_max_s: float = 30.0,
    ) -> None:
        super().__init__(uri, frame_stride)
        self.max_reconnects = max_reconnects
        self.backoff_start_s = backoff_start_s
        self.backoff_max_s = backoff_max_s

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        # Keep the internal queue shallow: on a live feed, a stale frame is
        # worth less than a dropped one.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        idx = 0
        attempts = 0
        backoff = self.backoff_start_s
        t0 = time.monotonic()

        while True:
            cap = self._open()
            if not cap.isOpened():
                cap.release()
                attempts += 1
                if self.max_reconnects and attempts > self.max_reconnects:
                    raise RuntimeError(f"RTSP unreachable after {attempts} attempts: {self.uri}")
                time.sleep(backoff)
                backoff = min(backoff * 2, self.backoff_max_s)
                continue

            if self.info is None:
                self.info = self._probe(cap, live=True)
            attempts = 0
            backoff = self.backoff_start_s

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break  # stream dropped — fall through to reconnect
                    if idx % self.frame_stride == 0:
                        yield idx, time.monotonic() - t0, frame
                    idx += 1
            finally:
                cap.release()

            time.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max_s)


def open_source(uri: str, frame_stride: int = 1) -> VideoSource:
    """Pick a source implementation from the URI scheme."""
    lowered = uri.lower()
    if lowered.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return RtspSource(uri, frame_stride)
    return FileSource(uri, frame_stride)
