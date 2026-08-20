"""Rolling frame buffer for evidence pre-roll.

An evidence clip that starts at the moment of detection is nearly useless — the
animal is already mid-frame and the reviewer cannot see where it came from. The
clip has to start *before* the detection, which means continuously retaining the
last few seconds of every camera.

Frames are held JPEG-encoded rather than raw: 8 s at 15 fps of 1080p BGR is
~1.5 GB per camera, which is impossible across sixteen cameras; the same window
as JPEG is ~25 MB.

Production note: past roughly eight cameras per node this should become
disk-backed segment recording — write continuous short MP4 segments and cut
clips from them — which trades a little I/O for near-zero resident memory. The
in-memory version here is right for a single-site collector and for development.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BufferedFrame:
    frame_idx: int
    t_s: float
    jpeg: bytes

    def decode(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


class RingBuffer:
    def __init__(self, seconds: float, fps: float, jpeg_quality: int = 80) -> None:
        self.seconds = seconds
        self.fps = max(1.0, fps)
        self.jpeg_quality = jpeg_quality
        # A little headroom so a variable frame rate cannot shorten the window.
        self._buf: deque[BufferedFrame] = deque(maxlen=int(seconds * self.fps * 1.25) + 15)
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def bytes_held(self) -> int:
        return sum(len(f.jpeg) for f in self._buf)

    def push(self, frame_idx: int, t_s: float, frame_bgr: np.ndarray) -> None:
        ok, enc = cv2.imencode(".jpg", frame_bgr, self._encode_params)
        if ok:
            self._buf.append(BufferedFrame(frame_idx, t_s, enc.tobytes()))

    def window(self, start_t: float, end_t: float) -> list[BufferedFrame]:
        """Every retained frame in ``[start_t, end_t]``, oldest first.

        Returns whatever the buffer still holds — if the requested pre-roll
        predates the buffer, the clip is simply shorter rather than failing.
        """
        return [f for f in self._buf if start_t <= f.t_s <= end_t]

    def oldest_t(self) -> float | None:
        return self._buf[0].t_s if self._buf else None

    def newest_t(self) -> float | None:
        return self._buf[-1].t_s if self._buf else None
