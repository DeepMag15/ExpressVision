"""Detector-protocol adapter, so the Roboflow model can be plugged into the
existing pipeline experimentally without changing any of it.

This exists to answer "how would it behave inside our cascade?" — not to become
a production path. It is in ``experiments/`` for that reason, and importing it
from ``src/expressvision`` would be a mistake.

Two properties make it unusable in production regardless of accuracy:

* Every tile is an HTTPS round trip. Our cascade produces roughly 25,000 tiles
  per camera-night; at a 200 ms round trip that is 1.4 hours of pure network
  wait per camera, per night, and it scales linearly with cameras.
* Client CCTV frames leave the site. Hospital, food-processing and residential
  deployments have constraints on that which the architecture already commits
  to honouring.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from experiments.roboflow_eval.client import RoboflowDetector
from expressvision.tiling import map_to_frame, nms
from expressvision.types import Box, Detection, Tile

# Roboflow class names vary by project; map whatever it returns onto our L2
# taxonomy so downstream stages see the vocabulary they expect.
RODENT_ALIASES = {"rat", "rats", "mouse", "mice", "rodent", "vermin", "pest"}


class RoboflowTileDetector:
    """Runs a hosted Roboflow model over the cascade's native-resolution tiles."""

    def __init__(
        self,
        model_id: str,
        tile_size: int = 640,
        confidence: float = 0.25,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.tile_size = tile_size
        self.name = f"roboflow/{model_id}"
        self._client = RoboflowDetector(
            model_id, api_key=api_key, confidence=confidence
        )
        self._tmp = Path(tempfile.mkdtemp(prefix="exv-roboflow-"))
        self.calls = 0
        self.total_latency_ms = 0.0

    def detect(
        self,
        frame: np.ndarray,
        tiles: list[Tile],
        rois: list[Box],
        frame_idx: int,
    ) -> list[Detection]:
        if not tiles:
            return []

        height, width = frame.shape[:2]
        detections: list[Detection] = []

        for n, tile in enumerate(tiles):
            patch = frame[tile.y : tile.y + tile.size, tile.x : tile.x + tile.size]
            if tile.size != self.tile_size:
                patch = cv2.resize(
                    patch, (self.tile_size, self.tile_size), interpolation=cv2.INTER_AREA
                )

            # The SDK takes a path or URL, so each tile becomes a temp file.
            # Another reason this is an experiment: a per-tile disk write and
            # HTTPS upload is not a viable inner loop.
            path = self._tmp / f"f{frame_idx}_t{n}.jpg"
            cv2.imwrite(str(path), patch, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            result = self._client.infer(path)
            self.calls += 1
            self.total_latency_ms += result.latency_ms
            path.unlink(missing_ok=True)

            if not result.ok:
                continue

            for prediction in result.predictions:
                x1, y1, x2, y2 = prediction.xyxy
                detections.append(
                    Detection(
                        box=map_to_frame(
                            tile, Box(x1, y1, x2, y2), self.tile_size, width, height
                        ),
                        score=prediction.confidence,
                        label=self._map_label(prediction.label),
                        frame_idx=frame_idx,
                    )
                )

        return nms(detections)

    @staticmethod
    def _map_label(label: str) -> str:
        lowered = label.lower().strip()
        if any(alias in lowered for alias in RODENT_ALIASES):
            return "rodent"
        return lowered or "unknown"

    def runtime_info(self) -> dict[str, str]:
        mean = self.total_latency_ms / self.calls if self.calls else 0.0
        return {
            "variant": self.model_id,
            "requested_device": "roboflow-hosted",
            "resolved_device": "roboflow-hosted",
            "actual_device": "remote (Roboflow cloud)",
            "actual_dtype": "n/a",
            "image_size": str(self.tile_size),
            "tile_size": str(self.tile_size),
            "calls": str(self.calls),
            "mean_latency_ms": f"{mean:.0f}",
        }

    def warmup(self, iterations: int = 1) -> None:
        """One throwaway call, so the first timed tile is not paying for TLS
        handshake and DNS."""
        blank = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        path = self._tmp / "warmup.jpg"
        cv2.imwrite(str(path), blank)
        for _ in range(max(1, iterations)):
            self._client.infer(path)
        path.unlink(missing_ok=True)
