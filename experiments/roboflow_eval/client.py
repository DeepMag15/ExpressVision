"""Thin wrapper over the Roboflow hosted inference API.

Deliberately isolated from ``src/expressvision``. This is an evaluation, and a
hosted third-party detector is architecturally incompatible with the production
design in several ways that are worth stating up front rather than discovering
after integration:

* **It needs the internet.** The architecture requires each site to keep
  detecting, recording and alerting with its uplink down. A cloud detector
  breaks that outright.
* **It sends client footage to a third party.** Hospital, food-plant and
  residential sites have contractual and regulatory constraints on where video
  goes. This would have to be disclosed and agreed per site.
* **Per-call latency is network-bound.** At ~25k tiles per camera-night, a
  200 ms round trip is over an hour of pure waiting per camera.

None of that makes it useless for evaluation — it makes it unusable as a
production detector, which is a different question and one worth separating.

The API key is read from the ``ROBOFLOW_API_KEY`` environment variable and is
never written to disk or logged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENDPOINT = "https://serverless.roboflow.com"


class MissingApiKey(RuntimeError):
    pass


@dataclass
class Prediction:
    """One detection, normalised out of Roboflow's response shape."""

    label: str
    confidence: float
    x: float          # centre-x, pixels
    y: float          # centre-y, pixels
    width: float
    height: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    @property
    def area_fraction_of(self) -> float:
        return self.width * self.height


@dataclass
class InferenceResult:
    predictions: list[Prediction] = field(default_factory=list)
    latency_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    error: str | None = None
    raw: dict | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def best(self) -> Prediction | None:
        return max(self.predictions, key=lambda p: p.confidence, default=None)


class RoboflowDetector:
    """Calls a hosted Roboflow model. One image per request."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        confidence: float = 0.10,
        overlap: float = 0.50,
    ) -> None:
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise MissingApiKey(
                "ROBOFLOW_API_KEY is not set.\n"
                "  PowerShell:  $env:ROBOFLOW_API_KEY = 'your-key'\n"
                "  bash:        export ROBOFLOW_API_KEY=your-key\n"
                "Never put the key in a file inside the repo."
            )
        self._key = key
        self.model_id = model_id
        self.endpoint = endpoint
        # A low default threshold on purpose: for an evaluation we want to see
        # weak responses. A model that fires at 0.12 on our domain and 0.9 on
        # its own is telling us something a 0.5 cutoff would hide.
        self.confidence = confidence
        self.overlap = overlap
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from inference_sdk import InferenceConfiguration, InferenceHTTPClient
            except ImportError as exc:
                raise ImportError(
                    "inference-sdk is not installed. This is an experimental "
                    "dependency, kept out of the main project:\n"
                    "  uv pip install inference-sdk"
                ) from exc
            client = InferenceHTTPClient(api_url=self.endpoint, api_key=self._key)
            client.configure(
                InferenceConfiguration(
                    confidence_threshold=self.confidence,
                    iou_threshold=self.overlap,
                )
            )
            self._client = client
        return self._client

    def infer(self, image_path: str | Path) -> InferenceResult:
        """Run one image. Network and API errors are returned, not raised, so a
        single failure does not abandon a whole evaluation run."""
        path = str(image_path)
        started = time.perf_counter()
        try:
            raw = self.client.infer(path, model_id=self.model_id)
        except Exception as exc:  # noqa: BLE001 - surface any API failure verbatim
            return InferenceResult(
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=self._redact(f"{type(exc).__name__}: {exc}"),
            )
        latency_ms = (time.perf_counter() - started) * 1000.0

        if isinstance(raw, list):                       # some endpoints batch
            raw = raw[0] if raw else {}

        predictions = [
            Prediction(
                label=str(p.get("class", "?")),
                confidence=float(p.get("confidence", 0.0)),
                x=float(p.get("x", 0.0)),
                y=float(p.get("y", 0.0)),
                width=float(p.get("width", 0.0)),
                height=float(p.get("height", 0.0)),
            )
            for p in (raw.get("predictions") or [])
        ]
        image_meta = raw.get("image") or {}
        return InferenceResult(
            predictions=predictions,
            latency_ms=latency_ms,
            image_width=int(image_meta.get("width", 0) or 0),
            image_height=int(image_meta.get("height", 0) or 0),
            raw=raw,
        )

    def probe(self) -> dict:
        """Check the model resolves and the key works, before spending quota.

        Done by running one inference on a small generated image rather than by
        fetching metadata: the serverless endpoint is inference-only and answers
        a bare GET with 400, and the hosted API exposes no public metadata route
        for a model in someone else's workspace. The response shape and any
        class names it returns are what we can actually observe.
        """
        import tempfile

        import cv2
        import numpy as np

        probe_image = np.full((320, 320, 3), 90, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.jpg"
            cv2.imwrite(str(path), probe_image)
            result = self.infer(path)

        if not result.ok:
            return {"ok": False, "error": result.error}
        return {
            "ok": True,
            "latency_ms": round(result.latency_ms, 1),
            "response_keys": sorted((result.raw or {}).keys()),
            "n_predictions_on_blank": len(result.predictions),
            "raw": result.raw,
        }

    def _redact(self, text: str) -> str:
        """Strip the key from anything we are about to print or write.

        Error bodies and exception messages routinely echo the request URL, and
        the URL carries the key as a query parameter.
        """
        return text.replace(self._key, "***REDACTED***")
