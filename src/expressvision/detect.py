"""Detectors — stage C of the cascade.

Two implementations, for two different jobs.

``MotionPassthrough`` promotes every gated region to a detection. Use it when
the goal is to *harvest* candidate footage for labelling: it keeps everything,
including the unusual examples a model trained on someone else's domain would
silently discard.

``MegaDetectorAdapter`` wraps Microsoft's MegaDetector v6 — animal / person /
vehicle proposals trained on camera-trap imagery, which is the closest public
domain to night-time pest footage that exists. It is the bootstrap model: not a
pest detector, but good enough to turn "all motion" into "mostly animals" and
make labelling several times faster.

Both run on the *tile crops* the gate asked for, at native resolution, never on
a downscaled whole frame — that is the point of stage C.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .tiling import map_to_frame, nms
from .types import Box, Detection, Tile


class Detector(Protocol):
    """Given a frame and the tiles the gate asked for, return detections in
    full-resolution frame coordinates."""

    name: str

    def detect(
        self,
        frame: np.ndarray,
        tiles: list[Tile],
        rois: list[Box],
        frame_idx: int,
    ) -> list[Detection]:
        ...


class MotionPassthrough:
    """Every ROI covered by a planned tile becomes a ``motion`` detection.

    Only ROIs that a tile actually covers are emitted, so the tile budget is
    honoured and the funnel counts stay honest — a frame whose ROIs exceeded
    ``max_tiles_per_frame`` really did drop the surplus, and the numbers should
    say so.
    """

    name = "motion-passthrough-v0"

    def detect(
        self,
        frame: np.ndarray,
        tiles: list[Tile],
        rois: list[Box],
        frame_idx: int,
    ) -> list[Detection]:
        covered: list[int] = []
        for tile in tiles:
            covered.extend(tile.roi_indices)

        return [
            Detection(box=rois[i], score=1.0, label="motion", frame_idx=frame_idx)
            for i in sorted(set(covered))
            if 0 <= i < len(rois)
        ]


# MegaDetector's three classes, mapped onto the L1 taxonomy from the
# architecture. It cannot tell a rat from a cat — that is what fine-tuning on
# the client's own footage is for. What it can do is separate living things from
# the swaying, flapping, dripping things that dominate a motion gate's output.
MEGADETECTOR_LABELS = {
    0: "animal",
    1: "human",
    2: "vehicle",
}

# Verified against PytorchWildlife 1.3.0 by reading megadetectorv6.py — the
# published docs list a nine-variant zoo with MIT/Apache names that the released
# package does not accept. These five are what actually load.
MEGADETECTOR_VARIANTS = {
    # id                architecture              note
    "MDV6-yolov9-c":   ("YOLOv9 compact",        "fast; sensible default for harvesting"),
    "MDV6-yolov9-e":   ("YOLOv9 extra @1280",    "best recall; slowest on CPU"),
    "MDV6-yolov10-c":  ("YOLOv10 compact",       "smallest"),
    "MDV6-yolov10-e":  ("YOLOv10 extra @1280",   "high recall"),
    "MDV6-rtdetr-c":   ("RT-DETR compact",       "transformer; matches the production arch"),
}

MEGADETECTOR_DEFAULT = "MDV6-yolov9-c"

# The operative licence constraint is the runtime, not the weights.
# PytorchWildlife loads every one of these through Ultralytics, and depends on
# ultralytics and yolov5 — both AGPL-3.0 — whichever variant is selected. So no
# choice here makes the stack shippable.
#
# That is acceptable, because of where this runs: sorting our own collected
# footage on our own machines is internal use, and AGPL copyleft triggers on
# distribution and on network interaction with third parties. Neither applies to
# a labelling pipeline.
#
# It must never be installed on an edge node at a client site. The production
# detector is a separately trained model exported to ONNX and served through
# ONNX Runtime or TensorRT, with none of this in the image. The Detector
# protocol already makes that a packaging rule rather than a code change.
HARVEST_ONLY_NOTICE = (
    "MegaDetector runs through Ultralytics (AGPL-3.0). Use it for internal "
    "harvesting and labelling only — it must not ship to a client site."
)


class MegaDetectorAdapter:
    """MegaDetector v6 over native-resolution tiles.

    Loaded lazily so that importing this module — and running the rest of the
    test suite — does not require torch to be installed.
    """

    def __init__(
        self,
        version: str = MEGADETECTOR_DEFAULT,
        device: str = "auto",
        conf_threshold: float = 0.20,
        tile_size: int = 640,
        image_size: int | None = None,
        half: bool = False,
    ) -> None:
        if version not in MEGADETECTOR_VARIANTS:
            raise ValueError(
                f"Unknown MegaDetector variant {version!r}. "
                f"Available: {sorted(MEGADETECTOR_VARIANTS)}"
            )

        self.version = version
        self.conf_threshold = conf_threshold
        self.tile_size = tile_size
        self.image_size = image_size
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.half = half
        self.name = f"megadetector/{version}"
        self._model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self):
        """Load on first use. Weights download automatically and are cached."""
        if self._model is None:
            try:
                from PytorchWildlife.models import detection as pw_detection
            except ImportError as exc:  # pragma: no cover - environment dependent
                # Surface the real missing module. PytorchWildlife's package
                # __init__ imports its bioacoustics subpackage eagerly, so an
                # undeclared transitive dependency surfaces here as an
                # ImportError that has nothing to do with the ml extra.
                raise ImportError(
                    f"Could not import PytorchWildlife: {exc}\n"
                    f"Install the ml extra with `uv sync --extra ml`. If the extra "
                    f"is already installed, the module named above is a missing "
                    f"transitive dependency — install it directly."
                ) from exc
            self._model = pw_detection.MegaDetectorV6(
                version=self.version, device=self.device, pretrained=True
            )
            if self.image_size is not None:
                # PytorchWildlife hardcodes 1280 for every variant, so a 640 px
                # tile is upscaled and costs ~4x the compute. Higher resolution
                # does help small targets, so 1280 stays the default — but on
                # CPU, or when harvesting a backlog, 640 is the difference
                # between hours and days.
                self._model.IMAGE_SIZE = self.image_size
                self._model.predictor.args.imgsz = self.image_size
            self._apply_runtime()
        return self._model

    def _apply_runtime(self) -> None:
        """Put the model on the requested device, in the requested precision.

        PytorchWildlife takes a ``device`` argument and never applies it — the
        line that would is commented out in its source (``yolov8_base.py``:
        ``# self.predictor.args.device = device # Will uncomment later``). It
        happens to work on a CUDA box because Ultralytics auto-selects a GPU,
        but "works by accident" is not good enough when the whole point of a
        benchmark is to know what the hardware did.
        """
        import torch

        predictor = getattr(self._model, "predictor", None)
        if predictor is None:
            return

        predictor.args.device = self.device
        predictor.args.half = self.half

        inner = getattr(predictor, "model", None)
        if inner is None:
            return

        target = torch.device(self.device)
        inner.to(target)
        if self.half and target.type == "cuda":
            inner.half()
        # AutoBackend caches both; leaving them stale makes it cast inputs for
        # the wrong device or dtype.
        for holder in (predictor, inner):
            if hasattr(holder, "device"):
                holder.device = target
            if hasattr(holder, "fp16"):
                holder.fp16 = bool(self.half and target.type == "cuda")

    def runtime_info(self) -> dict[str, str]:
        """Where the model *actually* is, read back from its parameters.

        Never report the requested device — report the observed one. A benchmark
        that silently measured CPU and labelled it GPU is worse than no
        benchmark, because the capacity plan built on it looks credible.
        """
        info = {
            "variant": self.version,
            "requested_device": self.requested_device,
            "resolved_device": self.device,
            "actual_device": "not loaded",
            "actual_dtype": "not loaded",
            "image_size": str(self.image_size or "model default (1280)"),
            "tile_size": str(self.tile_size),
        }
        if self._model is None:
            return info

        backend = getattr(getattr(self._model, "predictor", None), "model", None)
        param = _first_parameter(backend)
        if param is not None:
            info["actual_device"] = str(param.device)
            info["actual_dtype"] = str(param.dtype).replace("torch.", "")
        elif backend is not None and getattr(backend, "device", None) is not None:
            # Ultralytics' AutoBackend records its own device; less direct than
            # reading a tensor but better than reporting nothing.
            info["actual_device"] = str(backend.device)
            info["actual_dtype"] = "float16" if getattr(backend, "fp16", False) else "float32"
        else:
            info["actual_device"] = "unknown"
            info["actual_dtype"] = "unknown"
        return info

    def warmup(self, iterations: int = 3) -> None:
        """Run throwaway inferences so timings exclude one-off setup.

        The first CUDA call pays for context creation, cuDNN autotuning and lazy
        kernel loading — often several seconds. Measuring that as if it were
        steady-state throughput understates the hardware badly.
        """
        blank = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        for _ in range(max(1, iterations)):
            self._infer([blank])

    def detect(
        self,
        frame: np.ndarray,
        tiles: list[Tile],
        rois: list[Box],
        frame_idx: int,
    ) -> list[Detection]:
        if not tiles:
            return []

        import cv2

        # MegaDetector wants RGB; OpenCV decodes BGR. Crops are taken at native
        # resolution — the model's own input size is 1280, so a 640 tile is
        # upscaled internally, which costs time but destroys no detail.
        patches = []
        for tile in tiles:
            patch = frame[tile.y : tile.y + tile.size, tile.x : tile.x + tile.size]
            if tile.size != self.tile_size:
                patch = cv2.resize(
                    patch, (self.tile_size, self.tile_size), interpolation=cv2.INTER_AREA
                )
            patches.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

        height, width = frame.shape[:2]
        detections: list[Detection] = []
        for tile, result in zip(tiles, self._infer(patches)):
            for box, score, class_id in result:
                detections.append(
                    Detection(
                        box=map_to_frame(tile, box, self.tile_size, width, height),
                        score=score,
                        label=MEGADETECTOR_LABELS.get(class_id, "unknown"),
                        frame_idx=frame_idx,
                    )
                )

        # Suppress duplicates from overlapping tiles, in frame coordinates.
        return nms(detections)

    def _infer(self, patches: list[np.ndarray]) -> list[list[tuple[Box, float, int]]]:
        """Run the model over a batch of tiles, one result list per tile.

        Batched because a busy frame yields several tiles and per-tile calls
        waste most of the GPU. Results are read defensively: PytorchWildlife
        returns a supervision ``Detections`` whose field population has varied
        across releases, and a missing confidence array must degrade rather than
        end a night's collection.
        """
        if not patches:
            return []

        if len(patches) == 1:
            raw = [self.model.single_image_detection(
                patches[0], det_conf_thres=self.conf_threshold
            )]
        else:
            raw = self.model.batch_image_detection(
                patches, batch_size=len(patches), det_conf_thres=self.conf_threshold
            )

        out: list[list[tuple[Box, float, int]]] = []
        for result in raw:
            det = result.get("detections") if isinstance(result, dict) else None
            if det is None or not hasattr(det, "xyxy") or len(det.xyxy) == 0:
                out.append([])
                continue

            boxes = np.asarray(det.xyxy, dtype=float)
            scores = getattr(det, "confidence", None)
            class_ids = getattr(det, "class_id", None)

            parsed: list[tuple[Box, float, int]] = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                parsed.append(
                    (
                        Box(float(x1), float(y1), float(x2), float(y2)),
                        float(scores[i]) if scores is not None else 1.0,
                        int(class_ids[i]) if class_ids is not None else 0,
                    )
                )
            out.append(parsed)

        # Guard against a release that reorders or drops results: a silent
        # mismatch would map every box onto the wrong tile.
        if len(out) != len(patches):
            raise RuntimeError(
                f"MegaDetector returned {len(out)} results for {len(patches)} tiles"
            )
        return out


def _first_parameter(module: object):
    """Find the first real tensor parameter under a model wrapper.

    Ultralytics wraps the network in ``AutoBackend``, which holds it in a plain
    attribute rather than as a registered submodule — so ``parameters()`` on the
    wrapper yields nothing at all and the obvious probe reports "unknown".
    Reading an actual tensor is worth the walk: it is the only way to know for
    certain which device and dtype inference will really use.
    """
    seen: set[int] = set()
    candidates = [module]
    for _ in range(4):                      # bounded: wrappers nest 1-2 deep
        nxt = []
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            params = getattr(candidate, "parameters", None)
            if callable(params):
                try:
                    param = next(params(), None)
                except (TypeError, StopIteration):
                    param = None
                if param is not None:
                    return param
            for attr in ("model", "net", "module"):
                nxt.append(getattr(candidate, attr, None))
        candidates = nxt
        if not any(c is not None for c in candidates):
            break
    return None


def build_detector(kind: str = "motion", **kwargs) -> Detector:
    """Detector factory, so the CLI can switch models without importing torch."""
    if kind == "motion":
        return MotionPassthrough()
    if kind == "megadetector":
        return MegaDetectorAdapter(**kwargs)
    raise ValueError(f"unknown detector: {kind!r} (expected 'motion' or 'megadetector')")
