"""Configuration for the collector pipeline.

Every threshold that varies by site lives here rather than being buried in code.
Per-camera overrides come from a YAML file so a field engineer can tune a site
without touching Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class GateConfig(BaseModel):
    """Motion gate — stage B of the cascade."""

    # Work at this width for background subtraction. Gating is cheap because it
    # happens on a downscaled copy; detection still gets native pixels (see tiling).
    work_width: int = 480

    history: int = 500
    var_threshold: float = 24.0
    learning_rate: float = 0.003

    # Frames to let the background model settle before emitting any candidate.
    warmup_frames: int = 60

    # Morphology kernel sizes, in work-space pixels.
    open_kernel: int = 3
    close_kernel: int = 7

    # Candidate blob area bounds as a fraction of the work-space frame area.
    min_area_frac: float = 0.00008  # ~15 px at 480x270
    max_area_frac: float = 0.15

    # Blobs whose bounding boxes are within this many work-space pixels of each
    # other get merged — one animal frequently fragments into several contours.
    merge_distance: int = 24

    # If more than this fraction of pixels change in one frame, treat it as a
    # global illumination change (lights, IR-cut toggle, camera bump) rather than
    # as motion: reset the model and emit nothing.
    global_change_frac: float = 0.25
    global_change_reset_frames: int = 30

    # Tamper / defocus: Laplacian variance collapsing well below its rolling
    # median for a sustained period means a fouled or covered lens.
    tamper_enabled: bool = True
    tamper_ratio: float = 0.35
    tamper_frames: int = 90
    tamper_sample_every: int = 5


class TileConfig(BaseModel):
    """Native-resolution tiling — stage C of the cascade."""

    tile_size: int = 640
    # Pad each ROI by this fraction of its size before fitting a tile, so the
    # animal is not flush against the tile edge.
    roi_padding: float = 0.35
    # Cap tiles per frame so one chaotic frame cannot stall the pipeline.
    max_tiles_per_frame: int = 8


class TrackConfig(BaseModel):
    """Multi-object tracking — stage D."""

    iou_threshold: float = 0.2
    # Frames a track survives without a match before it is closed. At 15 fps,
    # 30 frames is 2 s — enough to cross behind a pallet leg.
    max_age: int = 30
    # Matches required before a track is considered real at all.
    min_hits: int = 3

    # How far an object may travel between frames, as a multiple of its own
    # size, before we stop believing it is the same object.
    #
    # This exists because IoU association silently fails for exactly the targets
    # this system cares about. A rat is small and fast: at 5 fps on a sub-stream
    # it covers more than its own body length per frame, so consecutive boxes do
    # not overlap at all and IoU is zero. Falling back to centre distance keeps
    # the track alive. Proximity matches rank below genuine overlap, so a real
    # IoU match always wins when both are available.
    max_travel_factor: float = 2.5
    # Proximity matches are discounted so overlap is preferred where it exists.
    proximity_weight: float = 0.9


class ValidatorConfig(BaseModel):
    """Track validation — stage E, the false-positive firewall."""

    min_detections: int = 5
    min_frames_span: int = 8

    # Spatial extent of the whole trajectory, as a fraction of the frame
    # diagonal. This is the primary spatial gate: did the object cover ground?
    # Vegetation swaying about a fixed point cannot clear it however long it is
    # observed, and an animal that doubles back still can.
    min_extent_frac: float = 0.08

    # Straightness (net displacement / path length) is only decisive for tracks
    # that stayed in a small area. Above this extent a wandering path is normal
    # animal behaviour, not a sign of noise, so the check is not applied.
    straightness_below_extent_frac: float = 0.20
    min_straightness: float = 0.12
    # Reject tracks whose apparent size swings wildly — usually rain, insects
    # near the lens, or a detector latching onto changing shadow.
    max_area_cv: float = 1.2


class EventConfig(BaseModel):
    """Event assembly — stage F."""

    pre_roll_s: float = 8.0
    post_roll_s: float = 5.0
    # Longest stretch of the track itself to keep. An animal that settles in
    # frame for ten minutes should not produce a ten-minute clip, and the ring
    # buffer would have to hold ten minutes of video to cut one.
    max_clip_s: float = 20.0
    # JPEG quality for ring-buffer frames. Resident memory per camera is roughly
    # (pre_roll + max_clip + post_roll) x fps x frame size; see RingBuffer.
    jpeg_quality: int = 80
    write_clips: bool = True
    write_keyframes: bool = True


class CameraConfig(BaseModel):
    """One camera. `source` is a file path or an RTSP URL."""

    id: str
    source: str
    name: str = ""
    site_id: str = "site-001"
    # Process every Nth frame. Pest work wants >=10 fps; 2 on a 30 fps stream
    # gives 15 fps and halves decode cost.
    frame_stride: int = 1
    # Survey findings from the architecture doc, carried so the pipeline can warn
    # when a camera is being asked to do something its optics cannot support.
    lens_mm: float | None = None
    resolution_w: int | None = None
    max_detect_range_m: float | None = None
    notes: str = ""


class PipelineConfig(BaseModel):
    gate: GateConfig = Field(default_factory=GateConfig)
    tile: TileConfig = Field(default_factory=TileConfig)
    track: TrackConfig = Field(default_factory=TrackConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    event: EventConfig = Field(default_factory=EventConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)

    out_dir: Path = Path("out")
    db_path: Path = Path("out/expressvision.db")

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    @classmethod
    def for_source(cls, source: str, camera_id: str = "cam-001") -> PipelineConfig:
        """Single-source config for ad-hoc runs against one file or stream."""
        return cls(cameras=[CameraConfig(id=camera_id, source=source, name=camera_id)])
