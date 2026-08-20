"""Camera assessment — the Phase 0 deliverable.

Answers one question per camera, from a sample of its footage: **can this camera
physically see a rodent, and if not, why not?**

This is the highest-rated risk in the project. Most CCTV is mounted for faces at
head height, not for rodents on the floor, and a system deployed onto cameras
that cannot resolve a rat will be blamed for missing what it never had a chance
of seeing. Finding that out in week one is enormously cheaper than in month six.

The measurements are deliberately separated into two kinds, because they have
very different costs to fix:

* **Fixable by configuration** — night shutter too slow, gain too high, wrong
  stream selected. A settings push, free.
* **Fixable only by hardware** — too far, lens too wide, resolution too low,
  camera aimed at the wrong thing. Money and a site visit.

Every number is reported with the raw value, not just a verdict, because these
are indicators rather than certainties and a human should be able to disagree
with the tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# A rodent's long axis. Everything about range is derived from this.
RODENT_M = 0.25
DETECT_FLOOR_PX = 40      # below this, no model recovers the animal
IDENTIFY_FLOOR_PX = 80    # below this, species calls are not trustworthy

# Common CCTV sensor widths, in mm. 1/2.8" dominates 2-4 MP cameras.
SENSOR_WIDTHS_MM = {
    '1/4"': 3.60,
    '1/3"': 4.80,
    '1/2.8"': 5.37,
    '1/2.7"': 5.37,
    '1/2.5"': 5.76,
    '1/1.8"': 7.18,
}


@dataclass
class Optics:
    """What we know about the camera's physical setup.

    All optional: the tool still reports image quality without them, it just
    cannot compute range.
    """

    hfov_deg: float | None = None
    lens_mm: float | None = None
    sensor: str = '1/2.8"'
    nearest_floor_m: float | None = None
    furthest_floor_m: float | None = None

    def resolved_hfov(self) -> float | None:
        """Horizontal field of view, preferring a measured/quoted figure.

        Computing FOV from focal length assumes a sensor size that datasheets
        often report loosely, so a manufacturer's quoted HFOV is the better
        input where it exists.
        """
        if self.hfov_deg:
            return self.hfov_deg
        if self.lens_mm and self.lens_mm > 0:
            width = SENSOR_WIDTHS_MM.get(self.sensor, 5.37)
            return math.degrees(2 * math.atan(width / (2 * self.lens_mm)))
        return None


@dataclass
class Finding:
    """One observation, with enough context to argue with it."""

    name: str
    value: str
    status: str          # ok | warn | fail | info
    note: str = ""
    fix: str = ""        # blank when nothing can be done about it


@dataclass
class SurveyResult:
    source: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    frames_sampled: int = 0

    night_fraction: float = 0.0
    sharpness: float = 0.0
    temporal_noise: float = 0.0
    blur_ratio: float = 1.0
    corner_falloff: float = 1.0
    blockiness: float = 1.0
    exposure_drift: float = 0.0
    motion_fraction: float = 0.0

    findings: list[Finding] = field(default_factory=list)
    optics: Optics = field(default_factory=Optics)
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.error:
            return "unreadable"
        if any(f.status == "fail" for f in self.findings):
            return "replace or relocate"
        if any(f.status == "warn" for f in self.findings):
            return "usable after reconfiguration"
        return "usable"

    def pixels_on_rodent_at(self, distance_m: float) -> float | None:
        """How many pixels a rodent spans at a given distance."""
        hfov = self.optics.resolved_hfov()
        if not hfov or not self.width or distance_m <= 0:
            return None
        scene_width_m = 2 * distance_m * math.tan(math.radians(hfov) / 2)
        if scene_width_m <= 0:
            return None
        return RODENT_M * self.width / scene_width_m

    def max_range_m(self, floor_px: float = DETECT_FLOOR_PX) -> float | None:
        """Furthest distance at which a rodent still spans ``floor_px``."""
        hfov = self.optics.resolved_hfov()
        if not hfov or not self.width:
            return None
        ppm_needed = floor_px / RODENT_M
        return self.width / (2 * math.tan(math.radians(hfov) / 2) * ppm_needed)


def survey(
    source: str | Path,
    optics: Optics | None = None,
    sample_frames: int = 120,
) -> SurveyResult:
    """Assess one camera from a clip or a still image."""
    optics = optics or Optics()
    path = Path(source)
    result = SurveyResult(source=str(path), optics=optics)

    if not path.exists():
        result.error = f"file not found: {path}"
        return result

    frames, meta = _sample(path, sample_frames)
    if not frames:
        result.error = "could not decode any frames"
        return result

    result.width, result.height = meta["width"], meta["height"]
    result.fps, result.duration_s = meta["fps"], meta["duration_s"]
    result.frames_sampled = len(frames)

    greys = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    result.night_fraction = _night_fraction(frames)
    result.sharpness = float(np.median([cv2.Laplacian(g, cv2.CV_64F).var() for g in greys]))
    result.temporal_noise = _temporal_noise(greys)
    result.blur_ratio = _motion_blur_ratio(greys)
    result.corner_falloff = _corner_falloff(greys)
    result.blockiness = _blockiness(greys)
    result.exposure_drift = _exposure_drift(greys)
    result.motion_fraction = _motion_fraction(greys)

    result.findings = _assess(result)
    return result


# ---------------------------------------------------------------- sampling

def _sample(path: Path, want: int) -> tuple[list[np.ndarray], dict]:
    """Read frames spread across the clip, not just the opening seconds.

    A camera's first few seconds are unrepresentative — auto-exposure is still
    settling, and night footage often starts at dusk.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps <= 0 or fps > 240:
        fps = 15.0

    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(fps),
        "duration_s": total / fps if total else 0.0,
    }

    frames: list[np.ndarray] = []
    try:
        if total <= 1:                      # a still image
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        else:
            # Consecutive pairs at intervals: temporal noise and motion blur
            # both need adjacent frames, not scattered singles.
            pairs = max(2, want // 2)
            step = max(1, total // pairs)
            for i in range(0, total - 1, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok1, f1 = cap.read()
                ok2, f2 = cap.read()
                if ok1 and ok2:
                    frames.extend((f1, f2))
                if len(frames) >= want:
                    break
    finally:
        cap.release()

    return frames, meta


# ------------------------------------------------------------ measurements

def _night_fraction(frames: list[np.ndarray]) -> float:
    """Share of frames in infrared night mode.

    IR footage is effectively monochrome: the camera swings its IR-cut filter
    out and all three channels carry the same signal. Near-zero saturation is a
    reliable tell.
    """
    night = 0
    for frame in frames:
        saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        if float(saturation.mean()) < 12.0:
            night += 1
    return night / len(frames) if frames else 0.0


def _temporal_noise(greys: list[np.ndarray]) -> float:
    """Sensor noise, measured where nothing is moving.

    This is what the motion gate has to see through. Measured as the standard
    deviation of frame-to-frame differences in the *quietest* areas, so a
    forklift driving past does not get counted as noise.
    """
    samples = []
    for a, b in zip(greys[::2], greys[1::2]):
        diff = cv2.absdiff(a, b).astype(np.float32)
        # Lower half of the difference distribution = the static parts.
        threshold = np.percentile(diff, 60)
        quiet = diff[diff <= threshold]
        if quiet.size:
            samples.append(float(quiet.std()))
    return float(np.median(samples)) if samples else 0.0


def _motion_blur_ratio(greys: list[np.ndarray]) -> float:
    """Edge sharpness inside moving regions, relative to static ones.

    A slow night shutter smears anything that moves while leaving the static
    background crisp — so moving areas being markedly blurrier than still ones
    is the signature. Near 1.0 means moving objects stay sharp; well below
    means the shutter is too slow to freeze them.

    Returns 1.0 when there is too little motion to judge.
    """
    ratios = []
    for a, b in zip(greys[::2], greys[1::2]):
        diff = cv2.absdiff(a, b)
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        moving = mask > 0
        if moving.sum() < 400 or (~moving).sum() < 400:
            continue

        lap = np.abs(cv2.Laplacian(b, cv2.CV_64F))
        moving_sharp = lap[moving].mean()
        static_sharp = lap[~moving].mean()
        if static_sharp > 1e-6:
            ratios.append(float(moving_sharp / static_sharp))

    return float(np.median(ratios)) if ratios else 1.0


def _corner_falloff(greys: list[np.ndarray]) -> float:
    """Corner brightness relative to the centre.

    IR illuminators are point sources mounted around the lens, so the centre is
    lit and the corners are not. Where the ratio is very low the camera is
    effectively blind at the edges of its own view, regardless of resolution.
    """
    ratios = []
    for grey in greys[:24]:
        h, w = grey.shape
        ch, cw = h // 4, w // 4
        centre = grey[h // 2 - ch // 2: h // 2 + ch // 2, w // 2 - cw // 2: w // 2 + cw // 2]
        corners = np.concatenate(
            [
                grey[:ch, :cw].ravel(), grey[:ch, -cw:].ravel(),
                grey[-ch:, :cw].ravel(), grey[-ch:, -cw:].ravel(),
            ]
        )
        centre_mean = float(centre.mean())
        if centre_mean > 1e-6:
            ratios.append(float(corners.mean()) / centre_mean)
    return float(np.median(ratios)) if ratios else 1.0


def _blockiness(greys: list[np.ndarray]) -> float:
    """Compression artefacts, as gradient energy on the 8-pixel grid.

    Heavy re-encoding produces visible 8x8 blocks. Above about 1.3 the footage
    has been compressed hard enough to erase small-target detail — usually a
    sign of the wrong export setting rather than the wrong camera.
    """
    ratios = []
    for grey in greys[:24]:
        g = grey.astype(np.float32)
        dx = np.abs(np.diff(g, axis=1))
        if dx.shape[1] < 16:
            continue
        on_grid = dx[:, 7::8].mean()
        off = np.delete(dx, np.s_[7::8], axis=1).mean()
        if off > 1e-6:
            ratios.append(float(on_grid / off))
    return float(np.median(ratios)) if ratios else 1.0


def _exposure_drift(greys: list[np.ndarray]) -> float:
    """Frame-to-frame swing in overall brightness.

    Auto-exposure hunting makes the whole image pulse, which a background
    subtractor reads as motion everywhere. A camera that does this constantly
    will keep the gate firing all night.
    """
    means = [float(g.mean()) for g in greys]
    if len(means) < 3:
        return 0.0
    return float(np.median(np.abs(np.diff(means))))


def _motion_fraction(greys: list[np.ndarray]) -> float:
    """Share of sampled frames with meaningful movement in them.

    Context for everything else: a clip with no motion cannot tell us anything
    about motion blur, and a clip with constant motion suggests something in
    view moves permanently.
    """
    moving = 0
    pairs = 0
    for a, b in zip(greys[::2], greys[1::2]):
        pairs += 1
        diff = cv2.absdiff(a, b)
        if float((diff > 18).mean()) > 0.001:
            moving += 1
    return moving / pairs if pairs else 0.0


# --------------------------------------------------------------- assessment

def _assess(r: SurveyResult) -> list[Finding]:
    out: list[Finding] = []

    # --- resolution -------------------------------------------------------
    megapixels = (r.width * r.height) / 1e6
    if r.width and r.width <= 720:
        out.append(Finding(
            "Resolution", f"{r.width}x{r.height} ({megapixels:.1f} MP)", "fail",
            "This is sub-stream resolution. A rodent here is around a dozen "
            "pixels — unrecoverable by any software.",
            "Re-export from the main/primary stream.",
        ))
    elif megapixels < 1.8:
        out.append(Finding(
            "Resolution", f"{r.width}x{r.height} ({megapixels:.1f} MP)", "warn",
            "Low for pest work; usable only at short range.",
        ))
    else:
        out.append(Finding(
            "Resolution", f"{r.width}x{r.height} ({megapixels:.1f} MP)", "ok"))

    # --- frame rate -------------------------------------------------------
    if r.fps and r.fps < 8:
        out.append(Finding(
            "Frame rate", f"{r.fps:.0f} fps", "warn",
            "Below ~10 fps a fast animal moves further than its own body "
            "between frames, which makes it much harder to follow.",
            "Raise the recording frame rate if the NVR allows it.",
        ))
    else:
        out.append(Finding("Frame rate", f"{r.fps:.0f} fps", "ok"))

    # --- night mode -------------------------------------------------------
    if r.night_fraction >= 0.6:
        out.append(Finding(
            "Night / IR mode", f"{r.night_fraction:.0%} of frames", "ok",
            "Infrared night footage — the condition that matters most."))
    elif r.night_fraction <= 0.05:
        out.append(Finding(
            "Night / IR mode", "none detected", "info",
            "This looks like daylight footage. Most pest activity is after "
            "dark, so we also need overnight recordings from this camera."))
    else:
        out.append(Finding(
            "Night / IR mode", f"{r.night_fraction:.0%} of frames", "info",
            "Mixed day and night — likely spans dusk or dawn."))

    # --- focus ------------------------------------------------------------
    if r.sharpness < 25:
        out.append(Finding(
            "Focus / lens condition", f"{r.sharpness:.0f}", "fail",
            "Very little detail in the image. Usually a defocused lens, a dirty "
            "or fogged dome, or a spider web across it.",
            "Clean and refocus the lens, then re-record.",
        ))
    elif r.sharpness < 80:
        out.append(Finding(
            "Focus / lens condition", f"{r.sharpness:.0f}", "warn",
            "Softer than expected. Worth a physical check of the dome.",
            "Clean the dome; confirm focus.",
        ))
    else:
        out.append(Finding("Focus / lens condition", f"{r.sharpness:.0f}", "ok"))

    # --- noise ------------------------------------------------------------
    if r.temporal_noise > 12:
        out.append(Finding(
            "Sensor noise", f"{r.temporal_noise:.1f}", "warn",
            "High. Usually the camera compensating for too little light by "
            "raising gain. Noise this heavy makes movement detection harder "
            "and costs accuracy.",
            "Add or repair IR illumination so gain can come down.",
        ))
    else:
        out.append(Finding("Sensor noise", f"{r.temporal_noise:.1f}", "ok"))

    # --- motion blur ------------------------------------------------------
    if r.motion_fraction < 0.05:
        out.append(Finding(
            "Motion blur", "not measurable", "info",
            "Almost nothing moved in this sample, so shutter speed could not "
            "be judged. Send a clip containing some activity."))
    elif r.blur_ratio < 0.55:
        out.append(Finding(
            "Motion blur", f"{r.blur_ratio:.2f} sharpness vs static", "fail",
            "Moving objects are heavily smeared while the background stays "
            "sharp — the classic slow-night-shutter signature. A rat crossing "
            "this view would be a streak, not a shape.",
            "Set a minimum shutter of 1/250 s in night mode. This usually "
            "needs more IR light to stay bright enough.",
        ))
    elif r.blur_ratio < 0.75:
        out.append(Finding(
            "Motion blur", f"{r.blur_ratio:.2f} sharpness vs static", "warn",
            "Some smearing on moving objects.",
            "Raise the night shutter speed toward 1/250 s.",
        ))
    else:
        out.append(Finding(
            "Motion blur", f"{r.blur_ratio:.2f} sharpness vs static", "ok"))

    # --- illumination -----------------------------------------------------
    if r.corner_falloff < 0.30:
        out.append(Finding(
            "IR illumination evenness", f"corners {r.corner_falloff:.0%} of centre",
            "warn",
            "The edges of this view are barely lit. The camera is effectively "
            "blind there whatever its resolution.",
            "Add a separate IR illuminator, or accept that only the centre of "
            "this view is covered.",
        ))
    else:
        out.append(Finding(
            "IR illumination evenness", f"corners {r.corner_falloff:.0%} of centre",
            "ok"))

    # --- compression ------------------------------------------------------
    if r.blockiness > 1.35:
        out.append(Finding(
            "Compression", f"{r.blockiness:.2f}x block-edge energy", "warn",
            "Heavily compressed. Small-target detail is being destroyed by the "
            "encoder, not by the lens.",
            "Export at a higher bitrate, or copy the original file without "
            "re-encoding.",
        ))
    else:
        out.append(Finding("Compression", f"{r.blockiness:.2f}x block-edge energy", "ok"))

    # --- exposure stability ----------------------------------------------
    if r.exposure_drift > 2.5:
        out.append(Finding(
            "Exposure stability", f"{r.exposure_drift:.1f} levels/frame", "warn",
            "Overall brightness is pulsing. Auto-exposure hunting looks like "
            "movement everywhere and will keep the detector busy all night.",
            "Lock exposure, or widen the auto-exposure deadband.",
        ))
    else:
        out.append(Finding(
            "Exposure stability", f"{r.exposure_drift:.1f} levels/frame", "ok"))

    out.extend(_assess_range(r))
    return out


def _assess_range(r: SurveyResult) -> list[Finding]:
    """The hardware question: is the animal big enough in this view at all?"""
    hfov = r.optics.resolved_hfov()
    if not hfov:
        return [Finding(
            "Detection range", "not calculated", "info",
            "Provide --hfov-deg (preferred) or --lens-mm, plus --furthest-m, "
            "and this reports the actual usable range for this camera.")]

    detect = r.max_range_m(DETECT_FLOOR_PX)
    identify = r.max_range_m(IDENTIFY_FLOOR_PX)
    out = [Finding(
        "Usable range", f"detect to {detect:.1f} m · identify to {identify:.1f} m",
        "info", f"Assumes a {RODENT_M * 100:.0f} cm animal and {hfov:.0f}° "
                f"horizontal field of view.")]

    furthest = r.optics.furthest_floor_m
    if furthest:
        px = r.pixels_on_rodent_at(furthest)
        if px is None:
            return out
        if px < DETECT_FLOOR_PX:
            out.append(Finding(
                "Coverage at furthest point",
                f"{px:.0f} px at {furthest:.1f} m", "fail",
                f"A rodent at the far end of this view is {px:.0f} pixels — "
                f"below the {DETECT_FLOOR_PX} px floor. Nothing can detect it "
                f"there. Only the nearer part of this view is covered.",
                f"Add a camera covering the far area, or fit a longer lens "
                f"(narrower view) to reach {furthest:.1f} m.",
            ))
        elif px < IDENTIFY_FLOOR_PX:
            out.append(Finding(
                "Coverage at furthest point",
                f"{px:.0f} px at {furthest:.1f} m", "warn",
                f"Detectable at the far end, but too small to identify the "
                f"species reliably ({IDENTIFY_FLOOR_PX} px needed).",
            ))
        else:
            out.append(Finding(
                "Coverage at furthest point",
                f"{px:.0f} px at {furthest:.1f} m", "ok",
                "Rodents are large enough across the whole view."))

    nearest = r.optics.nearest_floor_m
    if nearest:
        px = r.pixels_on_rodent_at(nearest)
        if px is not None:
            out.append(Finding(
                "Coverage at nearest point", f"{px:.0f} px at {nearest:.1f} m",
                "ok" if px >= IDENTIFY_FLOOR_PX else "info"))

    return out
