"""Scale-aware compositing: real rodents, at the size our optics actually deliver.

Public camera-trap data has the right *appearance* and the wrong *scale*. A
camera trap sits a metre from a burrow, so its rodents are 108-506 px across
(median 267). Inherited CCTV sees a rodent at 40 px. Training on the former and
deploying on the latter is exactly the mismatch measured in
``experiments/roboflow_eval``: an off-the-shelf rodent detector held confidence
to roughly 180 px and then collapsed to nothing.

This module closes that gap by resampling. It takes a real annotated rodent,
scales it down to a size drawn from the operating distribution, and composites it
into a night background — producing a ground-truth box that is exact by
construction rather than drawn by a human.

Three properties are deliberate:

**It only ever downscales.** Upscaling a small crop invents detail the sensor
never captured, and a detector trained on invented texture learns something that
will not be there at inference.

**The size distribution is weighted toward the decision boundary.** Uniform
sampling across 30-140 px would spend most of its examples where detection is
easy. The architecture's floor is 40 px, so that is where the model needs the
most examples, and :func:`sample_target_px` concentrates them there.

**Degradations are applied after compositing, not before.** Motion blur, sensor
noise and JPEG artefacts belong to the camera, so they must act on the assembled
scene. Applying them to the crop alone would leave a suspiciously clean animal
pasted onto a noisy background, and a detector will happily learn that seam
instead of learning the animal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .lila import BANNER_BOTTOM_FRAC, BANNER_TOP_FRAC, Crop

# The architecture's thresholds: below 40 px detection is not recoverable, and
# below 80 px a species call is not trustworthy.
DETECT_FLOOR_PX = 40
IDENTIFY_FLOOR_PX = 80


@dataclass
class ComposeConfig:
    """Everything that shapes the generated distribution."""

    out_w: int = 1920
    out_h: int = 1080

    # Target long-axis range, in pixels, for the composited animal.
    min_px: int = 24
    max_px: int = 140
    # Fraction of samples drawn tightly around the 40 px detection floor. This
    # is where the model's decision boundary sits and where examples are worth
    # the most.
    boundary_frac: float = 0.45
    boundary_sigma: float = 9.0

    # Source crops smaller than this are not worth downscaling from.
    source_min_px: int = 90

    # Animals travel near the floor, so they appear in the lower part of frame.
    # Expressed as a fraction of usable height.
    band_top: float = 0.45
    band_bottom: float = 0.94

    # Night-camera degradations.
    noise_sigma: float = 4.5
    blur_prob: float = 0.55
    blur_max_px: float = 9.0
    jpeg_quality: tuple[int, int] = (62, 88)

    # Fraction of output frames containing no animal at all. A detector needs
    # true negatives from the same distribution, or it learns that every frame
    # it sees must contain something.
    negative_frac: float = 0.25

    seed: int = 11


@dataclass
class Sample:
    """One generated training image and its exact labels."""

    image: np.ndarray
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    target_px: list[int] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)


def sample_target_px(cfg: ComposeConfig, rng: random.Random) -> int:
    """Draw a target long-axis size, concentrated at the detection floor.

    A uniform draw over 24-140 px would put most examples where the problem is
    already solved. Roughly half the mass sits in a narrow band around 40 px
    instead, because that is where a detector's behaviour actually decides
    whether a deployment works.
    """
    if rng.random() < cfg.boundary_frac:
        px = rng.gauss(DETECT_FLOOR_PX, cfg.boundary_sigma)
    else:
        px = rng.uniform(cfg.min_px, cfg.max_px)
    return int(max(cfg.min_px, min(cfg.max_px, round(px))))


def usable_band(h: int) -> tuple[int, int]:
    """Vertical extent of the frame that is scene rather than camera banner."""
    return int(h * BANNER_TOP_FRAC), int(h * (1.0 - BANNER_BOTTOM_FRAC))


def extract(image: np.ndarray, crop: Crop, pad: float = 0.08) -> np.ndarray | None:
    """Cut the annotated animal out of its source frame, with a little context.

    The padding is what makes seamless cloning work: Poisson blending needs a
    ring of surrounding pixels to match gradients against. With a box cut flush
    to the animal there is nothing to blend and the paste shows a hard edge.
    """
    x, y, w, h = crop.x, crop.y, crop.w, crop.h
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(image.shape[1], x + w + px), min(image.shape[0], y + h + py)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return image[y0:y1, x0:x1].copy()


def is_night_ir(image: np.ndarray, max_spread: float = 6.0) -> bool:
    """Whether a frame was captured in infrared night mode.

    Active-IR night mode is genuinely monochrome — the sensor's colour channels
    agree to within about half a level, measured at 0.6 on sample frames —
    whereas a daylight capture has real channel separation even on a drab scene.

    This matters because converting a sunlit frame to greyscale does **not**
    produce a night frame. Daylight has directional shadows, sky-lit fill and a
    different noise floor; IR has a point-source falloff from the illuminator
    and heavy sensor noise. A detector trained on greyscaled daylight learns the
    wrong background statistics and will not transfer.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return True
    b, g, r = (image[:, :, i].astype(np.int16) for i in range(3))
    spread = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)))
    return spread < max_spread


def to_night_grey(image: np.ndarray) -> np.ndarray:
    """Force a 3-channel greyscale frame.

    Infrared night mode is genuinely monochrome — the sample frames measure a
    channel spread of 0.6 — so anything with residual colour is a daylight
    capture and must be converted rather than passed through. A detector fed a
    mix of the two learns colour as a shortcut feature that vanishes at night.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def motion_blur(image: np.ndarray, length: float, angle_deg: float) -> np.ndarray:
    """Directional blur, standing in for a slow night shutter.

    A rodent at 1.5 m/s under a 1/30 s exposure smears about 8 px across a 40 px
    target. That is the defect the architecture's 1/250 s shutter requirement
    exists to remove — and a detector that has never seen it will fail on every
    camera that has not been reconfigured yet.
    """
    n = max(3, round(length) | 1)
    kernel = np.zeros((n, n), np.float32)
    kernel[n // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (n, n))
    total = kernel.sum()
    if total <= 0:
        return image
    return cv2.filter2D(image, -1, kernel / total)


def paste(
    background: np.ndarray,
    patch: np.ndarray,
    target_px: int,
    rng: random.Random,
    cfg: ComposeConfig,
) -> tuple[int, int, int, int] | None:
    """Scale a crop to `target_px` and blend it into the background.

    Seamless (Poisson) cloning is preferred because it matches the patch's
    illumination to the destination automatically — important when the source
    frame was lit by a camera-trap flash and the destination was not. It fails
    on degenerate geometry near frame edges, so an alpha blend with a feathered
    mask is kept as a fallback rather than letting a sample be dropped.
    """
    ph, pw = patch.shape[:2]
    if pw < 2 or ph < 2:
        return None

    scale = target_px / max(pw, ph)
    if scale >= 1.0:
        return None  # never upscale

    new_w, new_h = max(2, int(pw * scale)), max(2, int(ph * scale))
    small = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

    bh, bw = background.shape[:2]
    top, bottom = usable_band(bh)
    y_lo = max(top, int(bottom * cfg.band_top))
    y_hi = max(y_lo + 1, min(bottom - new_h, int(bottom * cfg.band_bottom) - new_h))
    x_lo, x_hi = 2, max(3, bw - new_w - 2)
    if y_hi <= y_lo or x_hi <= x_lo:
        return None

    x = rng.randint(x_lo, x_hi)
    y = rng.randint(y_lo, y_hi)

    # Feathered ellipse: the annotation is a box, but the animal inside it is
    # not, and a rectangular paste leaves corner artefacts a detector can latch
    # onto instead of the animal.
    mask = np.zeros((new_h, new_w), np.uint8)
    cv2.ellipse(
        mask,
        (new_w // 2, new_h // 2),
        (max(1, int(new_w * 0.46)), max(1, int(new_h * 0.46))),
        0, 0, 360, 255, -1,
    )
    blur_k = max(3, (min(new_w, new_h) // 6) | 1)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

    centre = (x + new_w // 2, y + new_h // 2)
    try:
        blended = cv2.seamlessClone(small, background, mask, centre, cv2.NORMAL_CLONE)
        background[:] = blended
    except cv2.error:
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        region = background[y:y + new_h, x:x + new_w].astype(np.float32)
        background[y:y + new_h, x:x + new_w] = (
            small.astype(np.float32) * alpha + region * (1 - alpha)
        ).astype(np.uint8)

    # The label is the animal's own extent, not the padded patch. The padding
    # added context for blending; counting it would inflate every box by ~8%
    # and teach the detector to predict boxes larger than the animal.
    inset_x = int(new_w * 0.08)
    inset_y = int(new_h * 0.08)
    return (
        x + inset_x,
        y + inset_y,
        max(1, new_w - 2 * inset_x),
        max(1, new_h - 2 * inset_y),
    )


def degrade(image: np.ndarray, rng: random.Random, cfg: ComposeConfig) -> np.ndarray:
    """Apply the camera's own defects to the assembled scene."""
    out = image
    if rng.random() < cfg.blur_prob:
        out = motion_blur(out, rng.uniform(3.0, cfg.blur_max_px), rng.uniform(0, 180))

    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, cfg.noise_sigma, out.shape[:2]
    )
    out = np.clip(out.astype(np.int16) + noise[..., None].astype(np.int16), 0, 255)
    out = out.astype(np.uint8)

    quality = rng.randint(*cfg.jpeg_quality)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is not None:
            out = decoded
    return out


class Composer:
    """Builds training samples from real crops and real backgrounds."""

    def __init__(self, cfg: ComposeConfig | None = None) -> None:
        self.cfg = cfg or ComposeConfig()
        self.rng = random.Random(self.cfg.seed)

    def build(
        self,
        background_path: Path,
        sources: list[tuple[Crop, Path]],
        max_objects: int = 2,
    ) -> Sample | None:
        """Compose one frame from a background and zero or more animals."""
        cfg = self.cfg
        background = cv2.imread(str(background_path))
        if background is None:
            return None

        background = cv2.resize(background, (cfg.out_w, cfg.out_h))
        background = to_night_grey(background)
        sample = Sample(image=background)

        if self.rng.random() < cfg.negative_frac or not sources:
            sample.image = degrade(background, self.rng, cfg)
            return sample

        for _ in range(self.rng.randint(1, max_objects)):
            crop, path = self.rng.choice(sources)
            source = cv2.imread(str(path))
            if source is None:
                continue
            patch = extract(to_night_grey(source), crop)
            if patch is None:
                continue

            target_px = sample_target_px(cfg, self.rng)
            box = paste(background, patch, target_px, self.rng, cfg)
            if box is None:
                continue

            sample.boxes.append(box)
            sample.labels.append(crop.category)
            sample.target_px.append(target_px)
            sample.source_files.append(crop.file_name)

        sample.image = degrade(background, self.rng, cfg)
        return sample


def write_coco(
    samples: list[tuple[str, Sample]],
    out_dir: Path,
    classes: list[str],
    citation: str,
) -> dict[str, int]:
    """Write images plus a COCO annotations file.

    Unlike the export in ``review.py``, these boxes are **ground truth, not
    pre-annotations** — the compositor knows exactly where it put each animal.
    That distinction is recorded in the file so nobody has to guess later.
    """
    import json
    from datetime import UTC, datetime

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i + 1, "name": n, "supercategory": "pest"} for i, n in enumerate(classes)]
    by_name = {c["name"]: c["id"] for c in categories}

    images: list[dict] = []
    annotations: list[dict] = []

    for image_id, (name, sample) in enumerate(samples, start=1):
        cv2.imwrite(str(images_dir / name), sample.image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        h, w = sample.image.shape[:2]
        images.append({"id": image_id, "file_name": name, "width": w, "height": h})

        for box, label, px, src in zip(
            sample.boxes, sample.labels, sample.target_px, sample.source_files
        ):
            x, y, bw, bh = box
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": by_name.get(label, 1),
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "attributes": {
                        "ground_truth": True,
                        "target_long_axis_px": px,
                        "source_image": src,
                    },
                }
            )

    payload = {
        "info": {
            "description": (
                "ExpressVision scale-corrected training set. Real camera-trap "
                "animals resampled to inherited-CCTV pixel sizes and composited "
                "into night backgrounds. Boxes are exact by construction."
            ),
            "version": "1.0",
            "date_created": datetime.now(UTC).isoformat(),
            "source_attribution": citation,
        },
        "licenses": [{"id": 1, "name": "CDLA-Permissive-1.0 (source imagery)"}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    (out_dir / "annotations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    sizes = [a["attributes"]["target_long_axis_px"] for a in annotations]
    return {
        "images": len(images),
        "annotations": len(annotations),
        "negatives": sum(1 for _, s in samples if not s.boxes),
        "median_px": int(sorted(sizes)[len(sizes) // 2]) if sizes else 0,
        "at_or_below_floor": sum(1 for s in sizes if s <= DETECT_FLOOR_PX),
    }


def size_histogram(sizes: list[int], bins: int = 8) -> str:
    """A text histogram, for confirming the distribution landed where intended."""
    if not sizes:
        return "(no samples)"
    lo, hi = min(sizes), max(sizes)
    span = max(1, hi - lo)
    counts = [0] * bins
    for s in sizes:
        counts[min(bins - 1, int((s - lo) / span * bins))] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        a = lo + span * i / bins
        b = lo + span * (i + 1) / bins
        bar = "#" * round(c / peak * 34)
        marker = "  <- detection floor" if a <= DETECT_FLOOR_PX <= b else ""
        lines.append(f"  {a:5.0f}-{b:<5.0f} px |{bar:<34}| {c:>6,}{marker}")
    return "\n".join(lines)
