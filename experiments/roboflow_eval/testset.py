"""Build a controlled test set for evaluating a third-party rodent detector.

The point of this evaluation is *not* "does it find a rat in a nice photo" — any
rodent model does. The question is whether it works on the imagery Express
Pesticides actually has: greyscale, near-infrared-illuminated CCTV at night,
with the animal small, distant, motion-blurred and low-contrast.

So the design is a controlled ablation rather than a grab-bag of images. Each
base subject is rendered under a series of single-variable transformations, so a
drop in detection can be attributed to a specific property rather than guessed
at. Running one image through and declaring success is exactly the mistake this
is built to avoid.

The distinction that matters most:

* **Thermal (LWIR, 8-14 um)** is passive heat emission. A rodent is a bright
  blob on a cool background, with no texture, no shadow and no colour. Sensors
  are typically 320x240 and cost 10-50x a CCTV camera.
* **Near-IR CCTV night mode (~850 nm)** is a normal CMOS sensor with the IR-cut
  filter swung out, lit by IR LEDs. It looks like a greyscale *photograph* —
  texture, shadows, reflections and all. A rodent is mid-grey on a grey floor,
  frequently *darker* than its background.

These are different sensing modalities, not different lighting conditions. A
detector trained on one has no particular reason to work on the other, and the
ablation below is designed to tell them apart.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

UA = {"User-Agent": "ExpressVision-eval/0.1 (model evaluation)"}


@dataclass
class TestImage:
    """One test case, with everything needed to interpret the result."""

    name: str
    path: Path
    subject: str                  # what is in it
    condition: str                # the single variable under test
    modality: str                 # visible | nir_sim | thermal_sim | thermal_real | synthetic
    expect_rodent: bool
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def fetch_wikipedia_image(title: str, timeout: int = 60) -> np.ndarray | None:
    """Grab the lead image for a Wikipedia article."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            meta = json.load(response)
        source = (meta.get("originalimage") or meta.get("thumbnail") or {}).get("source")
        if not source:
            return None
        with urllib.request.urlopen(
            urllib.request.Request(source, headers=UA), timeout=timeout
        ) as response:
            buffer = np.frombuffer(response.read(), dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001 - network fetch, report and continue
        print(f"    fetch failed for {title}: {exc}")
        return None


# --------------------------------------------------------------- transformations

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Colour removed, nothing else. Isolates colour dependence."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def to_nir_night(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Approximate a near-IR CCTV night frame.

    Greyscale, compressed dynamic range (IR scenes are flat), an illuminator
    hot-spot falling off toward the edges, and sensor noise. This is what our
    cameras actually produce, and it is the condition that matters most.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # IR illuminators are point sources: bright centre, dark corners.
    h, w = grey.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    radius = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    falloff = np.clip(1.0 - 0.55 * strength * radius**1.6, 0.25, 1.0)

    # Flatten contrast toward mid-grey; IR scenes have little dynamic range.
    flattened = 128.0 + (grey - grey.mean()) * 0.55
    out = flattened * falloff

    rng = np.random.default_rng(11)
    out = out + rng.normal(0, 5.0 * strength, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def to_thermal_sim(img: np.ndarray, colormap: int | None = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Approximate a thermal (LWIR) frame: warm subject, cool background.

    A real microbolometer image is not obtainable by transforming a photograph —
    there are no shadows, no texture, and the intensity encodes temperature, not
    reflectance. What this *does* reproduce is the property a thermal-trained
    detector most likely keys on: a bright, compact, high-contrast blob on a
    dark, near-uniform background. Treat a hit here as evidence about that
    property, not proof the model saw real thermal data.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Push the subject bright and the surround dark: strong contrast stretch
    # around the upper intensity range, then blur away texture detail.
    blurred = cv2.GaussianBlur(grey, (0, 0), sigmaX=max(1.0, grey.shape[1] / 320))
    stretched = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    gamma = np.power(stretched.astype(np.float32) / 255.0, 1.9) * 255.0
    hot = np.clip(gamma, 0, 255).astype(np.uint8)

    if colormap is None:
        return cv2.cvtColor(hot, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(hot, colormap)


def shrink_subject(img: np.ndarray, fraction: float, canvas: int = 1280) -> np.ndarray:
    """Place the subject small in a large frame — the distance case.

    A rat 10 m from a 2 MP camera occupies a low single-digit percentage of the
    frame. Most published rodent models are evaluated on close-ups, so this is
    where the gap between a demo and a deployment shows up.
    """
    height = int(canvas * 9 / 16)
    target_w = max(8, int(canvas * fraction))
    scale = target_w / img.shape[1]
    small = cv2.resize(
        img, (target_w, max(4, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )

    # Mid-grey floor, roughly the tone of a concrete warehouse floor under IR.
    frame = np.full((height, canvas, 3), 62, dtype=np.uint8)
    rng = np.random.default_rng(3)
    frame = np.clip(
        frame.astype(np.int16) + rng.normal(0, 4, frame.shape).astype(np.int16), 0, 255
    ).astype(np.uint8)

    y = int(height * 0.62)
    x = int(canvas * 0.35)
    y = min(y, height - small.shape[0] - 1)
    x = min(x, canvas - small.shape[1] - 1)
    frame[y : y + small.shape[0], x : x + small.shape[1]] = small
    return frame


def motion_blur(img: np.ndarray, length: int = 21) -> np.ndarray:
    """Horizontal smear, as from a slow night shutter on a moving animal.

    Cameras routinely drop to 1/8 s at night. A rat at 1.5 m/s smears across a
    meaningful fraction of its own body length, which is the failure the
    architecture's shutter-speed requirement exists to prevent.
    """
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(img, -1, kernel)


def low_contrast(img: np.ndarray, factor: float = 0.35) -> np.ndarray:
    mean = img.mean()
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def high_angle(img: np.ndarray) -> np.ndarray:
    """Perspective warp approximating a ceiling-mounted camera looking down.

    Security cameras are mounted for faces, not floors, so most real views of a
    rodent are steeply foreshortened from above — an aspect almost absent from
    curated rodent datasets, which are shot from the side at eye level.
    """
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.22, 0], [w * 0.78, 0], [w, h], [0, h]])
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
    return warped


def multi_subject(img: np.ndarray, count: int = 3, canvas: int = 1280) -> np.ndarray:
    """Several rodents in one frame at CCTV scale."""
    height = int(canvas * 9 / 16)
    frame = np.full((height, canvas, 3), 62, dtype=np.uint8)
    target_w = max(20, int(canvas * 0.07))
    scale = target_w / img.shape[1]
    small = cv2.resize(
        img, (target_w, max(6, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )

    for i in range(count):
        x = int(canvas * (0.12 + 0.28 * i))
        y = int(height * (0.45 + 0.13 * (i % 2)))
        x = min(x, canvas - small.shape[1] - 1)
        y = min(y, height - small.shape[0] - 1)
        frame[y : y + small.shape[0], x : x + small.shape[1]] = small
    return frame


# ------------------------------------------------------------------- assembly

SUBJECTS = [
    ("Brown rat", "rat"),
    ("House mouse", "mouse"),
    ("Black rat", "rat"),
]

NEGATIVE_SUBJECTS = [
    ("Cat", "cat"),
    ("Domestic pigeon", "bird"),
]

# Real thermal imagery, to check the thermal hypothesis against actual LWIR
# rather than only against a simulation of it.
THERMAL_SUBJECTS = [
    ("Thermography", "thermal scene"),
    ("Thermographic camera", "thermal scene"),
]


def build(out_dir: Path, fixture_video: Path | None = None) -> list[TestImage]:
    """Assemble the test set on disk and return its manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[TestImage] = []

    def save(img: np.ndarray, name: str, **kwargs) -> None:
        path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cases.append(TestImage(name=name, path=path, **kwargs))

    print("  fetching rodent subjects...")
    for title, species in SUBJECTS:
        img = fetch_wikipedia_image(title)
        if img is None:
            continue
        slug = title.lower().replace(" ", "_")

        # The ablation: one variable at a time, same subject throughout.
        save(img, f"{slug}__visible", subject=species, condition="daylight close-up",
             modality="visible", expect_rodent=True,
             notes="Best case. Any rodent model should hit this.",
             tags=["baseline"])

        save(to_grayscale(img), f"{slug}__grayscale", subject=species,
             condition="colour removed", modality="visible", expect_rodent=True,
             notes="Isolates colour dependence.", tags=["ablation"])

        save(to_nir_night(img), f"{slug}__nir_night", subject=species,
             condition="near-IR CCTV night", modality="nir_sim", expect_rodent=True,
             notes="OUR ACTUAL DOMAIN. Greyscale, flat contrast, IR falloff, noise.",
             tags=["ablation", "critical"])

        save(to_thermal_sim(img), f"{slug}__thermal_colour", subject=species,
             condition="thermal, inferno palette", modality="thermal_sim",
             expect_rodent=True,
             notes="Hot-blob-on-cool-background, the property LWIR training keys on.",
             tags=["ablation", "critical"])

        save(to_thermal_sim(img, colormap=None), f"{slug}__thermal_whitehot",
             subject=species, condition="thermal, white-hot", modality="thermal_sim",
             expect_rodent=True, notes="Same, monochrome white-hot palette.",
             tags=["ablation", "critical"])

        save(shrink_subject(img, 0.10), f"{slug}__small_10pct", subject=species,
             condition="subject 10% of frame width", modality="visible",
             expect_rodent=True, notes="Mid-range CCTV distance.",
             tags=["ablation", "small-object"])

        save(shrink_subject(img, 0.03), f"{slug}__small_3pct", subject=species,
             condition="subject 3% of frame width", modality="visible",
             expect_rodent=True, notes="Realistic CCTV distance (~10 m on 2 MP).",
             tags=["ablation", "small-object"])

        save(to_nir_night(shrink_subject(img, 0.05)), f"{slug}__small_nir", subject=species,
             condition="small AND near-IR", modality="nir_sim", expect_rodent=True,
             notes="The realistic combination: both conditions at once.",
             tags=["ablation", "critical", "small-object"])

        save(motion_blur(img), f"{slug}__motion_blur", subject=species,
             condition="horizontal motion blur", modality="visible",
             expect_rodent=True, notes="Slow night shutter on a moving animal.",
             tags=["ablation"])

        save(low_contrast(img), f"{slug}__low_contrast", subject=species,
             condition="compressed dynamic range", modality="visible",
             expect_rodent=True, tags=["ablation"])

        save(high_angle(img), f"{slug}__high_angle", subject=species,
             condition="ceiling-mounted viewpoint", modality="visible",
             expect_rodent=True,
             notes="Security cameras look down; rodent datasets are shot side-on.",
             tags=["ablation"])

        save(multi_subject(img), f"{slug}__multiple", subject=f"3x {species}",
             condition="three rodents, CCTV scale", modality="visible",
             expect_rodent=True, tags=["ablation", "multi"])

    print("  fetching real thermal imagery...")
    for title, description in THERMAL_SUBJECTS:
        img = fetch_wikipedia_image(title)
        if img is None:
            continue
        slug = title.lower().replace(" ", "_")
        save(img, f"{slug}__thermal_real", subject=description,
             condition="genuine LWIR thermogram", modality="thermal_real",
             expect_rodent=False,
             notes="Real thermal, no rodent. Checks whether the model fires on "
                   "thermal-looking imagery regardless of content.",
             tags=["control", "thermal"])

    print("  fetching negative controls...")
    for title, species in NEGATIVE_SUBJECTS:
        img = fetch_wikipedia_image(title)
        if img is None:
            continue
        slug = title.lower().replace(" ", "_")
        save(img, f"{slug}__negative", subject=species, condition="non-rodent animal",
             modality="visible", expect_rodent=False,
             notes="False-positive check: a cat is not a rodent.",
             tags=["control"])
        save(to_nir_night(img), f"{slug}__negative_nir", subject=species,
             condition="non-rodent animal, near-IR", modality="nir_sim",
             expect_rodent=False, tags=["control"])

    # Empty scene: nothing should ever be detected here.
    empty = np.full((720, 1280, 3), 60, dtype=np.uint8)
    rng = np.random.default_rng(5)
    empty = np.clip(
        empty.astype(np.int16) + rng.normal(0, 5, empty.shape).astype(np.int16), 0, 255
    ).astype(np.uint8)
    cv2.rectangle(empty, (0, 0), (1280, 210), (40, 40, 40), -1)
    save(empty, "empty_floor", subject="nothing", condition="empty warehouse floor",
         modality="nir_sim", expect_rodent=False,
         notes="Pure false-positive check.", tags=["control"])

    if fixture_video and fixture_video.exists():
        print("  extracting frames from our own fixture...")
        _add_fixture_frames(fixture_video, save)

    return cases


def _add_fixture_frames(video: Path, save) -> None:
    """Frames from our synthetic fixture — the exact input our pipeline sees."""
    cap = cv2.VideoCapture(str(video))
    # Frames during a rodent crossing, per the fixture plan.
    for idx, label in ((135, "crossing"), (150, "crossing"), (300, "crossing")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            save(frame, f"fixture_f{idx}", subject="synthetic rodent",
                 condition=f"our pipeline fixture, {label}", modality="synthetic",
                 expect_rodent=True,
                 notes="Synthetic, so a miss here is weak evidence — but it is "
                       "literally what our collector processes today.",
                 tags=["fixture"])
    cap.release()


def write_manifest(cases: list[TestImage], path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "name": c.name,
                    "path": str(c.path),
                    "subject": c.subject,
                    "condition": c.condition,
                    "modality": c.modality,
                    "expect_rodent": c.expect_rodent,
                    "notes": c.notes,
                    "tags": c.tags,
                }
                for c in cases
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
