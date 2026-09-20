"""Index and selectively fetch public camera-trap imagery from LILA BC.

The corpus used here is **Channel Islands Camera Traps** (The Nature Conservancy,
2021), hosted by LILA BC under the Community Data License Agreement, permissive
variant. That licence matters as much as the content: this project already
refuses AGPL components in anything shipped to a client, and a training corpus
carries the same kind of obligation. CDLA-Permissive imposes none that conflict
with commercial deployment.

Why this corpus specifically — every annotation carries a bounding box, and the
images are 1920x1080, the same resolution as the CCTV this system reads:

    rodent    82,912 boxes   the target class, in infrared, at night
    human      5,981 boxes   the other Phase-1 class
    bird      11,099 boxes   the hard negative that matters most
    fox       48,145 boxes   four-legged non-target
    skunk      1,071 boxes   four-legged non-target
    empty    114,894 frames  real backgrounds, real sensor noise

The bird class earns its place: the evaluated third-party detector called a
pigeon a rodent at 0.729 confidence, and a corpus without birds cannot teach a
model not to.

The full image set is 86 GB. We never download it. The metadata is 18 MB, and
individual images are addressable, so a run pulls only the few thousand frames a
build actually needs.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

# LILA mirrors the same tree on GCP, AWS and Azure. GCP is the default; the
# others are drop-in replacements if a network blocks it.
IMAGE_BASE = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "channel-islands-camera-traps/images"
)
METADATA_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "channel-islands-camera-traps/channel-islands-camera-traps.json.zip"
)

CITATION = (
    "The Nature Conservancy (2021): Channel Islands Camera Traps 1.0. "
    "The Nature Conservancy. Dataset. Licensed CDLA-Permissive-1.0 via LILA BC."
)

# Reconyx cameras burn a status banner into the top and bottom of every frame.
# It is not scene content: a rodent must never be composited onto it, and a
# background must never be cropped from it.
BANNER_TOP_FRAC = 0.035
BANNER_BOTTOM_FRAC = 0.055


@dataclass(frozen=True)
class Crop:
    """One annotated animal, addressed by source file and box."""

    file_name: str
    category: str
    x: int
    y: int
    w: int
    h: int
    img_w: int
    img_h: int

    @property
    def long_axis(self) -> int:
        return max(self.w, self.h)

    def url(self) -> str:
        return f"{IMAGE_BASE}/{self.file_name}"

    def local_name(self) -> str:
        """A flat filename; the source tree is three levels deep."""
        return self.file_name.replace("/", "__")


def download_metadata(dest_dir: Path) -> Path:
    """Fetch and unpack the 18 MB metadata archive. Returns the .json path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(dest_dir.glob("*camera_traps*.json"))
    if existing:
        return existing[0]

    archive = dest_dir / "channel-islands-metadata.json.zip"
    if not archive.exists():
        urllib.request.urlretrieve(METADATA_URL, archive)

    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if not names:
            raise ValueError(f"no .json inside {archive}")
        zf.extract(names[0], dest_dir)
    return dest_dir / names[0]


class LilaIndex:
    """A searchable view over the metadata, built once and cached.

    The raw metadata is 152 MB of JSON and takes several seconds to parse. The
    cache written here loads instantly, which matters because a dataset build is
    iterative — it gets rebuilt many times while the size distribution is tuned.
    """

    def __init__(self, crops: list[Crop], empties: list[Crop]) -> None:
        self.crops = crops
        self.empties = empties

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, metadata_json: Path, cache: Path | None = None) -> LilaIndex:
        if cache and Path(cache).exists():
            return cls.load(Path(cache))

        raw = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
        images = {i["id"]: i for i in raw["images"]}
        names = {c["id"]: c["name"] for c in raw["categories"]}

        crops: list[Crop] = []
        empties: list[Crop] = []

        for ann in raw["annotations"]:
            box = ann.get("bbox")
            if not box:
                continue
            img = images.get(ann["image_id"])
            if not img or not img.get("width"):
                continue

            x, y, w, h = (round(v) for v in box)
            if w <= 0 or h <= 0:
                continue

            category = names.get(ann["category_id"], "unknown")
            item = Crop(
                file_name=img["file_name"],
                category=category,
                x=x, y=y, w=w, h=h,
                img_w=int(img["width"]), img_h=int(img["height"]),
            )

            # "empty" carries a full-frame placeholder box: it is a background,
            # not an object. Anything else that fills the frame is a labelling
            # artefact and is no use as a crop either.
            fills_frame = w >= item.img_w * 0.98 and h >= item.img_h * 0.98
            if category == "empty":
                empties.append(item)
            elif not fills_frame:
                crops.append(item)

        index = cls(crops, empties)
        if cache:
            index.save(Path(cache))
        return index

    # ------------------------------------------------------------------- io
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "citation": CITATION,
            "crops": [c.__dict__ for c in self.crops],
            "empties": [c.__dict__ for c in self.empties],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LilaIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [Crop(**c) for c in payload["crops"]],
            [Crop(**c) for c in payload["empties"]],
        )

    # ---------------------------------------------------------------- query
    def select(
        self,
        category: str,
        min_px: int = 0,
        max_px: int = 10_000,
        limit: int | None = None,
        seed: int = 7,
    ) -> list[Crop]:
        """Crops of one class whose long axis falls in a pixel range.

        The lower bound matters more than it looks. A 60 px source crop rescaled
        down to 40 px is honest; a 60 px crop rescaled *up* invents detail the
        optics never delivered, and a detector trained on it learns texture that
        will not exist at inference. Compose only ever downscales, and this floor
        is what guarantees there is something to downscale from.
        """
        pool = [
            c
            for c in self.crops
            if c.category == category and min_px <= c.long_axis <= max_px
        ]
        rng = random.Random(seed)
        rng.shuffle(pool)
        return pool[:limit] if limit else pool

    def backgrounds(self, limit: int | None = None, seed: int = 7) -> list[Crop]:
        pool = list(self.empties)
        rng = random.Random(seed)
        rng.shuffle(pool)
        return pool[:limit] if limit else pool

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.crops:
            counts[c.category] = counts.get(c.category, 0) + 1
        counts["empty"] = len(self.empties)
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def fetch(crop: Crop, cache_dir: Path, timeout: float = 60.0) -> Path | None:
    """Download one source image, or return it from cache.

    Returns None rather than raising on a network failure: a build pulling
    thousands of images should degrade to a smaller dataset, not abort at 90%.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / crop.local_name()
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    try:
        with urllib.request.urlopen(crop.url(), timeout=timeout) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None

    if not data:
        return None
    dest.write_bytes(data)
    return dest


def fetch_many(
    crops: list[Crop],
    cache_dir: Path,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> Iterator[tuple[Crop, Path]]:
    """Yield (crop, path) for every image that downloaded successfully."""
    ok = 0
    for i, crop in enumerate(crops, 1):
        path = fetch(crop, cache_dir)
        if path is not None:
            ok += 1
            yield crop, path
        if on_progress:
            on_progress(i, len(crops), ok)
