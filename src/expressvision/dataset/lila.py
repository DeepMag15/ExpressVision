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

Images can come from any of three places, and :class:`ImageSource` picks one:

* **Network** — the metadata is 18 MB and individual frames are addressable, so
  a build pulls only the few thousand it needs rather than the 86 GB set. Needs
  egress, which a shared research box often lacks.
* **A local directory** — the archive already extracted.
* **The archive itself** — read members straight out of the zip. A zip has a
  central directory, so members are random-access; extracting first would cost
  another 86 GB of quota to gain nothing. On a user account with a disk limit
  this is usually the right choice.
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


class ImageSource:
    """Where source frames come from.

    Three cases, because the right one depends on what the machine has:

    * **Network** — fetch individually over HTTPS. Pulls only what a build needs
      (a few thousand of 245,529 frames) but requires egress, which a shared
      research box often lacks.
    * **Directory** — the archive already extracted on disk.
    * **Archive** — read members straight out of the 86 GB zip. A zip carries a
      central directory, so individual members are random-access; extracting
      first would cost another 86 GB of quota to gain nothing.

    The archive case is the one worth having on a user account with a disk
    quota, and it is why this abstraction exists at all.
    """

    def open(self, crop: Crop) -> bytes | None:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


class NetworkSource(ImageSource):
    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout

    def open(self, crop: Crop) -> bytes | None:
        try:
            with urllib.request.urlopen(crop.url(), timeout=self.timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None

    def describe(self) -> str:
        return f"network ({IMAGE_BASE})"


class DirectorySource(ImageSource):
    """An extracted copy. Tolerant about how deeply it was unpacked.

    The archive may unpack as ``images/loc-.../000/000.jpg`` or with the
    dataset name prefixed, depending on the tool used, so the relative path is
    tried against a few plausible roots rather than assuming one.
    """

    PREFIXES = (
        "",
        "images",
        "channel-islands-camera-traps/images",
        "channel-islands-camera-traps",
    )

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.prefix: str | None = None

    def _resolve(self, crop: Crop) -> Path | None:
        if self.prefix is not None:
            candidate = self.root / self.prefix / crop.file_name
            return candidate if candidate.exists() else None
        for prefix in self.PREFIXES:
            candidate = self.root / prefix / crop.file_name
            if candidate.exists():
                self.prefix = prefix       # every later lookup uses the same root
                return candidate
        return None

    def open(self, crop: Crop) -> bytes | None:
        path = self._resolve(crop)
        if path is None:
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def describe(self) -> str:
        return f"directory ({self.root})"


class ArchiveSource(ImageSource):
    """Read members directly from the downloaded zip, without extracting."""

    def __init__(self, archive: Path) -> None:
        self.archive = Path(archive)
        self._zip = zipfile.ZipFile(self.archive)
        # Map the metadata's relative path to the member name, whatever prefix
        # the archive happens to use.
        self._members: dict[str, str] = {}
        for name in self._zip.namelist():
            if name.endswith("/"):
                continue
            key = name.split("images/", 1)[-1] if "images/" in name else name
            self._members[key] = name

    def open(self, crop: Crop) -> bytes | None:
        member = self._members.get(crop.file_name)
        if member is None:
            return None
        try:
            with self._zip.open(member) as handle:
                return handle.read()
        except (KeyError, OSError, zipfile.BadZipFile):
            return None

    def describe(self) -> str:
        return f"archive ({self.archive.name}, {len(self._members):,} members)"


def make_source(
    images_dir: Path | None = None, archive: Path | None = None
) -> ImageSource:
    """Pick a source, preferring local data over the network."""
    if archive:
        return ArchiveSource(archive)
    if images_dir:
        return DirectorySource(images_dir)
    return NetworkSource()


def fetch(
    crop: Crop,
    cache_dir: Path,
    timeout: float = 60.0,
    source: ImageSource | None = None,
) -> Path | None:
    """Materialise one source image in the cache, or return it if already there.

    Returns None rather than raising when a frame cannot be obtained: a build
    pulling thousands of images should degrade to a smaller dataset, not abort
    at 90%. Human frames are a routine case — they 404 by design.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / crop.local_name()
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    data = (source or NetworkSource(timeout)).open(crop)
    if not data:
        return None
    dest.write_bytes(data)
    return dest


def fetch_many(
    crops: list[Crop],
    cache_dir: Path,
    on_progress: Callable[[int, int, int], None] | None = None,
    source: ImageSource | None = None,
) -> Iterator[tuple[Crop, Path]]:
    """Yield (crop, path) for every image that could be obtained."""
    source = source or NetworkSource()
    ok = 0
    for i, crop in enumerate(crops, 1):
        path = fetch(crop, cache_dir, source=source)
        if path is not None:
            ok += 1
            yield crop, path
        if on_progress:
            on_progress(i, len(crops), ok)
