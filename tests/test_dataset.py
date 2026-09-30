"""Tests for training-data acquisition and synthesis.

These run offline. Network fetching is exercised by the CLI, not here — a test
suite that needs 86 GB of camera-trap imagery to pass is a test suite nobody
runs.

The properties worth guarding are the ones that would silently poison a training
set rather than crash:

* **Never upscale.** Enlarging a small crop invents sensor detail that will not
  exist at inference, and nothing downstream would notice.
* **Labels must match pixels.** The compositor's whole value is that its boxes
  are exact by construction; a box that drifts from the animal is worse than a
  human-drawn one, because nobody thinks to check it.
* **The size distribution must land on the decision boundary.** If it drifts
  back toward the source corpus's 267 px median, we have rebuilt the very bias
  this package exists to correct.
"""

from __future__ import annotations

import json
import random

import cv2
import numpy as np
import pytest

from expressvision.dataset.compose import (
    DETECT_FLOOR_PX,
    ComposeConfig,
    Composer,
    is_night_ir,
    motion_blur,
    paste,
    sample_target_px,
    to_night_grey,
    write_coco,
)
from expressvision.dataset.lila import (
    BANNER_TOP_FRAC,
    ArchiveSource,
    Crop,
    DirectorySource,
    LilaIndex,
    NetworkSource,
    fetch,
    make_source,
)


# --------------------------------------------------------------------- lila
def _crop(category: str, w: int, h: int, name: str = "loc-a/000/001.jpg") -> Crop:
    return Crop(
        file_name=name, category=category, x=10, y=20, w=w, h=h,
        img_w=1920, img_h=1080,
    )


def test_index_separates_empties_from_objects(tmp_path):
    """Full-frame boxes are backgrounds, not animals."""
    meta = {
        "images": [
            {"id": "i1", "file_name": "a.jpg", "width": 1920, "height": 1080},
            {"id": "i2", "file_name": "b.jpg", "width": 1920, "height": 1080},
        ],
        "annotations": [
            {"image_id": "i1", "category_id": 4, "bbox": [10, 20, 200, 120]},
            {"image_id": "i2", "category_id": 0, "bbox": [0, 0, 1919, 1079]},
        ],
        "categories": [{"id": 4, "name": "rodent"}, {"id": 0, "name": "empty"}],
    }
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")

    index = LilaIndex.build(path)
    assert len(index.crops) == 1
    assert index.crops[0].category == "rodent"
    assert len(index.empties) == 1


def test_index_drops_frame_filling_object_boxes(tmp_path):
    """A box covering the whole frame is a labelling artefact, not a subject."""
    meta = {
        "images": [{"id": "i1", "file_name": "a.jpg", "width": 1920, "height": 1080}],
        "annotations": [{"image_id": "i1", "category_id": 4, "bbox": [0, 0, 1920, 1080]}],
        "categories": [{"id": 4, "name": "rodent"}],
    }
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert LilaIndex.build(path).crops == []


def test_select_respects_the_source_size_floor():
    """Only crops large enough to downscale from are eligible."""
    index = LilaIndex(
        crops=[_crop("rodent", 300, 150), _crop("rodent", 40, 20)],
        empties=[],
    )
    picked = index.select("rodent", min_px=90)
    assert len(picked) == 1
    assert picked[0].long_axis == 300


def test_index_round_trips_through_cache(tmp_path):
    index = LilaIndex(crops=[_crop("rodent", 300, 150)], empties=[_crop("empty", 1919, 1079)])
    cache = tmp_path / "index.json"
    index.save(cache)
    back = LilaIndex.load(cache)
    assert back.crops[0].long_axis == 300
    assert back.summary()["rodent"] == 1


# ------------------------------------------------------------------ scaling
def test_target_sizes_concentrate_on_the_detection_floor():
    """The distribution must favour where the decision actually happens.

    Uniform sampling across the range would spend most examples where detection
    is easy. The architecture's floor is 40 px, so that band needs the mass.
    """
    cfg = ComposeConfig()
    rng = random.Random(0)
    sizes = [sample_target_px(cfg, rng) for _ in range(4000)]

    near_floor = sum(1 for s in sizes if abs(s - DETECT_FLOOR_PX) <= 12) / len(sizes)
    assert near_floor > 0.35, f"only {near_floor:.0%} of samples near the floor"
    assert min(sizes) >= cfg.min_px
    assert max(sizes) <= cfg.max_px

    median = sorted(sizes)[len(sizes) // 2]
    assert median < 80, (
        f"median {median}px has drifted toward the source corpus bias this "
        f"package exists to correct"
    )


def test_paste_refuses_to_upscale():
    """Enlarging invents detail the sensor never captured."""
    background = np.full((1080, 1920, 3), 60, np.uint8)
    patch = np.full((20, 20, 3), 200, np.uint8)
    result = paste(background, patch, target_px=120, rng=random.Random(1), cfg=ComposeConfig())
    assert result is None


def test_paste_places_a_box_of_the_requested_size():
    background = np.full((1080, 1920, 3), 60, np.uint8)
    patch = np.full((300, 300, 3), 200, np.uint8)
    box = paste(background, patch, target_px=40, rng=random.Random(2), cfg=ComposeConfig())
    assert box is not None
    _, _, w, h = box
    # The label excludes the blending padding, so it sits just under the target.
    assert 30 <= max(w, h) <= 40


def test_paste_keeps_clear_of_the_camera_banner():
    """A rodent must never be composited onto the burnt-in timestamp strip."""
    cfg = ComposeConfig()
    background = np.full((1080, 1920, 3), 60, np.uint8)
    patch = np.full((300, 300, 3), 200, np.uint8)
    for seed in range(40):
        box = paste(background, patch, 60, random.Random(seed), cfg)
        if box is None:
            continue
        _, y, _, h = box
        assert y >= int(1080 * BANNER_TOP_FRAC)
        assert y + h <= 1080


def test_paste_actually_changes_the_background():
    background = np.full((1080, 1920, 3), 60, np.uint8)
    before = background.copy()
    patch = np.full((300, 300, 3), 220, np.uint8)
    assert paste(background, patch, 60, random.Random(3), ComposeConfig()) is not None
    assert not np.array_equal(before, background)


# ------------------------------------------------------------- night / IR
def test_is_night_ir_separates_greyscale_from_colour():
    grey = np.dstack([np.full((64, 64), 90, np.uint8)] * 3)
    assert is_night_ir(grey)

    colour = np.zeros((64, 64, 3), np.uint8)
    colour[:, :, 2] = 200  # strong red channel
    colour[:, :, 1] = 60
    assert not is_night_ir(colour)


def test_to_night_grey_removes_all_channel_separation():
    colour = np.zeros((32, 32, 3), np.uint8)
    colour[:, :, 0], colour[:, :, 1], colour[:, :, 2] = 20, 120, 220
    out = to_night_grey(colour)
    assert out.shape == colour.shape
    assert np.array_equal(out[:, :, 0], out[:, :, 2])
    assert is_night_ir(out)


def test_motion_blur_smears_without_shifting_brightness():
    """Blur redistributes light; it must not add or remove any."""
    img = np.zeros((64, 64, 3), np.uint8)
    img[30:34, 30:34] = 255
    blurred = motion_blur(img, length=9, angle_deg=0)
    assert blurred.std() < img.std()
    assert abs(float(blurred.mean()) - float(img.mean())) < 2.0


# ------------------------------------------------------------------- build
@pytest.fixture()
def fake_corpus(tmp_path):
    """A night background and a source frame with a bright animal-shaped blob."""
    bg = tmp_path / "bg.jpg"
    noise = np.random.default_rng(3).normal(70, 6, (1080, 1920)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(bg), cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR))

    src = tmp_path / "src.jpg"
    frame = np.full((1080, 1920, 3), 60, np.uint8)
    cv2.ellipse(frame, (500, 600), (110, 55), 0, 0, 360, (185, 185, 185), -1)
    cv2.imwrite(str(src), frame)

    crop = Crop(
        file_name="loc-x/000/000.jpg", category="rodent",
        x=390, y=545, w=220, h=110, img_w=1920, img_h=1080,
    )
    return bg, [(crop, src)]


def test_composer_emits_labels_that_match_the_image(fake_corpus):
    bg, sources = fake_corpus
    cfg = ComposeConfig(negative_frac=0.0, seed=5)
    sample = Composer(cfg).build(bg, sources)

    assert sample is not None
    assert sample.boxes, "expected at least one animal"
    h, w = sample.image.shape[:2]
    for (x, y, bw, bh), px in zip(sample.boxes, sample.target_px):
        assert 0 <= x and 0 <= y
        assert x + bw <= w and y + bh <= h
        assert cfg.min_px <= px <= cfg.max_px
    assert is_night_ir(sample.image), "output must be night-IR monochrome"


def test_composer_produces_true_negatives(fake_corpus):
    """A detector needs empty frames from the same distribution."""
    bg, sources = fake_corpus
    cfg = ComposeConfig(negative_frac=1.0, seed=5)
    sample = Composer(cfg).build(bg, sources)
    assert sample is not None
    assert sample.boxes == []


def test_composer_is_deterministic(fake_corpus):
    bg, sources = fake_corpus
    a = Composer(ComposeConfig(seed=9)).build(bg, sources)
    b = Composer(ComposeConfig(seed=9)).build(bg, sources)
    assert a is not None and b is not None
    assert a.boxes == b.boxes
    assert a.target_px == b.target_px


def test_write_coco_marks_boxes_as_ground_truth(tmp_path, fake_corpus):
    """The distinction from review.py's pre-annotations must survive to disk."""
    bg, sources = fake_corpus
    composer = Composer(ComposeConfig(negative_frac=0.0, seed=4))
    samples = []
    for i in range(4):
        sample = composer.build(bg, sources)
        if sample is not None:
            samples.append((f"f{i}.jpg", sample))

    stats = write_coco(samples, tmp_path / "ds", ["rodent"], "test citation")
    assert stats["images"] == len(samples)

    payload = json.loads((tmp_path / "ds" / "annotations.json").read_text(encoding="utf-8"))
    assert "test citation" in payload["info"]["source_attribution"]
    for ann in payload["annotations"]:
        assert ann["attributes"]["ground_truth"] is True
        assert ann["attributes"]["target_long_axis_px"] > 0
    assert (tmp_path / "ds" / "images" / "f0.jpg").exists()


# ------------------------------------------------------------ image sources
def test_archive_source_reads_without_extracting(tmp_path):
    """The 86 GB zip is random-access; extracting would cost another 86 GB."""
    import zipfile

    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("channel-islands-camera-traps/images/loc-a/000/001.jpg", b"JPEGBYTES")

    source = ArchiveSource(archive)
    assert source.open(_crop("rodent", 200, 100, "loc-a/000/001.jpg")) == b"JPEGBYTES"
    assert source.open(_crop("rodent", 200, 100, "loc-a/000/999.jpg")) is None
    assert "1 members" in source.describe()


def test_archive_source_handles_a_flat_layout(tmp_path):
    """Different unzip tools produce different nesting; both must resolve."""
    import zipfile

    archive = tmp_path / "flat.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("loc-b/000/002.jpg", b"DATA")

    assert ArchiveSource(archive).open(_crop("rodent", 200, 100, "loc-b/000/002.jpg")) == b"DATA"


def test_directory_source_finds_the_right_nesting(tmp_path):
    root = tmp_path / "extracted"
    (root / "images" / "loc-c" / "000").mkdir(parents=True)
    (root / "images" / "loc-c" / "000" / "003.jpg").write_bytes(b"PIXELS")

    source = DirectorySource(root)
    assert source.open(_crop("rodent", 200, 100, "loc-c/000/003.jpg")) == b"PIXELS"
    # The resolved prefix is reused, so later lookups do not re-probe.
    assert source.prefix == "images"


def test_make_source_prefers_local_data_over_the_network(tmp_path):
    import zipfile

    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("images/x.jpg", b"1")

    assert isinstance(make_source(archive=archive), ArchiveSource)
    assert isinstance(make_source(images_dir=tmp_path), DirectorySource)
    assert isinstance(make_source(), NetworkSource)


def test_fetch_uses_the_supplied_source_and_caches(tmp_path):
    import zipfile

    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("images/loc-d/000/004.jpg", b"CACHED")

    crop = _crop("rodent", 200, 100, "loc-d/000/004.jpg")
    cache = tmp_path / "cache"
    path = fetch(crop, cache, source=ArchiveSource(archive))
    assert path is not None and path.read_bytes() == b"CACHED"

    # Second call must hit the cache, not the source.
    class Exploding(ArchiveSource):
        def open(self, crop):
            raise AssertionError("should not re-read a cached frame")

    assert fetch(crop, cache, source=Exploding(archive)) == path
