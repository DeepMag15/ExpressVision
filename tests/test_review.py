"""Tests for the review, labelling and GPU-probe machinery.

The review path matters because it is how operator judgements become training
data. A silent failure here does not break a run — it just means the system
never improves, which is much harder to notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from expressvision.assembler import Event
from expressvision.gpu import probe_environment, sample_gpu, torch_vram_mb
from expressvision.review import (
    REVIEW_LABELS,
    build_review_sheet,
    export_coco,
    import_verdicts,
)
from expressvision.store import Store


@pytest.fixture()
def populated(tmp_path):
    """A store with three events, each with a real keyframe and sidecar."""
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()

    store = Store(tmp_path / "test.db")
    store.start_run("run-1", "cam-001", "fixture.mp4", "{}")

    for n in range(3):
        img = np.full((360, 640, 3), 40, dtype=np.uint8)
        cv2.rectangle(img, (100 + n * 20, 150), (140 + n * 20, 175), (200, 200, 200), -1)
        path = keyframes / f"event{n}.jpg"
        cv2.imwrite(str(path), img)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "event_id": f"event{n}",
                    "box": [100 + n * 20, 150, 140 + n * 20, 175],
                    "polyline": [[100, 160], [140, 165]],
                    "label": "motion",
                }
            ),
            encoding="utf-8",
        )
        store.add_event(
            "run-1",
            Event(
                id=f"event{n}",
                camera_id="cam-001",
                site_id="site-001",
                track_id=n,
                label="motion",
                started_at=f"2026-08-13T22:{10 + n:02d}:00+00:00",
                start_t_s=float(n * 10),
                end_t_s=float(n * 10 + 3),
                duration_s=3.0,
                n_detections=20,
                displacement_px=800.0,
                straightness=0.95,
                heading_deg=90.0,
                origin_xy=(100.0, 160.0),
                terminus_xy=(900.0, 160.0),
                polyline=[(100.0, 160.0), (900.0, 160.0)],
                keyframe_path=str(path),
            ),
        )
    yield store, tmp_path
    store.close()


# ------------------------------------------------------------------ review sheet

def test_review_sheet_contains_every_pending_event(populated, tmp_path):
    store, _ = populated
    out = tmp_path / "sheet.html"
    stats = build_review_sheet(store, out)

    assert stats.total == 3
    assert stats.missing_keyframes == 0
    html = out.read_text(encoding="utf-8")
    for n in range(3):
        assert f'data-id="event{n}"' in html


def test_review_sheet_is_self_contained(populated, tmp_path):
    """It must survive being emailed to a technician with no filesystem access,
    so images are inlined rather than referenced."""
    store, _ = populated
    out = tmp_path / "sheet.html"
    build_review_sheet(store, out)
    html = out.read_text(encoding="utf-8")

    assert "data:image/jpeg;base64," in html
    assert "<script" in html and "src=" not in html.split("<script")[1][:200]


def test_review_sheet_excludes_already_judged_events(populated, tmp_path):
    store, _ = populated
    store.set_verdict("event0", "confirmed")

    stats = build_review_sheet(store, tmp_path / "sheet.html")
    assert stats.total == 2

    stats_all = build_review_sheet(
        store, tmp_path / "all.html", include_verified=True
    )
    assert stats_all.total == 3


def test_review_sheet_offers_every_taxonomy_label(populated, tmp_path):
    store, _ = populated
    out = tmp_path / "sheet.html"
    build_review_sheet(store, out)
    html = out.read_text(encoding="utf-8")
    for name, _key in REVIEW_LABELS:
        assert name in html


def test_review_sheet_links_clips_stored_as_relative_paths(populated, tmp_path, monkeypatch):
    """Regression: clip paths are stored as written, which is normally relative
    to the working directory, and a relative path cannot become a file:// URI."""
    store, _ = populated
    monkeypatch.chdir(tmp_path)

    clip_dir = tmp_path / "out" / "clips"
    clip_dir.mkdir(parents=True)
    (clip_dir / "event0.mp4").write_bytes(b"not really a video")
    store.conn.execute(
        "UPDATE event SET clip_path = ? WHERE id = 'event0'",
        (str(Path("out") / "clips" / "event0.mp4"),),
    )
    store.conn.commit()

    out = tmp_path / "sheet.html"
    build_review_sheet(store, out)          # must not raise
    assert "file:///" in out.read_text(encoding="utf-8")


def test_review_sheet_tolerates_a_clip_path_that_no_longer_exists(populated, tmp_path):
    store, _ = populated
    store.conn.execute(
        "UPDATE event SET clip_path = '/gone/missing.mp4' WHERE id = 'event0'"
    )
    store.conn.commit()

    out = tmp_path / "sheet.html"
    build_review_sheet(store, out)
    assert "clip missing" in out.read_text(encoding="utf-8")


def test_review_sheet_handles_missing_keyframes(populated, tmp_path):
    store, _ = populated
    store.conn.execute("UPDATE event SET keyframe_path = NULL WHERE id = 'event1'")
    store.conn.commit()

    stats = build_review_sheet(store, tmp_path / "sheet.html")
    assert stats.missing_keyframes == 1
    assert stats.total == 3          # still listed, just without a picture


# --------------------------------------------------------------------- verdicts

def test_import_verdicts_applies_them(populated, tmp_path):
    store, _ = populated
    path = tmp_path / "verdicts.json"
    path.write_text(
        json.dumps(
            {
                "verdicts": [
                    {"event_id": "event0", "verdict": "confirmed"},
                    {"event_id": "event1", "verdict": "rejected"},
                    {"event_id": "event2", "verdict": "reclassified",
                     "corrected_label": "rodent"},
                ]
            }
        ),
        encoding="utf-8",
    )

    applied, skipped = import_verdicts(store, path)
    assert (applied, skipped) == (3, 0)

    counts = store.counts()
    assert counts["confirmed"] == 1
    assert counts["rejected"] == 1
    assert counts["unverified"] == 0

    row = store.conn.execute(
        "SELECT corrected_label FROM event WHERE id = 'event2'"
    ).fetchone()
    assert row["corrected_label"] == "rodent"


def test_import_verdicts_skips_unknown_ids_and_bad_verdicts(populated, tmp_path):
    store, _ = populated
    path = tmp_path / "verdicts.json"
    path.write_text(
        json.dumps(
            {
                "verdicts": [
                    {"event_id": "nonexistent", "verdict": "confirmed"},
                    {"event_id": "event0", "verdict": "banana"},
                    {"event_id": "event1", "verdict": "confirmed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    applied, skipped = import_verdicts(store, path)
    assert applied == 1
    assert skipped == 2


# ------------------------------------------------------------------ coco export

def test_coco_export_only_includes_judged_events_by_default(populated, tmp_path):
    store, _ = populated
    store.set_verdict("event0", "confirmed")
    store.set_verdict("event1", "reclassified", "rodent")
    store.set_verdict("event2", "rejected")

    result = export_coco(store, tmp_path / "dataset")
    assert result["images"] == 2          # rejected events are not training positives


def test_coco_export_is_valid_and_marked_as_pre_annotation(populated, tmp_path):
    store, _ = populated
    store.set_verdict("event0", "reclassified", "rodent")

    out = tmp_path / "dataset"
    export_coco(store, out)
    coco = json.loads((out / "annotations.json").read_text(encoding="utf-8"))

    assert {"images", "annotations", "categories", "info"} <= set(coco)
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1

    image, annotation = coco["images"][0], coco["annotations"][0]
    assert image["width"] == 640 and image["height"] == 360
    assert annotation["image_id"] == image["id"]

    _x, _y, w, h = annotation["bbox"]
    assert w > 0 and h > 0
    assert annotation["area"] == pytest.approx(w * h)
    # A reviewer must never mistake these for verified ground truth.
    assert annotation["attributes"]["pre_annotation"] is True

    names = {c["name"]: c["id"] for c in coco["categories"]}
    assert annotation["category_id"] == names["rodent"]


def test_coco_export_copies_the_images(populated, tmp_path):
    store, _ = populated
    store.set_verdict("event0", "confirmed")

    out = tmp_path / "dataset"
    export_coco(store, out)
    copied = list((out / "images").glob("*.jpg"))
    assert len(copied) == 1
    assert cv2.imread(str(copied[0])) is not None


def test_coco_export_with_nothing_confirmed_is_empty_not_broken(populated, tmp_path):
    store, _ = populated
    result = export_coco(store, tmp_path / "dataset")
    assert result["images"] == 0
    coco = json.loads((tmp_path / "dataset" / "annotations.json").read_text())
    assert coco["images"] == []
    assert coco["categories"], "categories should exist even with no images"


# --------------------------------------------------------------------- gpu probe

def test_probe_environment_never_raises():
    """Runs on machines with no torch, no CUDA and no NVML — it must report,
    not explode."""
    env = probe_environment()
    assert env.python
    assert isinstance(env.cuda_available, bool)
    assert isinstance(env.problems, list)


def test_probe_flags_a_cpu_only_torch_build():
    """The exact failure we must not hit tomorrow: CPU torch on a GPU box,
    silently producing CPU numbers."""
    env = probe_environment()
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    if torch.version.cuda is None:
        assert not env.gpu_ready
        assert any("CPU-only" in p for p in env.problems)


def test_runtime_probe_reports_a_real_device_not_unknown():
    """The guard on tomorrow's benchmark.

    Ultralytics wraps the network in AutoBackend, whose ``parameters()`` yields
    nothing — the obvious probe silently reports "unknown", and a benchmark that
    cannot say which device it used cannot be trusted to say a GPU was involved.
    """
    pytest.importorskip("torch")
    pytest.importorskip("PytorchWildlife")

    from expressvision.detect import MegaDetectorAdapter

    model = MegaDetectorAdapter(version="MDV6-yolov9-c", image_size=640)
    model.warmup(1)
    info = model.runtime_info()

    assert info["actual_device"] not in {"unknown", "not loaded"}
    assert info["actual_dtype"] in {"float32", "float16"}
    # Whatever we resolved to must be where the weights actually are.
    assert info["actual_device"].split(":")[0] == info["resolved_device"].split(":")[0]


def test_gpu_sampling_degrades_gracefully():
    snapshot = sample_gpu()
    for value in (snapshot.utilisation_pct, snapshot.memory_used_mb):
        assert value is None or value >= 0

    allocated, peak = torch_vram_mb()
    assert allocated is None or allocated >= 0
    assert peak is None or peak >= 0


# ---------------------------------------------------------- multi-GPU probe
def test_device_busy_is_judged_on_memory_not_utilisation():
    """A job between batches shows 0% but still holds its allocation.

    Picking that GPU because it looked idle would collide the moment it resumes,
    which is exactly the failure mode on a shared DGX with no scheduler.
    """
    from expressvision.gpu import DeviceInfo

    paused = DeviceInfo(index=0, used_mb=23596, utilisation_pct=0.0)
    assert paused.is_busy

    idle = DeviceInfo(index=1, used_mb=4, utilisation_pct=0.0)
    assert not idle.is_busy


def test_bf16_requires_ampere():
    """Volta and Turing do fp16 only; an A100 recipe copied here would fail."""
    from expressvision.gpu import DeviceInfo

    assert not DeviceInfo(index=0, capability="7.0").supports_bf16   # V100
    assert not DeviceInfo(index=0, capability="7.5").supports_bf16   # T4
    assert DeviceInfo(index=0, capability="8.0").supports_bf16       # A100
    assert DeviceInfo(index=0, capability="9.0").supports_bf16       # H100


def test_recommended_devices_skips_the_occupied_one():
    from expressvision.gpu import DeviceInfo, EnvReport

    report = EnvReport(
        cuda_available=True,
        device_count=4,
        devices=[
            DeviceInfo(index=0, used_mb=23596),
            DeviceInfo(index=1, used_mb=4),
            DeviceInfo(index=2, used_mb=4),
            DeviceInfo(index=3, used_mb=4),
        ],
    )
    assert report.recommended_devices() == "1,2,3"
    assert len(report.free_devices()) == 3


def test_no_recommendation_when_every_device_is_taken():
    from expressvision.gpu import DeviceInfo, EnvReport

    report = EnvReport(
        cuda_available=True,
        device_count=2,
        devices=[DeviceInfo(index=i, used_mb=20000) for i in range(2)],
    )
    assert report.recommended_devices() is None
