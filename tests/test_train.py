"""Tests for dataset loading and size-stratified evaluation.

No GPU, no torch, no transformers. The training loop itself cannot be
meaningfully unit-tested without a GPU and is exercised by running it; what
*can* be tested is everything that decides whether the resulting numbers mean
anything, and that is the part worth guarding:

* **Recall must be stratified by size, not averaged.** A detector that fails
  below 60 px while posting a respectable overall score is the exact failure
  measured on the third-party model. If the bands stop separating, the project
  loses the only metric that would catch it.
* **An empty band reports None, not zero.** The same unknown-versus-zero
  distinction the console enforces for camera uptime: a band with no examples
  has undefined recall, and calling it 0.0 both reads as total failure and
  drags any average down.
* **The pixel floor must require the bands above it to hold.** One band
  scraping past the threshold is luck, not a floor.
"""

from __future__ import annotations

import json

import pytest

from expressvision.train.data import Annotation, CocoSet, Frame
from expressvision.train.evaluate import (
    DETECT_FLOOR_PX,
    GroundTruth,
    Prediction,
    evaluate,
    iou,
    report,
)


# ------------------------------------------------------------------ loading
def _write_dataset(root, images, annotations, categories=("rodent",)):
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    payload = {
        "info": {"source_attribution": "test citation"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": n} for i, n in enumerate(categories)
        ],
    }
    (root / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_load_reads_boxes_and_target_sizes(tmp_path):
    root = _write_dataset(
        tmp_path / "ds",
        images=[{"id": 1, "file_name": "a.jpg", "width": 1920, "height": 1080}],
        annotations=[
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10, 20, 40, 20],
                "attributes": {"ground_truth": True, "target_long_axis_px": 44},
            }
        ],
    )
    coco = CocoSet.load(root)
    assert len(coco) == 1
    ann = coco.frames[0].annotations[0]
    assert ann.category == "rodent"
    assert ann.size_px == 44
    assert ann.xyxy() == (10, 20, 50, 40)
    assert coco.citation == "test citation"


def test_size_px_falls_back_to_the_box_when_unlabelled(tmp_path):
    """Real human-labelled data has no target_px; evaluation must still work."""
    root = _write_dataset(
        tmp_path / "ds",
        images=[{"id": 1, "file_name": "a.jpg", "width": 1920, "height": 1080}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 70, 30]}],
    )
    assert CocoSet.load(root).frames[0].annotations[0].size_px == 70


def test_frames_without_annotations_are_negatives(tmp_path):
    root = _write_dataset(
        tmp_path / "ds",
        images=[
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "b.jpg", "width": 640, "height": 480},
        ],
        annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 40, 20]}],
    )
    coco = CocoSet.load(root)
    assert sum(1 for f in coco if f.is_negative) == 1
    assert coco.class_counts()["(negative frames)"] == 1


def test_split_is_deterministic_and_disjoint():
    frames = [Frame(path=f"{i}.jpg", width=640, height=480) for i in range(100)]
    coco = CocoSet(frames, ["rodent"])
    a1, b1 = coco.split(val_frac=0.2, seed=5)
    a2, _ = coco.split(val_frac=0.2, seed=5)

    assert [f.path for f in a1] == [f.path for f in a2]
    assert len(b1) == 20
    assert not {f.path for f in a1} & {f.path for f in b1}


def test_size_profile_reports_the_operating_regime():
    frames = [
        Frame(
            path="x.jpg", width=1920, height=1080,
            annotations=[Annotation("rodent", 0, 0, px, px / 2, target_px=px)],
        )
        for px in [30, 35, 40, 45, 50, 120]
    ]
    profile = CocoSet(frames, ["rodent"]).size_profile()
    assert profile["n"] == 6
    assert profile["median"] <= 50
    assert profile["frac_at_or_below_40px"] == pytest.approx(3 / 6)


# --------------------------------------------------------------------- iou
def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


# -------------------------------------------------------------- evaluation
def _gt(size_px, label="rodent", box=(100, 100, 140, 120)):
    return GroundTruth(box=box, label=label, size_px=size_px)


def _pred(box=(100, 100, 140, 120), label="rodent", score=0.9):
    return Prediction(box=box, label=label, score=score)


def test_recall_is_reported_per_size_band():
    """The headline behaviour: a model good at large and bad at small."""
    frames = []
    for _ in range(10):
        frames.append(([_gt(36)], []))            # small: always missed
    for _ in range(10):
        frames.append(([_gt(150)], [_pred()]))    # large: always hit

    result = evaluate(frames)
    bands = {b.label: b.recall for b in result.bands}
    assert bands["32-48 px"] == 0.0
    assert bands["144+ px"] == pytest.approx(1.0)
    # The average would read 50% and conceal a detector that is useless at range.
    assert result.recall == pytest.approx(0.5)


def test_empty_band_reports_none_not_zero():
    result = evaluate([([_gt(150)], [_pred()])])
    empty = next(b for b in result.bands if b.label == "32-48 px")
    assert empty.total == 0
    assert empty.recall is None


def test_pixel_floor_requires_every_band_above_it_to_hold():
    """A single passing band surrounded by failures is not a floor."""
    frames = []
    for _ in range(10):
        frames.append(([_gt(40)], [_pred()]))   # 32-48 passes
    for _ in range(10):
        frames.append(([_gt(70)], []))          # 64-96 fails

    result = evaluate(frames)
    assert result.pixel_floor(min_recall=0.85) is None


def test_pixel_floor_found_when_all_larger_bands_hold():
    frames = []
    for _ in range(10):
        frames.append(([_gt(20)], []))          # below floor: fails
    for size in (40, 70, 120, 200):
        for _ in range(10):
            frames.append(([_gt(size)], [_pred()]))

    result = evaluate(frames)
    assert result.pixel_floor(min_recall=0.85) == 32


def test_verdict_names_the_camera_cost_of_missing_the_target():
    """The consequence a client cares about, not the raw number."""
    frames = []
    for _ in range(10):
        frames.append(([_gt(40)], []))
    for size in (100, 150, 200):
        for _ in range(10):
            frames.append(([_gt(size)], [_pred()]))

    result = evaluate(frames)
    floor = result.pixel_floor()
    assert floor is not None and floor > DETECT_FLOOR_PX
    assert "more cameras" in result.verdict()


def test_verdict_confirms_when_the_design_target_is_met():
    frames = [([_gt(size)], [_pred()]) for size in (36, 40, 60, 100, 200)] * 4
    result = evaluate(frames)
    assert "meets the 40 px design target" in result.verdict()


def test_unmatched_predictions_count_as_false_positives():
    frames = [([], [_pred(box=(10, 10, 50, 30))])]
    result = evaluate(frames)
    assert result.false_positives == 1
    assert result.negative_frames == 1
    assert result.fp_on_negatives == 1
    assert result.fp_per_frame == pytest.approx(1.0)


def test_low_scoring_predictions_are_ignored():
    frames = [([_gt(150)], [_pred(score=0.1)])]
    result = evaluate(frames, score_threshold=0.35)
    assert result.matched == 0
    assert result.missed == 1
    assert result.false_positives == 0


def test_wrong_class_does_not_match():
    """A bird detected where a rodent is, is a miss and a false positive.

    This is the failure the evaluated third-party model had — it called a pigeon
    a rodent at 0.729 — so the metric must not quietly forgive it.
    """
    frames = [([_gt(150, label="rodent")], [_pred(label="bird")])]
    result = evaluate(frames)
    assert result.matched == 0
    assert result.missed == 1
    assert result.false_positives == 1


def test_class_agnostic_mode_forgives_the_label():
    frames = [([_gt(150, label="rodent")], [_pred(label="bird")])]
    result = evaluate(frames, class_agnostic=True)
    assert result.matched == 1


def test_one_prediction_cannot_satisfy_two_objects():
    truths = [_gt(150, box=(100, 100, 140, 120)), _gt(150, box=(102, 102, 142, 122))]
    frames = [(truths, [_pred(box=(100, 100, 140, 120))])]
    result = evaluate(frames)
    assert result.matched == 1
    assert result.missed == 1


def test_report_renders_the_size_table_and_a_verdict():
    frames = [([_gt(size)], [_pred()]) for size in (36, 40, 60, 100, 200)] * 3
    text = report(evaluate(frames))
    assert "Recall by target size" in text
    assert "design floor" in text
    assert "32-48 px" in text
