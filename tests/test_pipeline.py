"""Unit tests for the cascade's load-bearing logic.

Focused on the parts where a silent bug would corrupt every downstream number:
coordinate mapping through the tiler, ROI merging, track association, and the
validator's rejection rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from expressvision.assembler import simplify
from expressvision.config import TileConfig, TrackConfig, ValidatorConfig
from expressvision.detect import (
    MEGADETECTOR_DEFAULT,
    MEGADETECTOR_LABELS,
    MEGADETECTOR_VARIANTS,
    MegaDetectorAdapter,
    MotionPassthrough,
    build_detector,
)
from expressvision.gate import merge_nearby
from expressvision.tiling import Tiler, map_to_frame, nms
from expressvision.tracking import Tracker
from expressvision.types import Box, Detection, Track, TrackPoint
from expressvision.validator import TrackValidator

# --------------------------------------------------------------------- geometry

def test_iou_identical_boxes_is_one():
    box = Box(10, 10, 50, 50)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert Box(0, 0, 10, 10).iou(Box(100, 100, 110, 110)) == 0.0


def test_gap_is_zero_when_overlapping():
    assert Box(0, 0, 20, 20).gap_to(Box(10, 10, 30, 30)) == 0.0


def test_gap_measures_edge_distance():
    assert Box(0, 0, 10, 10).gap_to(Box(20, 0, 30, 10)) == pytest.approx(10.0)


# ------------------------------------------------------------------ roi merging

def test_merge_nearby_unions_fragments_of_one_animal():
    # A rat's body and tail, separated by a dark patch of floor.
    boxes = [Box(100, 100, 120, 112), Box(126, 104, 140, 114)]
    merged = merge_nearby(boxes, max_gap=24)
    assert len(merged) == 1
    assert merged[0] == Box(100, 100, 140, 114)


def test_merge_nearby_leaves_distant_boxes_alone():
    boxes = [Box(0, 0, 20, 20), Box(500, 500, 520, 520)]
    assert len(merge_nearby(boxes, max_gap=24)) == 2


def test_merge_nearby_is_transitive():
    # A-B and B-C are close; A-C is not. All three must still collapse to one.
    boxes = [Box(0, 0, 10, 10), Box(20, 0, 30, 10), Box(40, 0, 50, 10)]
    assert len(merge_nearby(boxes, max_gap=12)) == 1


# ----------------------------------------------------------------------- tiling

def test_small_roi_gets_native_resolution_tile():
    """The central claim of stage C: a small target is never downscaled."""
    tiler = Tiler(TileConfig(tile_size=640), 1920, 1080)
    tiles = tiler.plan([Box(900, 500, 940, 520)])
    assert len(tiles) == 1
    assert tiles[0].size == 640  # 1:1 with the source — no resize on crop


def test_oversized_roi_expands_the_tile_rather_than_cropping():
    tiler = Tiler(TileConfig(tile_size=640), 1920, 1080)
    tiles = tiler.plan([Box(100, 100, 1000, 900)])
    assert tiles[0].size > 640


def test_tiles_stay_inside_the_frame():
    tiler = Tiler(TileConfig(tile_size=640), 1920, 1080)
    for roi in (Box(0, 0, 20, 20), Box(1900, 1060, 1920, 1080)):
        tile = tiler.plan([roi])[0]
        assert 0 <= tile.x <= 1920 - tile.size
        assert 0 <= tile.y <= 1080 - tile.size


def test_nearby_rois_share_one_tile():
    """Two ROIs inside one tile must cost one inference, not two."""
    tiler = Tiler(TileConfig(tile_size=640), 1920, 1080)
    tiles = tiler.plan([Box(900, 500, 930, 520), Box(1000, 560, 1030, 580)])
    assert len(tiles) == 1
    assert sorted(tiles[0].roi_indices) == [0, 1]


def test_tile_budget_is_respected():
    tiler = Tiler(TileConfig(tile_size=320, max_tiles_per_frame=3), 1920, 1080)
    rois = [Box(x, 100, x + 20, 120) for x in range(0, 1800, 400)]
    assert len(tiler.plan(rois)) <= 3


def test_coordinates_round_trip_through_the_tiler():
    """A box found in tile space must map back to where it really is."""
    tiler = Tiler(TileConfig(tile_size=640), 1920, 1080)
    roi = Box(900, 500, 940, 520)
    tile = tiler.plan([roi])[0]

    local = Box(roi.x1 - tile.x, roi.y1 - tile.y, roi.x2 - tile.x, roi.y2 - tile.y)
    restored = tiler.to_frame_coords(tile, local)

    assert restored.x1 == pytest.approx(roi.x1, abs=1.0)
    assert restored.y1 == pytest.approx(roi.y1, abs=1.0)
    assert restored.x2 == pytest.approx(roi.x2, abs=1.0)


def test_crop_returns_detector_input_size():
    tiler = Tiler(TileConfig(tile_size=640), 1920, 1080)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for roi in (Box(900, 500, 940, 520), Box(100, 100, 1000, 900)):
        patch = tiler.crop(frame, tiler.plan([roi])[0])
        assert patch.shape[:2] == (640, 640)


# ---------------------------------------------------------------- passthrough

def test_passthrough_only_emits_rois_a_tile_covers():
    tiler = Tiler(TileConfig(tile_size=320, max_tiles_per_frame=1), 1920, 1080)
    rois = [Box(100, 100, 120, 120), Box(1700, 900, 1720, 920)]
    tiles = tiler.plan(rois)

    detections = MotionPassthrough().detect(
        np.zeros((1080, 1920, 3), dtype=np.uint8), tiles, rois, frame_idx=0
    )
    assert len(detections) == 1


# -------------------------------------------------------------- detector config

def test_unknown_variant_is_refused_with_the_available_list():
    with pytest.raises(ValueError, match="Unknown MegaDetector variant"):
        MegaDetectorAdapter(version="MDV6-apa-rtdetr-e")  # docs name, not in the package


def test_every_variant_constructs_without_loading_torch():
    """The adapter must be constructible where torch is absent — loading is
    lazy so the CLI and the test suite do not need the ml extra."""
    for variant in MEGADETECTOR_VARIANTS:
        model = MegaDetectorAdapter(version=variant)
        assert model._model is None
        assert model.name.endswith(variant)


def test_default_variant_is_real():
    """Regression: the published docs advertise variant names the released
    package rejects, so the default must be one that actually loads."""
    assert MEGADETECTOR_DEFAULT in MEGADETECTOR_VARIANTS


def test_class_mapping_matches_megadetector():
    """MegaDetector's own CLASS_NAMES are {0: animal, 1: person, 2: vehicle}."""
    assert MEGADETECTOR_LABELS[0] == "animal"
    assert MEGADETECTOR_LABELS[1] == "human"   # our taxonomy calls it human
    assert MEGADETECTOR_LABELS[2] == "vehicle"


def test_build_detector_factory():
    assert isinstance(build_detector("motion"), MotionPassthrough)
    with pytest.raises(ValueError):
        build_detector("nonsense")


# --------------------------------------------------------------------------- nms

def test_nms_suppresses_a_duplicate_from_an_overlapping_tile():
    """One animal near a shared tile boundary gets reported by both tiles."""
    kept = nms(
        [
            Detection(Box(100, 100, 160, 140), score=0.9, label="animal"),
            Detection(Box(104, 102, 164, 143), score=0.7, label="animal"),
        ]
    )
    assert len(kept) == 1
    assert kept[0].score == 0.9  # the more confident one survives


def test_nms_keeps_two_genuinely_separate_animals():
    kept = nms(
        [
            Detection(Box(100, 100, 160, 140), score=0.9),
            Detection(Box(900, 600, 960, 640), score=0.8),
        ]
    )
    assert len(kept) == 2


def test_map_to_frame_matches_the_tiler():
    """The detector and the tiler must agree on coordinates, or every box is
    subtly and silently in the wrong place."""
    tiler = Tiler(TileConfig(tile_size=640), 1920, 1080)
    tile = tiler.plan([Box(900, 500, 940, 520)])[0]
    local = Box(10, 20, 90, 100)
    assert map_to_frame(tile, local, 640, 1920, 1080) == tiler.to_frame_coords(tile, local)


# ---------------------------------------------------------------------- tracking

def _walk(tracker: Tracker, steps: int, dx: float = 12.0, start: float = 100.0):
    """Drive a tracker with one object moving steadily right."""
    for i in range(steps):
        x = start + i * dx
        tracker.update([Detection(Box(x, 300, x + 40, 320))], frame_idx=i, t_s=i / 15.0)


def test_tracker_keeps_one_id_for_one_object():
    tracker = Tracker(TrackConfig(), "cam-001")
    _walk(tracker, 20)
    assert tracker.created == 1
    assert len(tracker.tracks) == 1


def test_tracker_confirms_after_min_hits():
    tracker = Tracker(TrackConfig(min_hits=3), "cam-001")
    _walk(tracker, 5)
    assert tracker.tracks[0].confirmed


def test_tracker_bridges_a_short_occlusion():
    """A rat crossing behind a pallet leg must stay one track, not become two."""
    cfg = TrackConfig(max_age=30)
    tracker = Tracker(cfg, "cam-001")

    for i in range(10):
        x = 100 + i * 12
        tracker.update([Detection(Box(x, 300, x + 40, 320))], i, i / 15.0)
    for i in range(10, 18):                       # occluded — no detections
        tracker.update([], i, i / 15.0)
    for i in range(18, 28):                       # reappears, displaced
        x = 100 + i * 12
        tracker.update([Detection(Box(x, 300, x + 40, 320))], i, i / 15.0)

    assert tracker.created == 1


def test_tracker_closes_a_track_after_max_age():
    tracker = Tracker(TrackConfig(max_age=5, min_hits=2), "cam-001")
    _walk(tracker, 8)

    finished: list[Track] = []
    for i in range(8, 20):
        finished.extend(tracker.update([], i, i / 15.0))

    assert len(finished) == 1
    assert finished[0].confirmed


def test_tracker_follows_an_animal_that_outruns_its_own_bounding_box():
    """Regression: a small fast target moving further than its own width per
    frame has zero IoU between consecutive detections.

    This is the normal case for a rat on a low-frame-rate stream, not an edge
    case, and it used to fail silently — the gate fired, tiles were inferred,
    detections were produced, and no track ever formed.
    """
    tracker = Tracker(TrackConfig(), "cam-001")
    width, step = 16.0, 14.0            # travels ~0.9x its own width per frame

    for i in range(20):
        x = 100 + i * step
        tracker.update([Detection(Box(x, 300, x + width, 308))], i, i / 15.0)

    a = Box(100, 300, 100 + width, 308)
    b = Box(100 + step, 300, 100 + step + width, 308)
    assert a.iou(b) < 0.1, "fixture must actually have near-zero overlap"

    assert tracker.created == 1
    assert tracker.tracks[0].confirmed


def test_proximity_matching_does_not_join_unrelated_objects():
    """The distance fallback must not glue together things that are simply far
    apart — otherwise two pests become one track."""
    tracker = Tracker(TrackConfig(), "cam-001")
    for i in range(6):
        tracker.update([Detection(Box(100, 100, 120, 120))], i, i / 15.0)
    # Something appears far away, well beyond any plausible travel.
    tracker.update([Detection(Box(1500, 900, 1520, 920))], 6, 6 / 15.0)
    assert tracker.created == 2


def test_real_overlap_is_preferred_over_proximity():
    """When both are available, a genuine IoU match must win, so a nearby
    distractor cannot steal a track from the object actually overlapping it."""
    cfg = TrackConfig()
    tracker = Tracker(cfg, "cam-001")
    overlapping = Box(100, 100, 140, 140)
    nearby = Box(150, 100, 190, 140)
    predicted = Box(100, 100, 140, 140)

    assert tracker._affinity(predicted, overlapping) > tracker._affinity(predicted, nearby)


def test_two_objects_get_two_ids():
    tracker = Tracker(TrackConfig(), "cam-001")
    for i in range(12):
        tracker.update(
            [
                Detection(Box(100 + i * 10, 300, 140 + i * 10, 320)),
                Detection(Box(900 - i * 10, 700, 940 - i * 10, 720)),
            ],
            i,
            i / 15.0,
        )
    assert tracker.created == 2


# --------------------------------------------------------------------- validator

def _track_from(points: list[tuple[float, float]], area: float = 400.0) -> Track:
    track = Track(id=1, camera_id="cam-001", box=Box(0, 0, 20, 20))
    for i, (x, y) in enumerate(points):
        track.points.append(TrackPoint(i, i / 15.0, x, y, area, 1.0))
    track.start_frame, track.end_frame = 0, len(points) - 1
    track.start_t_s, track.end_t_s = 0.0, (len(points) - 1) / 15.0
    return track


def test_validator_accepts_a_purposeful_traverse():
    validator = TrackValidator(ValidatorConfig(), 1280, 720)
    track = _track_from([(100 + i * 25, 400) for i in range(15)])
    assert validator.judge(track).passed


def test_validator_rejects_swaying_vegetation():
    """The most common real-world false positive: motion that goes nowhere."""
    validator = TrackValidator(ValidatorConfig(), 1280, 720)
    track = _track_from(
        [(600 + 18 * np.sin(i * 0.6), 500 + 8 * np.cos(i * 0.6)) for i in range(40)]
    )
    verdict = validator.judge(track)
    assert not verdict.passed
    assert verdict.reason == "confined"


def test_validator_rejects_a_drifting_fragment_of_a_static_object():
    """Regression: a plant arm drifting one way for half a second looks
    purposeful by net displacement — 40 px of travel with high straightness —
    and this is exactly what produced every false positive on the fixture.
    Extent is what rejects it."""
    validator = TrackValidator(ValidatorConfig(), 1280, 720)
    track = _track_from([(1090 + i * 4, 545 - i * 2) for i in range(10)])

    assert track.net_displacement() > 40      # would have passed the old gate
    assert track.straightness() > 0.9         # and looked purposeful
    verdict = validator.judge(track)
    assert not verdict.passed
    assert verdict.reason == "confined"


def test_validator_accepts_an_animal_that_doubles_back():
    """A rat working along a wall and returning the way it came has near-zero
    net displacement and terrible straightness, but it is a real event. Extent
    is what keeps it."""
    validator = TrackValidator(ValidatorConfig(), 1280, 720)
    out = [(200 + i * 30, 400) for i in range(20)]
    back = [(790 - i * 30, 410) for i in range(20)]
    track = _track_from(out + back)

    assert track.net_displacement() < 60      # ended up back where it started
    assert track.straightness() < 0.1         # and wandered to get there
    assert validator.judge(track).passed


def test_validator_rejects_a_track_with_too_few_detections():
    validator = TrackValidator(ValidatorConfig(min_detections=5), 1280, 720)
    verdict = validator.judge(_track_from([(100, 100), (200, 200), (300, 300)]))
    assert not verdict.passed
    assert verdict.reason == "too_few_detections"


def test_validator_rejects_wildly_unstable_size():
    """An insect crossing close to the lens: one enormous blurred blob among
    otherwise small observations.

    Note this needs a genuine outlier, not an alternating pattern — a two-value
    area sequence caps out at a CV of 1.0 however extreme the two values are.
    """
    validator = TrackValidator(ValidatorConfig(), 1280, 720)
    track = _track_from([(100 + i * 30, 400) for i in range(15)], area=20.0)
    track.points[7].area = 20000.0

    verdict = validator.judge(track)
    assert not verdict.passed
    assert verdict.reason == "unstable_size"


def test_straightness_separates_traverse_from_oscillation():
    straight = _track_from([(100 + i * 25, 400) for i in range(15)])
    swaying = _track_from([(600 + 18 * np.sin(i * 0.6), 500) for i in range(40)])
    assert straight.straightness() > 0.9
    assert swaying.straightness() < 0.12


def test_features_are_complete_for_training():
    """Every rejected track must carry a full feature vector, or Phase 2 has to
    re-collect data to train the plausibility classifier."""
    validator = TrackValidator(ValidatorConfig(), 1280, 720)
    features = validator.features(_track_from([(100 + i * 25, 400) for i in range(15)]))
    expected = {
        "n_detections", "frame_span", "duration_s", "displacement_px",
        "displacement_frac", "path_length_px", "straightness", "area_cv",
        "mean_area_px", "mean_speed_px_s", "peak_speed_px_s", "speed_ratio",
        "displacement_per_s", "mean_score",
    }
    assert expected <= set(features)
    assert all(isinstance(v, float) for v in features.values())


# ------------------------------------------------------------------- trajectory

def test_simplify_collapses_a_straight_line():
    points = [(float(i), 0.0) for i in range(50)]
    assert len(simplify(points, epsilon=2.0)) == 2


def test_simplify_keeps_a_real_corner():
    points = [(float(i), 0.0) for i in range(20)] + [(19.0, float(i)) for i in range(1, 20)]
    simplified = simplify(points, epsilon=2.0)
    assert 3 <= len(simplified) <= 5


def test_simplify_preserves_endpoints():
    points = [(float(i), float(i % 7)) for i in range(40)]
    simplified = simplify(points, epsilon=1.0)
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]
