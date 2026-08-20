"""End-to-end regression tests against synthetic footage.

These are the guard rail for tomorrow's GPU work. The unit tests in
``test_pipeline.py`` check individual stages; these check that the whole cascade
still produces the same *answers* — one event per rodent crossing, the swaying
plant rejected, the lighting change absorbed by the global-change guard.

Assertions are made against the fixture's own plan rather than against a magic
number, so a change to the fixture cannot silently weaken the test.

A small frame size is used deliberately: geometry scales with resolution, so the
rodent stays proportionally the same size and the test runs in seconds rather
than minutes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from expressvision.config import PipelineConfig
from expressvision.detect import MotionPassthrough
from expressvision.pipeline import CameraPipeline
from expressvision.synth import make_test_video, plan_fixture

FIXTURE_SECONDS = 24
FIXTURE_W, FIXTURE_H = 640, 360


@pytest.fixture(scope="module")
def fixture_clip(tmp_path_factory):
    """Render one clip for the whole module — generation dominates runtime."""
    path = tmp_path_factory.mktemp("synth") / "fixture.mp4"
    plan = plan_fixture(
        seconds=FIXTURE_SECONDS, width=FIXTURE_W, height=FIXTURE_H
    )
    make_test_video(path, plan)
    return path, plan


@pytest.fixture(scope="module")
def motion_run(fixture_clip, tmp_path_factory):
    path, plan = fixture_clip
    cfg = PipelineConfig.for_source(str(path), "cam-test")
    cfg.out_dir = tmp_path_factory.mktemp("out")
    cfg.event.write_clips = False      # the answers are what matter, not the media
    cfg.event.write_keyframes = False

    pipeline = CameraPipeline(cfg, cfg.cameras[0], store=None)
    stats = pipeline.run()
    return stats, pipeline.events, plan


# ------------------------------------------------------------------ the answers

def test_fixture_plan_has_crossings_to_find(fixture_clip):
    _, plan = fixture_clip
    assert len(plan.crossings) >= 2, "fixture must contain crossings to detect"


def test_one_event_per_rodent_crossing(motion_run):
    """The headline regression: the cascade finds every crossing and nothing else.

    If a GPU change breaks detection, tracking or validation, this is the test
    that fails.
    """
    stats, events, plan = motion_run
    assert stats.events == len(plan.crossings)
    assert len(events) == len(plan.crossings)


def test_events_line_up_with_the_crossings(motion_run):
    """Right count is not enough — they must be the right events.

    A tolerance is needed because a track is only confirmed after several
    detections, so an event starts slightly after the animal enters frame.
    """
    _, events, plan = motion_run
    starts = sorted(e.start_t_s for e in events)
    expected = [a for a, _ in plan.crossing_times_s()]

    for actual, want in zip(starts, expected):
        assert want - 0.5 <= actual <= want + 2.0, (
            f"event at {actual:.2f}s does not match crossing at {want:.2f}s"
        )


def test_keyframe_box_matches_the_frame_it_was_drawn_on(fixture_clip, tmp_path):
    """Regression: the keyframe is chosen at the track's clearest moment, so the
    box must come from that moment too.

    Using the track's final box instead put a sliver at the frame edge — where
    the animal exits — on top of a mid-track image. That geometry becomes a
    pre-annotation a human is told to trust, so a wrong box is worse than none.
    """
    import json

    path, _plan = fixture_clip
    cfg = PipelineConfig.for_source(str(path), "cam-key")
    cfg.out_dir = tmp_path
    cfg.event.write_clips = False

    pipeline = CameraPipeline(cfg, cfg.cameras[0], store=None)
    pipeline.run()
    assert pipeline.events

    for event in pipeline.events:
        sidecar = Path(event.keyframe_path).with_suffix(".json")
        x1, y1, x2, y2 = json.loads(sidecar.read_text(encoding="utf-8"))["box"]
        width, height = x2 - x1, y2 - y1

        assert width > 0 and height > 0
        # A clipped exit-frame sliver is a few pixels wide; a real observation
        # of the animal is not.
        assert width >= 10, f"box {width}x{height} looks like a clipped edge box"


def test_events_are_traverses_not_jitter(motion_run):
    """Every event should look like an animal crossing the frame: near-straight,
    covering most of the width."""
    _, events, plan = motion_run
    for event in events:
        assert event.straightness > 0.8
        assert event.displacement_px > plan.width * 0.5
        assert event.n_detections >= 10


def test_swaying_plant_is_rejected_as_confined(motion_run):
    """The plant moves in every frame and must never become an event."""
    stats, _, _ = motion_run
    assert stats.rejected.get("confined", 0) > 0


def test_lighting_change_trips_the_global_change_guard(motion_run):
    stats, _, _ = motion_run
    assert stats.global_change_frames > 0


def test_gate_discards_work_and_tiles_stay_bounded(motion_run):
    """Sizing assumptions the edge-node capacity estimate depends on."""
    stats, _, _ = motion_run
    assert stats.frames_gated < stats.frames_processed
    assert stats.tiles > 0
    tiles_per_gated = stats.tiles / stats.frames_gated
    assert tiles_per_gated < 3.0, "ROIs are fragmenting; check gate.merge_distance"


def test_no_tamper_false_positive_on_clean_footage(motion_run):
    """The fixture is in focus throughout; tamper detection must stay quiet."""
    stats, _, _ = motion_run
    assert stats.tamper_frames == 0


# ------------------------------------------------------- detector interchangeability

def test_motion_detector_is_the_pipeline_default(fixture_clip, tmp_path):
    """Guard the default: swapping detectors must be explicit, never incidental."""
    path, _ = fixture_clip
    cfg = PipelineConfig.for_source(str(path), "cam-default")
    cfg.out_dir = tmp_path
    cfg.event.write_clips = False
    cfg.event.write_keyframes = False

    pipeline = CameraPipeline(cfg, cfg.cameras[0], store=None)
    assert isinstance(pipeline.detector, MotionPassthrough)


def test_pipeline_accepts_an_injected_detector(fixture_clip, tmp_path):
    """The seam the GPU detector plugs into. If this breaks, MegaDetector and
    every future ONNX model break with it."""
    path, _ = fixture_clip

    class CountingDetector:
        name = "counting-stub"

        def __init__(self):
            self.calls = 0

        def detect(self, frame, tiles, rois, frame_idx):
            self.calls += 1
            return []

    cfg = PipelineConfig.for_source(str(path), "cam-stub")
    cfg.out_dir = tmp_path
    cfg.event.write_clips = False
    cfg.event.write_keyframes = False

    stub = CountingDetector()
    pipeline = CameraPipeline(cfg, cfg.cameras[0], store=None, detector=stub)
    stats = pipeline.run(max_frames=120)

    assert stub.calls > 0, "detector was never called"
    assert stats.events == 0, "a detector returning nothing cannot produce events"


# -------------------------------------------------------------------- persistence

def test_events_and_rejections_reach_the_store(fixture_clip, tmp_path):
    """Rejections carry the features that train the plausibility classifier —
    losing them means paying for a second collection round."""
    from expressvision.store import Store

    path, plan = fixture_clip
    cfg = PipelineConfig.for_source(str(path), "cam-store")
    cfg.out_dir = tmp_path
    cfg.event.write_clips = False
    cfg.event.write_keyframes = False

    with Store(tmp_path / "test.db") as store:
        pipeline = CameraPipeline(cfg, cfg.cameras[0], store=store)
        pipeline.run()
        counts = store.counts()

        assert counts["events"] == len(plan.crossings)
        assert counts["unverified"] == len(plan.crossings)
        assert counts["discarded_tracks"] > 0

        row = store.conn.execute(
            "SELECT features_json FROM rejection LIMIT 1"
        ).fetchone()
        assert row and "extent_px" in row["features_json"]
