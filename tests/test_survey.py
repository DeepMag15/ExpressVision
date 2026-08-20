"""Tests for the camera assessment tool.

The survey answers the highest-rated risk in the project — can this camera see a
rodent at all — so a wrong answer here sends someone to the wrong site with the
wrong bill of materials. The optics maths in particular is worth pinning down,
because it is the part that turns into money.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from expressvision.survey import (
    DETECT_FLOOR_PX,
    IDENTIFY_FLOOR_PX,
    RODENT_M,
    Optics,
    SurveyResult,
    survey,
)

# ------------------------------------------------------------------- optics

def test_hfov_is_preferred_over_computed_lens_value():
    """A quoted field of view beats one derived from focal length, because
    datasheet sensor sizes are reported loosely."""
    optics = Optics(hfov_deg=78.0, lens_mm=4.0)
    assert optics.resolved_hfov() == 78.0


def test_hfov_computed_from_lens_when_not_given():
    optics = Optics(lens_mm=4.0, sensor='1/2.8"')
    hfov = optics.resolved_hfov()
    expected = math.degrees(2 * math.atan(5.37 / (2 * 4.0)))
    assert hfov == pytest.approx(expected, abs=0.1)


def test_wider_lens_gives_wider_field_of_view():
    wide = Optics(lens_mm=2.8).resolved_hfov()
    narrow = Optics(lens_mm=12.0).resolved_hfov()
    assert wide > narrow


def test_no_optics_means_no_range_calculation():
    assert Optics().resolved_hfov() is None


# -------------------------------------------------------------------- range

def _result(width: int = 1920, hfov: float = 78.0, **kwargs) -> SurveyResult:
    return SurveyResult(
        source="test", width=width, height=int(width * 9 / 16),
        optics=Optics(hfov_deg=hfov, **kwargs),
    )


def test_pixels_on_rodent_shrink_with_distance():
    r = _result()
    near = r.pixels_on_rodent_at(2.0)
    far = r.pixels_on_rodent_at(10.0)
    assert near > far
    # Inverse proportionality: five times the distance, a fifth the pixels.
    assert near / far == pytest.approx(5.0, rel=0.01)


def test_max_range_matches_pixels_on_target():
    """The two calculations must agree, or the report contradicts itself."""
    r = _result()
    limit = r.max_range_m(DETECT_FLOOR_PX)
    assert r.pixels_on_rodent_at(limit) == pytest.approx(DETECT_FLOOR_PX, rel=0.01)


def test_identify_range_is_half_the_detect_range():
    """80 px needs twice the pixel density of 40 px, so half the distance."""
    r = _result()
    assert r.max_range_m(DETECT_FLOOR_PX) / r.max_range_m(IDENTIFY_FLOOR_PX) == (
        pytest.approx(IDENTIFY_FLOOR_PX / DETECT_FLOOR_PX, rel=0.01)
    )


def test_higher_resolution_extends_range():
    assert _result(width=3840).max_range_m() > _result(width=1920).max_range_m()


def test_range_matches_the_architecture_table():
    """Regression against the published figures in architecture section 2.

    A 2 MP camera with a ~53 degree lens detects to about 12 m. If this drifts,
    the architecture document and the tool disagree, and someone specifies the
    wrong camera.
    """
    assert _result(width=1920, hfov=53.0).max_range_m() == pytest.approx(12.0, abs=0.6)


def test_range_is_none_without_optics():
    r = SurveyResult(source="test", width=1920, height=1080)
    assert r.max_range_m() is None
    assert r.pixels_on_rodent_at(5.0) is None


def test_rodent_size_assumption_is_explicit():
    """If this constant changes, every range figure changes with it."""
    assert RODENT_M == 0.25


# ------------------------------------------------------------------ verdicts

def _synthetic_clip(path, width=1920, height=1080, frames=40, night=True,
                    blur=False, noise=3.0, fps=15):
    """A short clip with controllable defects."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    rng = np.random.default_rng(4)
    for i in range(frames):
        base = np.full((height, width, 3), 70, dtype=np.uint8)
        if not night:
            base[:, :, 0] = 120     # give it colour so saturation is non-zero
            base[:, :, 2] = 40
        # Static texture, so sharpness is measurable.
        for x in range(0, width, 90):
            cv2.line(base, (x, 0), (x, height), (110, 110, 110), 2)

        # A moving object.
        obj = np.zeros_like(base)
        cx = int((i / frames) * width)
        cv2.rectangle(obj, (cx, 500), (cx + 70, 560), (200, 200, 200), -1)
        if blur:
            k = np.zeros((31, 31), np.float32)
            k[15, :] = 1 / 31
            obj = cv2.filter2D(obj, -1, k)
        base = np.clip(base.astype(np.int16) + obj.astype(np.int16), 0, 255).astype(np.uint8)

        base = np.clip(
            base.astype(np.int16) + rng.normal(0, noise, base.shape).astype(np.int16),
            0, 255,
        ).astype(np.uint8)
        writer.write(base)
    writer.release()
    return path


def test_survey_reads_basic_properties(tmp_path):
    clip = _synthetic_clip(tmp_path / "cam.mp4")
    r = survey(clip, sample_frames=20)
    assert r.error is None
    assert (r.width, r.height) == (1920, 1080)
    assert r.fps == pytest.approx(15, abs=1)
    assert r.frames_sampled > 0


def test_survey_detects_night_mode(tmp_path):
    night = survey(_synthetic_clip(tmp_path / "n.mp4", night=True), sample_frames=20)
    day = survey(_synthetic_clip(tmp_path / "d.mp4", night=False), sample_frames=20)
    assert night.night_fraction > 0.8
    assert day.night_fraction < 0.3


def test_survey_flags_motion_blur(tmp_path):
    """The slow-shutter signature: moving things smeared, background sharp.

    This is the single most common fixable defect on real CCTV, so it must not
    be missed.
    """
    sharp = survey(_synthetic_clip(tmp_path / "s.mp4", blur=False), sample_frames=30)
    blurred = survey(_synthetic_clip(tmp_path / "b.mp4", blur=True), sample_frames=30)
    assert blurred.blur_ratio < sharp.blur_ratio


def test_survey_flags_substream_resolution(tmp_path):
    """A sub-stream export must fail loudly — a rat in it is unrecoverable."""
    clip = _synthetic_clip(tmp_path / "sub.mp4", width=704, height=576)
    r = survey(clip, sample_frames=12)
    resolution = next(f for f in r.findings if f.name == "Resolution")
    assert resolution.status == "fail"
    assert "main" in resolution.fix.lower()
    assert r.verdict == "replace or relocate"


def test_survey_flags_a_camera_that_cannot_reach_the_far_wall(tmp_path):
    """The hardware verdict: detectable near, invisible far."""
    clip = _synthetic_clip(tmp_path / "cam.mp4")
    r = survey(clip, Optics(hfov_deg=90.0, furthest_floor_m=20.0), sample_frames=12)
    coverage = next(f for f in r.findings if f.name == "Coverage at furthest point")
    assert coverage.status == "fail"
    assert r.verdict == "replace or relocate"


def test_survey_passes_a_camera_with_adequate_reach(tmp_path):
    clip = _synthetic_clip(tmp_path / "cam.mp4")
    r = survey(clip, Optics(hfov_deg=40.0, furthest_floor_m=6.0), sample_frames=12)
    coverage = next(f for f in r.findings if f.name == "Coverage at furthest point")
    assert coverage.status == "ok"


def test_survey_reports_range_as_unavailable_without_optics(tmp_path):
    clip = _synthetic_clip(tmp_path / "cam.mp4")
    r = survey(clip, sample_frames=12)
    finding = next(f for f in r.findings if f.name == "Detection range")
    assert finding.status == "info"
    assert "hfov" in finding.note.lower()


def test_survey_handles_a_missing_file():
    r = survey("does/not/exist.mp4")
    assert r.error is not None
    assert r.verdict == "unreadable"


def test_every_failing_finding_offers_a_fix_or_explains_why_not(tmp_path):
    """A verdict without a next action is not useful to whoever gets the report.

    Range failures are the exception that proves the rule — they still carry a
    fix, just an expensive one.
    """
    clip = _synthetic_clip(tmp_path / "sub.mp4", width=704, height=576)
    r = survey(clip, Optics(hfov_deg=90.0, furthest_floor_m=20.0), sample_frames=12)
    for f in r.findings:
        if f.status == "fail":
            assert f.fix or f.note, f"{f.name} fails with no guidance"
