"""Tests for the console API.

Three groups of these matter more than the rest:

* **Coverage honesty.** A rate with no denominator must come back ``None``, not
  ``0``. The whole reason the architecture insists on uptime normalisation is
  that a camera which observed nothing looks identical to a camera which saw
  nothing, and ``0.0`` erases the difference at the one place a reviewer would
  notice it.
* **Media confinement.** Clip and keyframe paths are free text in the database.
  If a row can point the server at an arbitrary file, the console is a file-read
  primitive rather than an evidence viewer.
* **The verdict path.** This is the training-data return path. A verdict that
  reports success without reaching the store silently corrupts the training set,
  which is worse than an error.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="console API is an optional extra: pip install -e '.[web]'")

from fastapi.testclient import TestClient

from expressvision.api.app import create_app
from expressvision.assembler import Event
from expressvision.store import Store


def _event(event_id: str, *, camera: str = "cam-001", minute: int = 10, **kwargs) -> Event:
    defaults = {
        "camera_id": camera,
        "site_id": "site-001",
        "track_id": 1,
        "label": "motion",
        "started_at": f"2026-08-13T22:{minute:02d}:00+00:00",
        "start_t_s": 0.0,
        "end_t_s": 3.0,
        "duration_s": 3.0,
        "n_detections": 20,
        "displacement_px": 800.0,
        "straightness": 0.95,
        "heading_deg": 90.0,
        "origin_xy": (100.0, 160.0),
        "terminus_xy": (900.0, 200.0),
        "polyline": [(100.0, 160.0), (900.0, 200.0)],
        "features": {"extent_frac": 0.6},
    }
    defaults.update(kwargs)
    return Event(id=event_id, **defaults)  # type: ignore[arg-type]


@pytest.fixture()
def store_path(tmp_path):
    """Two cameras, one of which recorded no uptime on one of its runs."""
    media = tmp_path / "media" / "keyframes"
    media.mkdir(parents=True)

    db = tmp_path / "test.db"
    store = Store(db)

    store.start_run("run-1", "cam-001", "a.mp4", "{}")
    for n in range(3):
        img = np.full((360, 640, 3), 40, dtype=np.uint8)
        path = media / f"e{n}.jpg"
        cv2.imwrite(str(path), img)
        store.add_event("run-1", _event(f"e{n}", minute=10 + n, keyframe_path=str(path)))
    store.add_rejection("run-1", "cam-001", 7, "confined", 0.0, 4.0, 30, {"extent_frac": 0.02})
    store.finish_run("run-1", fps=15.0, observed_seconds=3600.0)

    store.set_verdict("e0", "confirmed")
    store.close()
    return db, tmp_path / "media"


@pytest.fixture()
def client(store_path):
    db, media = store_path
    return TestClient(create_app(db_path=db, media_root=media))


def test_health_reports_store_and_demo_flag(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["store_exists"] is True
    assert body["demo"] is False


def test_health_without_a_store_does_not_error(tmp_path):
    """The console must be able to explain an empty state, not just fail."""
    client = TestClient(create_app(db_path=tmp_path / "absent.db"))
    body = client.get("/api/health").json()
    assert body["store_exists"] is False
    # Every data endpoint says why rather than 500-ing.
    assert client.get("/api/overview").status_code == 503


def test_overview_normalises_by_observed_hours(client):
    body = client.get("/api/overview").json()
    assert body["counts"]["events"] == 3
    assert body["coverage"]["complete"] is True
    assert body["coverage"]["observed_hours"] == pytest.approx(1.0)
    # 3 events in one observed hour.
    assert body["events_per_camera_hour"] == pytest.approx(3.0)


def test_rate_is_none_not_zero_when_uptime_is_missing(store_path):
    """The load-bearing case.

    A run with no observed_seconds makes coverage incomplete, and every derived
    rate must then be null. Zero would be indistinguishable from a quiet camera,
    which is the exact false reassurance the architecture calls the most
    dangerous output this system can produce.
    """
    db, media = store_path
    store = Store(db)
    store.start_run("run-2", "cam-002", "b.mp4", "{}")
    store.conn.execute("UPDATE run SET observed_seconds = NULL WHERE id = 'run-2'")
    store.conn.commit()
    store.close()

    body = TestClient(create_app(db_path=db, media_root=media)).get("/api/overview").json()
    assert body["coverage"]["complete"] is False
    assert body["coverage"]["runs_missing_uptime"] == 1
    assert body["events_per_camera_hour"] is None


def test_events_filter_pending_and_paginate(client):
    pending = client.get("/api/events", params={"verdict": "pending"}).json()
    assert pending["total"] == 2
    assert all(e["verdict"] is None for e in pending["events"])

    confirmed = client.get("/api/events", params={"verdict": "confirmed"}).json()
    assert confirmed["total"] == 1

    page = client.get("/api/events", params={"limit": 1, "offset": 1}).json()
    assert page["total"] == 3
    assert len(page["events"]) == 1


def test_events_do_not_leak_filesystem_paths(client):
    """The feed says whether media exists, never where it lives on disk."""
    event = client.get("/api/events").json()["events"][0]
    assert event["has_keyframe"] is True
    assert "keyframe_path" not in event
    assert "clip_path" not in event


def test_verdict_round_trips_to_the_store(client, store_path):
    db, _ = store_path
    assert client.post("/api/events/e1/verdict", json={"verdict": "rejected"}).status_code == 200

    store = Store(db)
    row = store.conn.execute("SELECT verdict FROM event WHERE id = 'e1'").fetchone()
    store.close()
    assert row["verdict"] == "rejected"


def test_reclassify_requires_a_label(client):
    """Reclassified with no label carries no training signal, so it is refused."""
    assert client.post("/api/events/e1/verdict", json={"verdict": "reclassified"}).status_code == 422


def test_reclassify_rejects_an_unknown_label(client):
    response = client.post(
        "/api/events/e1/verdict", json={"verdict": "reclassified", "corrected_label": "dragon"}
    )
    assert response.status_code == 422


def test_verdict_on_a_missing_event_is_404(client):
    assert (
        client.post("/api/events/nope/verdict", json={"verdict": "confirmed"}).status_code == 404
    )


def test_keyframe_is_served(client):
    response = client.get("/api/events/e0/keyframe")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_media_outside_the_root_is_refused(tmp_path):
    """A database row must not be able to make the server read arbitrary files.

    Paths are written by the collector and could be edited by anyone with write
    access to the store, so the confinement check is at the HTTP boundary rather
    than trusted from the row.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("not for serving", encoding="utf-8")

    media = tmp_path / "media"
    media.mkdir()
    db = tmp_path / "t.db"

    store = Store(db)
    store.start_run("r", "cam-001", "a.mp4", "{}")
    store.add_event("r", _event("escape", keyframe_path=str(secret)))
    store.finish_run("r", fps=15.0, observed_seconds=60.0)
    store.close()

    response = TestClient(create_app(db_path=db, media_root=media)).get(
        "/api/events/escape/keyframe"
    )
    assert response.status_code == 403


def test_media_recorded_but_deleted_is_410_not_500(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    db = tmp_path / "t.db"

    store = Store(db)
    store.start_run("r", "cam-001", "a.mp4", "{}")
    store.add_event("r", _event("gone", keyframe_path=str(media / "vanished.jpg")))
    store.finish_run("r", fps=15.0, observed_seconds=60.0)
    store.close()

    response = TestClient(create_app(db_path=db, media_root=media)).get("/api/events/gone/keyframe")
    assert response.status_code == 410


def test_funnel_shares_are_relative_to_the_previous_stage(client, store_path):
    db, media = store_path
    store = Store(db)
    from expressvision.types import FunnelStats

    store.add_funnel(
        "run-1",
        "cam-001",
        FunnelStats(
            frames_read=1000,
            frames_processed=1000,
            frames_gated=40,
            rois=60,
            tiles=56,
            detections=50,
            tracks_created=10,
            tracks_confirmed=8,
            events=3,
        ),
    )
    store.close()

    body = TestClient(create_app(db_path=db, media_root=media)).get("/api/funnel").json()
    stages = {s["key"]: s for s in body["stages"]}
    assert stages["frames_gated"]["share_of_previous"] == pytest.approx(0.04)
    assert body["gate_pass_rate"] == pytest.approx(0.04)
    assert body["tiles_per_gated_frame"] == pytest.approx(1.4)
    assert body["inference_saving"] == pytest.approx(1000 / 56)


def test_cameras_report_per_camera_coverage(client):
    rows = client.get("/api/cameras").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["camera_id"] == "cam-001"
    assert row["events"] == 3
    assert row["coverage"]["observed_hours"] == pytest.approx(1.0)
    assert row["events_per_camera_hour"] == pytest.approx(3.0)


def test_rejections_summarise_the_gate_features(client):
    body = client.get("/api/rejections").json()
    assert body["total"] == 1
    reason = body["reasons"][0]
    assert reason["reason"] == "confined"
    assert reason["features"]["extent_frac"]["p50"] == pytest.approx(0.02)


def test_analytics_are_labelled_utc(client):
    body = client.get("/api/analytics").json()
    assert body["hour_of_day"]["timezone"] == "UTC"
    assert sum(body["hour_of_day"]["bins"]) == 3
    # All three fixture events are at 22:xx UTC.
    assert body["hour_of_day"]["bins"][22] == 3
    assert len(body["origins"]) == 3


def test_analytics_headings_bin_to_eight_sectors(client):
    body = client.get("/api/analytics").json()
    assert len(body["headings"]) == 8
    assert sum(body["headings"]) == 3


# --------------------------------------------------------------------------
# demo store
# --------------------------------------------------------------------------


def test_seed_demo_marks_the_store_as_synthetic(tmp_path):
    """Nothing generated may be mistakable for collected data."""
    from expressvision.demo import seed_demo

    db = tmp_path / "demo.db"
    written = seed_demo(db_path=db, out_dir=tmp_path / "demo", nights=2, write_keyframes=False)

    assert written["events"] > 0
    store = Store(db)
    try:
        assert store.is_demo is True
        assert store.get_meta("demo") == "1"
        versions = {
            row["model_version"] for row in store.conn.execute("SELECT model_version FROM event")
        }
    finally:
        store.close()

    # The model version travels with every event and says so on its face.
    assert all("SYNTHETIC" in v for v in versions)
    assert TestClient(create_app(db_path=db)).get("/api/health").json()["demo"] is True


def test_seed_demo_records_uptime_for_every_run(tmp_path):
    """A demo that showed complete coverage by accident would teach nothing."""
    from expressvision.demo import seed_demo

    db = tmp_path / "demo.db"
    seed_demo(db_path=db, out_dir=tmp_path / "demo", nights=12, write_keyframes=False)

    body = TestClient(create_app(db_path=db)).get("/api/overview").json()
    assert body["coverage"]["complete"] is True
    assert body["events_per_camera_hour"] is not None


def test_seed_demo_leaves_a_coverage_gap_and_a_fouled_lens(tmp_path):
    """Both seeded defects must survive into the API, since they are the point.

    A camera offline for three nights must show fewer observed hours rather than
    simply fewer events, and a fouled camera must show tamper frames — otherwise
    the demo teaches that quiet means clean.
    """
    from expressvision.demo import seed_demo

    db = tmp_path / "demo.db"
    seed_demo(db_path=db, out_dir=tmp_path / "demo", nights=12, write_keyframes=False)

    rows = {r["camera_id"]: r for r in TestClient(create_app(db_path=db)).get("/api/cameras").json()}

    offline = rows["cam-loft-05"]
    healthy = rows["cam-dock-01"]
    assert offline["coverage"]["runs"] < healthy["coverage"]["runs"]
    assert offline["coverage"]["observed_hours"] < healthy["coverage"]["observed_hours"]

    assert rows["cam-store-04"]["tamper_frames"] > 1000


def test_seed_demo_is_deterministic(tmp_path):
    """The same seed rebuilds the same store, so a demo can be talked about."""
    from expressvision.demo import seed_demo

    first = seed_demo(tmp_path / "a.db", tmp_path / "a", nights=3, write_keyframes=False, seed=5)
    second = seed_demo(tmp_path / "b.db", tmp_path / "b", nights=3, write_keyframes=False, seed=5)
    assert first == second


def test_demo_keyframes_carry_a_visible_watermark(tmp_path):
    """The sidecar records the frame as synthetic, alongside the burned-in text."""
    from expressvision.demo import seed_demo

    seed_demo(tmp_path / "d.db", tmp_path / "d", nights=1, write_keyframes=True, seed=3)

    sidecars = list((tmp_path / "d" / "keyframes").rglob("*.json"))
    assert sidecars, "expected keyframe sidecars"
    assert json.loads(sidecars[0].read_text(encoding="utf-8"))["synthetic"] is True
