"""HTTP API for the operator console.

This is the seam the architecture draws between L3 and L4, brought forward to
what exists today: the console talks HTTP to a documented API, and the API reads
the collector's store. When real edge nodes arrive and the store behind this
becomes Postgres rather than SQLite, the console does not change.

Deliberately small. It serves the surfaces the current schema can answer
honestly and refuses to fake the rest — there is no incidents endpoint because
stage G has not been built, and no zone endpoint because zones do not exist.

Run it with ``exv serve``.
"""

# No `from __future__ import annotations` here, unlike the rest of the package.
# FastAPI resolves annotations against module globals, so a stringified
# `Annotated[Store, Depends(...)]` naming a closure-local dependency cannot be
# looked up — it silently degrades into a required query parameter and every
# endpoint 422s. Python 3.12 evaluates these signatures natively anyway.

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..store import Store
from . import queries

# Where the Vite dev server runs. Only these origins are allowed, and only when
# the API is started with --dev; a console served from the API's own origin
# needs no CORS at all.
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

MEDIA_KINDS = {"keyframe": "image/jpeg", "clip": "video/mp4"}


class VerdictRequest(BaseModel):
    """An operator's judgement — the return path in Figure 5."""

    verdict: Literal["confirmed", "rejected", "reclassified"]
    corrected_label: str | None = None
    user: str = Field(default="console", max_length=64)


class Settings:
    """Resolved paths for one server process."""

    def __init__(self, db_path: Path, media_root: Path, static_dir: Path | None) -> None:
        self.db_path = db_path.resolve()
        # Every media file served must resolve inside this directory. Paths come
        # out of the database as free text, so without this an event row holding
        # "../../../etc/passwd" would be served as a keyframe. Confining at the
        # HTTP boundary keeps the check in one auditable place.
        self.media_root = media_root.resolve()
        self.static_dir = static_dir


def create_app(
    db_path: Path,
    media_root: Path | None = None,
    static_dir: Path | None = None,
    dev: bool = False,
) -> FastAPI:
    settings = Settings(
        db_path=Path(db_path),
        media_root=Path(media_root) if media_root else Path(db_path).parent,
        static_dir=Path(static_dir) if static_dir else None,
    )

    app = FastAPI(
        title="ExpressVision console API",
        version="0.1.0",
        summary="Read the collector store; record operator verdicts.",
    )

    if dev:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(DEV_ORIGINS),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    def get_store() -> Iterator[Store]:
        """One connection per request.

        SQLite connections are not safe to share across threads and FastAPI runs
        sync endpoints in a threadpool, so a shared handle would be a latent
        corruption bug rather than an optimisation. WAL mode makes concurrent
        readers cheap, which is the whole access pattern here.
        """
        if not settings.db_path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No store at {settings.db_path}. Run `exv run` to collect "
                    f"events, or `exv seed-demo` for a labelled demo store."
                ),
            )
        store = Store(settings.db_path, check_same_thread=False)
        try:
            yield store
        finally:
            store.close()

    Dep = Annotated[Store, Depends(get_store)]

    # ---------------------------------------------------------------- health

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Reachable without a store, so the console can explain an empty state."""
        exists = settings.db_path.exists()
        demo = False
        if exists:
            store = Store(settings.db_path, check_same_thread=False)
            try:
                demo = store.is_demo
            finally:
                store.close()
        return {
            "ok": True,
            "store": str(settings.db_path),
            "store_exists": exists,
            "demo": demo,
        }

    @app.get("/api/overview")
    def overview(store: Dep) -> dict[str, Any]:
        return queries.overview(store)

    # ---------------------------------------------------------------- events

    @app.get("/api/events")
    def list_events(
        store: Dep,
        camera_id: str | None = None,
        verdict: str | None = Query(
            default=None,
            description="pending | confirmed | rejected | reclassified",
        ),
        label: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        order: Literal["asc", "desc"] = "asc",
    ) -> dict[str, Any]:
        return queries.list_events(
            store,
            camera_id=camera_id,
            verdict=verdict,
            label=label,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
            order=order,
        )

    @app.get("/api/events/{event_id}")
    def get_event(store: Dep, event_id: str) -> dict[str, Any]:
        event = queries.get_event(store, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail=f"No event {event_id}")
        return event

    @app.post("/api/events/{event_id}/verdict")
    def post_verdict(
        store: Dep,
        event_id: str,
        body: Annotated[VerdictRequest, Body()],
    ) -> dict[str, Any]:
        """Record Confirm / Reject / Reclassify against one event."""
        if body.verdict == "reclassified" and not body.corrected_label:
            raise HTTPException(
                status_code=422,
                detail="reclassified needs a corrected_label — that label is the training signal",
            )
        if body.corrected_label and body.corrected_label not in queries.LABELS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown label. Use one of: {', '.join(queries.LABELS)}",
            )

        ok = queries.set_verdict(
            store, event_id, body.verdict, body.corrected_label, body.user
        )
        if not ok:
            raise HTTPException(status_code=404, detail=f"No event {event_id}")
        return {"ok": True, "event_id": event_id, "verdict": body.verdict}

    @app.get("/api/events/{event_id}/{kind}")
    def get_media(store: Dep, event_id: str, kind: str) -> FileResponse:
        """Serve an event's keyframe or clip, confined to the media root."""
        if kind not in MEDIA_KINDS:
            raise HTTPException(status_code=404, detail="Unknown media kind")

        stored = queries.media_path(store, event_id, kind)
        if not stored:
            raise HTTPException(status_code=404, detail=f"Event has no {kind}")

        try:
            resolved = Path(stored).resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=410, detail=f"{kind} is recorded but missing from disk"
            ) from exc

        # Confinement check. `is_relative_to` compares resolved paths, so a
        # symlink or a "../" in the stored value cannot escape the root.
        if not resolved.is_relative_to(settings.media_root):
            raise HTTPException(
                status_code=403,
                detail=f"{kind} lies outside the media root and will not be served",
            )

        return FileResponse(resolved, media_type=MEDIA_KINDS[kind])

    # ------------------------------------------------------- fleet and health

    @app.get("/api/cameras")
    def cameras(store: Dep) -> list[dict[str, Any]]:
        return queries.cameras(store)

    @app.get("/api/runs")
    def runs(store: Dep, limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, Any]]:
        return queries.runs(store, limit=limit)

    @app.get("/api/funnel")
    def funnel(store: Dep, run_id: str | None = None) -> dict[str, Any]:
        return queries.funnel(store, run_id=run_id)

    @app.get("/api/rejections")
    def rejections(store: Dep) -> dict[str, Any]:
        return queries.rejections(store)

    @app.get("/api/analytics")
    def analytics(store: Dep, camera_id: str | None = None) -> dict[str, Any]:
        return queries.analytics(store, camera_id=camera_id) | {
            "extent": queries.frame_extent(store)
        }

    @app.exception_handler(sqlite3.DatabaseError)
    def _sqlite_error(_request: Any, exc: sqlite3.DatabaseError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": f"Store error: {exc}"})

    _mount_console(app, settings.static_dir)
    return app


def _mount_console(app: FastAPI, static_dir: Path | None) -> None:
    """Serve the built console, if it has been built.

    Absent a build the API still runs; `exv serve` says how to start the dev
    server instead. That keeps the backend usable on a machine with no Node.
    """
    if static_dir is None or not (static_dir / "index.html").exists():
        return

    index = static_dir / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def console(path: str) -> FileResponse:
        candidate = (static_dir / path).resolve() if path else index
        if (
            path
            and candidate.is_file()
            and candidate.is_relative_to(static_dir.resolve())
        ):
            return FileResponse(candidate)
        # Client-side routing: unknown paths are console routes, not 404s.
        return FileResponse(index)
