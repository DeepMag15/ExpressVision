"""Command line for the collector.

    exv make-fixture              synthesise test footage
    exv run <source>              run the cascade on a file or RTSP URL
    exv run-config <config.yaml>  run every camera in a config
    exv stats                     what is in the store
    exv review                    list events awaiting an operator verdict
    exv verdict <id> <verdict>    record a verdict
    exv serve                     run the operator console
    exv seed-demo                 build a labelled synthetic store to demo with
    exv init-config               write a starter cameras.yaml
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .config import CameraConfig, PipelineConfig
from .detect import (
    HARVEST_ONLY_NOTICE,
    MEGADETECTOR_DEFAULT,
    MEGADETECTOR_VARIANTS,
    Detector,
    build_detector,
)
from .pipeline import CameraPipeline
from .store import Store
from .synth import make_test_video, plan_fixture
from .types import FunnelStats

app = typer.Typer(add_completion=False, help="ExpressVision collector")
console = Console()


def _build(
    kind: str,
    variant: str,
    conf: float,
    tile_size: int,
    image_size: int | None = None,
    device: str = "auto",
    half: bool = False,
) -> Detector:
    """Construct the detector, failing early and legibly on a bad choice."""
    if kind == "motion":
        return build_detector("motion")

    if kind != "megadetector":
        console.print(f"[red]Unknown detector {kind!r} — use motion or megadetector.[/]")
        raise typer.Exit(1)

    if variant not in MEGADETECTOR_VARIANTS:
        console.print(f"[red]Unknown variant {variant!r}.[/] Available:")
        for name, (arch, note) in MEGADETECTOR_VARIANTS.items():
            console.print(f"  {name:16} {arch:22} {note}")
        raise typer.Exit(1)

    try:
        model = build_detector(
            "megadetector",
            version=variant,
            conf_threshold=conf,
            tile_size=tile_size,
            image_size=image_size,
            device=device,
            half=half,
        )
    except (ImportError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    arch, note = MEGADETECTOR_VARIANTS[variant]
    console.print(f"[bold]{variant}[/] · {arch} · {note}")
    console.print(f"[yellow]{HARVEST_ONLY_NOTICE}[/]\n")
    return model


def _runtime_table(info: dict[str, str]) -> Table:
    """Report where the model actually ended up, not where we asked it to go."""
    table = Table(title="Detector runtime", title_justify="left", header_style="bold")
    table.add_column("Property")
    table.add_column("Value")
    for key, value in info.items():
        table.add_row(key.replace("_", " "), str(value))
    return table


def _warn_if_device_mismatch(info: dict[str, str]) -> None:
    resolved = info.get("resolved_device", "")
    actual = info.get("actual_device", "")
    if actual in {"not loaded", "unknown", ""}:
        return
    # "cuda" vs "cuda:0" is the same device; compare the type only.
    if actual.split(":")[0] != resolved.split(":")[0]:
        console.print(
            f"[red]Device mismatch: asked for {resolved}, model is on {actual}.[/]\n"
            f"[red]Any throughput measured here describes {actual}, not {resolved}.[/]"
        )


def _funnel_table(camera_id: str, stats: FunnelStats, elapsed: float) -> Table:
    table = Table(title=f"Cascade — {camera_id}", title_justify="left", header_style="bold")
    table.add_column("Stage")
    table.add_column("Count", justify="right")
    table.add_column("Of previous", justify="right")

    rows = [
        ("Frames read", stats.frames_read),
        ("Frames processed", stats.frames_processed),
        ("Passed motion gate", stats.frames_gated),
        ("ROIs", stats.rois),
        ("Tiles inferred", stats.tiles),
        ("Detections", stats.detections),
        ("Tracks created", stats.tracks_created),
        ("Tracks confirmed", stats.tracks_confirmed),
        ("Events", stats.events),
    ]

    previous: int | None = None
    for label, value in rows:
        share = ""
        if previous is not None and previous > 0:
            share = f"{100.0 * value / previous:.1f}%"
        table.add_row(label, f"{value:,}", share)
        previous = value

    if stats.frames_processed:
        table.add_section()
        gate_ratio = 100.0 * stats.frames_gated / stats.frames_processed
        tiles_per_gated = stats.tiles / stats.frames_gated if stats.frames_gated else 0.0
        saving = stats.frames_processed / stats.tiles if stats.tiles else float("inf")
        table.add_row("Gate pass rate", f"{gate_ratio:.1f}%", "")
        table.add_row("Tiles per gated frame", f"{tiles_per_gated:.2f}", "")
        table.add_row("Inference saved vs every-frame", f"{saving:.1f}x", "")
        table.add_row("Throughput", f"{stats.frames_processed / max(elapsed, 1e-6):.1f} fps", "")

    if stats.global_change_frames or stats.tamper_frames:
        table.add_section()
        table.add_row("Global-change frames", f"{stats.global_change_frames:,}", "")
        table.add_row("Tamper frames", f"{stats.tamper_frames:,}", "")

    if stats.rejected:
        table.add_section()
        for reason, count in sorted(stats.rejected.items(), key=lambda kv: -kv[1]):
            table.add_row(f"  rejected: {reason}", f"{count:,}", "")

    return table


def _run_camera(cfg: PipelineConfig, camera: CameraConfig, store: Store | None,
                max_frames: int | None, detector: Detector | None = None) -> FunnelStats:
    pipeline = CameraPipeline(cfg, camera, store=store, detector=detector)

    start = time.monotonic()
    with console.status(f"[bold]{camera.id}[/] — processing…") as status:
        def progress(frame_idx: int, stats: FunnelStats) -> None:
            status.update(
                f"[bold]{camera.id}[/] — frame {frame_idx:,} · "
                f"gated {stats.frames_gated:,} · events {stats.events:,}"
            )

        stats = pipeline.run(max_frames=max_frames, on_progress=progress)
    elapsed = time.monotonic() - start

    console.print(_funnel_table(camera.id, stats, elapsed))
    return stats


@app.command("make-fixture")
def make_fixture(
    out: Annotated[Path, typer.Option(help="Output video path")] = Path("data/fixture.mp4"),
    seconds: Annotated[int, typer.Option(help="Clip length")] = 60,
    width: Annotated[int, typer.Option(help="Frame width")] = 1280,
    height: Annotated[int, typer.Option(help="Frame height")] = 720,
) -> None:
    """Synthesise test footage with a known-good answer."""
    plan = plan_fixture(seconds=seconds, width=width, height=height)
    make_test_video(out, plan)
    size_mb = out.stat().st_size / 1e6
    console.print(
        f"[green]Wrote[/] {out}  ({size_mb:.1f} MB, {seconds}s, {width}x{height})"
    )
    console.print(plan.describe())


@app.command()
def run(
    source: Annotated[str, typer.Argument(help="Video file path or RTSP URL")],
    camera_id: Annotated[str, typer.Option("--camera", help="Camera id")] = "cam-001",
    out_dir: Annotated[Path, typer.Option(help="Where clips and keyframes go")] = Path("out"),
    db: Annotated[Path, typer.Option(help="SQLite store path")] = Path("out/expressvision.db"),
    max_frames: Annotated[int, typer.Option(help="Stop after N frames (0 = all)")] = 0,
    detector: Annotated[
        str, typer.Option(help="motion (harvest everything) | megadetector")
    ] = "motion",
    variant: Annotated[
        str, typer.Option(help="MegaDetector variant")
    ] = MEGADETECTOR_DEFAULT,
    conf: Annotated[float, typer.Option(help="Detector confidence threshold")] = 0.20,
    image_size: Annotated[
        int, typer.Option(help="Detector input size; 0 = model default (1280). 640 is ~4x faster")
    ] = 0,
    device: Annotated[
        str, typer.Option(help="auto | cpu | cuda | cuda:0")
    ] = "auto",
    half: Annotated[
        bool, typer.Option("--half", help="fp16 inference (CUDA only)")
    ] = False,
    no_clips: Annotated[bool, typer.Option("--no-clips", help="Skip clip writing")] = False,
    no_store: Annotated[bool, typer.Option("--no-store", help="Skip the database")] = False,
) -> None:
    """Run the cascade on one source."""
    cfg = PipelineConfig.for_source(source, camera_id)
    cfg.out_dir = out_dir
    cfg.db_path = db
    if no_clips:
        cfg.event.write_clips = False

    model = _build(
        detector, variant, conf, cfg.tile.tile_size, image_size or None, device, half
    )
    if hasattr(model, "runtime_info"):
        model.warmup(1)                       # forces the load so the report is real
        info = model.runtime_info()
        console.print(_runtime_table(info))
        _warn_if_device_mismatch(info)

    store = None if no_store else Store(db)
    try:
        stats = _run_camera(cfg, cfg.cameras[0], store, max_frames or None, model)
    finally:
        if store:
            store.close()

    if stats.events and not no_clips:
        console.print(f"\nEvidence written to [bold]{out_dir}[/]")


@app.command("run-config")
def run_config(
    config: Annotated[Path, typer.Argument(help="Path to cameras.yaml")],
    max_frames: Annotated[int, typer.Option(help="Stop after N frames per camera")] = 0,
) -> None:
    """Run every camera defined in a config file."""
    cfg = PipelineConfig.load(config)
    if not cfg.cameras:
        console.print("[red]No cameras defined in config.[/]")
        raise typer.Exit(1)

    with Store(cfg.db_path) as store:
        totals = FunnelStats()
        for camera in cfg.cameras:
            stats = _run_camera(cfg, camera, store, max_frames or None)
            totals.frames_processed += stats.frames_processed
            totals.frames_gated += stats.frames_gated
            totals.tiles += stats.tiles
            totals.events += stats.events

    console.print(
        f"\n[bold]{len(cfg.cameras)} cameras[/] · "
        f"{totals.frames_processed:,} frames · "
        f"{totals.tiles:,} tiles · "
        f"{totals.events:,} events"
    )


@app.command()
def stats(
    db: Annotated[Path, typer.Option(help="SQLite store path")] = Path("out/expressvision.db"),
) -> None:
    """Summarise what is in the store."""
    if not db.exists():
        console.print(f"[yellow]No store at {db}. Run `exv run` first.[/]")
        raise typer.Exit(1)

    with Store(db) as store:
        counts = store.counts()
        table = Table(header_style="bold")
        table.add_column("Metric")
        table.add_column("Count", justify="right")
        for key, value in counts.items():
            table.add_row(key.replace("_", " "), f"{value:,}")
        console.print(table)

        rows = list(
            store.conn.execute(
                "SELECT reason, COUNT(*) n FROM rejection GROUP BY reason ORDER BY n DESC"
            )
        )
        if rows:
            rejected = Table(title="Discarded tracks by reason", title_justify="left",
                             header_style="bold")
            rejected.add_column("Reason")
            rejected.add_column("Count", justify="right")
            for row in rows:
                rejected.add_row(row["reason"], f"{row['n']:,}")
            console.print(rejected)


@app.command()
def review(
    db: Annotated[Path, typer.Option(help="SQLite store path")] = Path("out/expressvision.db"),
    limit: Annotated[int, typer.Option(help="How many to include")] = 20,
    html: Annotated[
        Path | None,
        typer.Option(help="Write a keyboard-driven HTML review sheet here instead"),
    ] = None,
    thumb_width: Annotated[int, typer.Option(help="Thumbnail width in the sheet")] = 480,
    include_verified: Annotated[
        bool, typer.Option("--include-verified", help="Also include already-judged events")
    ] = False,
) -> None:
    """List events awaiting a verdict, or build an HTML sheet for fast review."""
    if html is not None:
        from .review import build_review_sheet

        with Store(db) as store:
            stats = build_review_sheet(
                store, html, limit=limit, thumb_width=thumb_width,
                include_verified=include_verified,
            )
        if stats.total == 0:
            console.print("[green]Nothing to review.[/]")
            return
        size_mb = html.stat().st_size / 1e6
        console.print(
            f"[green]Wrote[/] {html}  ({stats.total} events, {size_mb:.1f} MB)"
        )
        if stats.missing_keyframes:
            console.print(
                f"[yellow]{stats.missing_keyframes} events have no keyframe[/] — "
                f"was the run made with --no-clips?"
            )
        console.print(
            "\nOpen it in a browser. J/K to move, C confirm, R reject, 1-7 reclassify.\n"
            "Then: [bold]exv import-verdicts verdicts.json[/]"
        )
        return

    with Store(db) as store:
        rows = store.pending_events(limit)
        if not rows:
            console.print("[green]Nothing pending.[/]")
            return

        table = Table(title="Awaiting verdict", title_justify="left", header_style="bold")
        for column in ("id", "camera", "when", "dur", "dets", "straight", "keyframe"):
            table.add_column(column)
        for row in rows:
            table.add_row(
                row["id"][:12],
                row["camera_id"],
                (row["started_at"] or "")[11:19],
                f"{row['duration_s']:.1f}s",
                str(row["n_detections"]),
                f"{row['straightness']:.2f}",
                Path(row["keyframe_path"]).name if row["keyframe_path"] else "—",
            )
        console.print(table)
        console.print("\nRecord one with: [bold]exv verdict <id> confirmed|rejected[/]")


@app.command()
def verdict(
    event_id: Annotated[str, typer.Argument(help="Event id (full or 12-char prefix)")],
    call: Annotated[str, typer.Argument(help="confirmed | rejected | reclassified")],
    label: Annotated[str, typer.Option(help="Corrected label if reclassified")] = "",
    db: Annotated[Path, typer.Option(help="SQLite store path")] = Path("out/expressvision.db"),
) -> None:
    """Record an operator verdict — the training-data return path."""
    if call not in {"confirmed", "rejected", "reclassified"}:
        console.print("[red]Verdict must be confirmed, rejected or reclassified.[/]")
        raise typer.Exit(1)

    with Store(db) as store:
        row = store.conn.execute(
            "SELECT id FROM event WHERE id = ? OR id LIKE ?", (event_id, f"{event_id}%")
        ).fetchone()
        if not row:
            console.print(f"[red]No event matching {event_id}.[/]")
            raise typer.Exit(1)
        store.set_verdict(row["id"], call, label or None)

    console.print(f"[green]Recorded[/] {row['id'][:12]} → {call}")


@app.command("survey")
def survey_cmd(
    sources: Annotated[list[Path], typer.Argument(help="Clips or stills, one per camera")],
    hfov: Annotated[
        float, typer.Option(help="Horizontal field of view in degrees (from the datasheet)")
    ] = 0.0,
    lens_mm: Annotated[
        float, typer.Option(help="Lens focal length; used only if --hfov is not given")
    ] = 0.0,
    sensor: Annotated[str, typer.Option(help='Sensor size, e.g. 1/2.8"')] = '1/2.8"',
    nearest_m: Annotated[
        float, typer.Option(help="Distance to the nearest floor point in view")
    ] = 0.0,
    furthest_m: Annotated[
        float, typer.Option(help="Distance to the furthest floor point in view")
    ] = 0.0,
    frames: Annotated[int, typer.Option(help="Frames to sample per clip")] = 120,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the full report here")
    ] = None,
) -> None:
    """Assess whether a camera can physically see a rodent.

    The Phase 0 question, answered per camera. Run it on the first footage that
    arrives from a site, before anything else.
    """
    import json as _json

    from .survey import Optics, survey

    optics = Optics(
        hfov_deg=hfov or None,
        lens_mm=lens_mm or None,
        sensor=sensor,
        nearest_floor_m=nearest_m or None,
        furthest_floor_m=furthest_m or None,
    )

    results = []
    for source in sources:
        with console.status(f"assessing {source.name}…"):
            result = survey(source, optics, sample_frames=frames)
        results.append(result)
        _print_survey(result)

    if len(results) > 1:
        table = Table(title="Summary", title_justify="left", header_style="bold")
        table.add_column("Camera")
        table.add_column("Verdict")
        for r in results:
            table.add_row(Path(r.source).name, _verdict_markup(r.verdict))
        console.print(table)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            _json.dumps(
                [
                    {
                        "source": r.source,
                        "verdict": r.verdict,
                        "width": r.width, "height": r.height, "fps": r.fps,
                        "duration_s": round(r.duration_s, 1),
                        "night_fraction": round(r.night_fraction, 3),
                        "sharpness": round(r.sharpness, 1),
                        "temporal_noise": round(r.temporal_noise, 2),
                        "blur_ratio": round(r.blur_ratio, 3),
                        "corner_falloff": round(r.corner_falloff, 3),
                        "blockiness": round(r.blockiness, 3),
                        "exposure_drift": round(r.exposure_drift, 2),
                        "motion_fraction": round(r.motion_fraction, 3),
                        "max_detect_range_m": r.max_range_m(),
                        "findings": [
                            {"name": f.name, "value": f.value, "status": f.status,
                             "note": f.note, "fix": f.fix}
                            for f in r.findings
                        ],
                        "error": r.error,
                    }
                    for r in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"[green]Wrote[/] {json_out}")


def _verdict_markup(verdict: str) -> str:
    colour = {
        "usable": "green",
        "usable after reconfiguration": "yellow",
        "replace or relocate": "red",
        "unreadable": "red",
    }.get(verdict, "white")
    return f"[{colour}]{verdict}[/]"


_MARKS = {
    "ok": "[green]  ok  [/]",
    "warn": "[yellow] warn [/]",
    "fail": "[red] FAIL [/]",
    "info": "[dim] info [/]",
}


def _print_survey(result) -> None:
    """Print one camera's assessment.

    Measurements go in a compact table; explanations go underneath as prose.
    Cramming long guidance into a table column squeezes it to two characters
    wide and makes the whole report unreadable.
    """
    console.print(
        f"\n[bold]{Path(result.source).name}[/] — {_verdict_markup(result.verdict)}"
    )

    if result.error:
        console.print(f"  [red]{result.error}[/]")
        return

    table = Table(box=None, show_header=True, header_style="dim", pad_edge=False)
    table.add_column(" ", no_wrap=True)
    table.add_column("Check", no_wrap=True)
    table.add_column("Measured", no_wrap=True)
    for f in result.findings:
        table.add_row(_MARKS.get(f.status, ""), f.name, f.value)
    console.print(table)

    problems = [f for f in result.findings if f.status in {"warn", "fail"}]
    if not problems:
        console.print("  [green]No problems found — this camera can see a rodent.[/]")
        return

    # Wrap to the real terminal, not an assumed width — otherwise long guidance
    # double-wraps and the indentation falls apart.
    body = max(40, console.width - 10)

    console.print("\n  [bold]What needs attention[/]")
    for f in problems:
        console.print(f"\n  {_MARKS[f.status]} [bold]{f.name}[/] — {f.value}")
        if f.note:
            for line in _wrap(f.note, body):
                console.print(f"        {line}")
        if f.fix:
            for i, line in enumerate(_wrap(f.fix, body - 5)):
                prefix = "        [bold]Fix:[/] " if i == 0 else "             "
                console.print(f"{prefix}{line}")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


@app.command()
def doctor() -> None:
    """Report the compute environment — run this first on a new machine."""
    from .gpu import probe_environment

    env = probe_environment()

    table = Table(title="Environment", title_justify="left", header_style="bold")
    table.add_column("Property")
    table.add_column("Value")
    rows = [
        ("python", env.python),
        ("platform", env.platform),
        ("torch", env.torch_version or "[red]not installed[/]"),
        ("torch CUDA build", env.torch_cuda_build or "[red]CPU-only[/]"),
        ("cuda available", "[green]yes[/]" if env.cuda_available else "[red]no[/]"),
        ("devices", str(env.device_count)),
        ("device", env.device_name or "—"),
        ("compute capability", env.device_capability or "—"),
        ("total VRAM", f"{env.total_vram_mb / 1000:.1f} GB" if env.total_vram_mb else "—"),
        ("driver", env.driver_version or "—"),
        ("cuDNN", env.cudnn_version or "—"),
        ("NVML (telemetry)", "yes" if env.nvml_available else "no"),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)

    for note in env.notes:
        console.print(f"[yellow]note:[/] {note}")
    for problem in env.problems:
        console.print(f"[red]problem:[/] {problem}")

    if env.gpu_ready:
        console.print("\n[green]GPU ready.[/] Benchmark with: exv benchmark --device cuda")
    else:
        console.print(
            "\n[yellow]No usable GPU — the pipeline still runs on CPU.[/] "
            "See README, 'GPU setup'."
        )


@app.command()
def benchmark(
    source: Annotated[str, typer.Option(help="Clip to benchmark; generated if omitted")] = "",
    variant: Annotated[str, typer.Option(help="MegaDetector variant")] = MEGADETECTOR_DEFAULT,
    device: Annotated[str, typer.Option(help="auto | cpu | cuda | cuda:0")] = "auto",
    image_sizes: Annotated[
        str, typer.Option(help="Comma-separated detector input sizes")
    ] = "640,1280",
    half: Annotated[bool, typer.Option("--half", help="Also measure fp16")] = False,
    max_frames: Annotated[int, typer.Option(help="Frames per configuration")] = 400,
    stream_fps: Annotated[float, typer.Option(help="Frame rate of one live stream")] = 15.0,
    include_motion: Annotated[
        bool, typer.Option("--include-motion/--no-include-motion",
                           help="Also benchmark the motion-only path as a ceiling")
    ] = True,
    out: Annotated[Path, typer.Option(help="Write results JSON here")] = Path(
        "out/benchmark.json"
    ),
) -> None:
    """Measure throughput and estimate how many streams this hardware carries."""
    import json

    from .benchmark import (
        bottleneck,
        default_bench_clip,
        results_to_dict,
        run_benchmark,
    )
    from .gpu import probe_environment

    env = probe_environment()
    console.print(
        f"[bold]{env.device_name or 'CPU'}[/] · torch {env.torch_version or '—'} · "
        f"CUDA {env.torch_cuda_build or 'n/a'} · "
        f"{'[green]GPU[/]' if env.gpu_ready else '[yellow]CPU only[/]'}\n"
    )
    for problem in env.problems:
        console.print(f"[red]problem:[/] {problem}")

    clip = Path(source) if source else default_bench_clip(Path("data/bench.mp4"))
    if not clip.exists():
        console.print(f"[red]No such clip: {clip}[/]")
        raise typer.Exit(1)
    console.print(f"clip: {clip}\n")

    cfg = PipelineConfig()
    results = []

    if include_motion:
        console.print("[dim]motion-only (no detector) — the pipeline's ceiling[/]")
        results.append(
            run_benchmark(
                str(clip), build_detector("motion"), cfg,
                max_frames=max_frames, label="motion-only",
            )
        )

    sizes = [int(s) for s in image_sizes.split(",") if s.strip()]
    precisions = [False, True] if half else [False]

    for size in sizes:
        for use_half in precisions:
            if use_half and not env.gpu_ready:
                console.print("[yellow]skipping fp16 — no GPU[/]")
                continue
            tag = f"{variant} @{size}{' fp16' if use_half else ''}"
            console.print(f"[dim]{tag}[/]")
            try:
                model = build_detector(
                    "megadetector", version=variant, tile_size=cfg.tile.tile_size,
                    image_size=size, device=device, half=use_half,
                )
                model.warmup(1)
                _warn_if_device_mismatch(model.runtime_info())
                results.append(
                    run_benchmark(
                        str(clip), model, cfg, max_frames=max_frames, label=tag
                    )
                )
            except (ImportError, ValueError) as exc:
                console.print(f"[red]{exc}[/]")
                raise typer.Exit(1) from exc

    console.print()
    console.print(_benchmark_table(results, stream_fps))

    console.print("\n[bold]Where the time goes[/]")
    for r in results:
        stage, share = bottleneck(r)
        console.print(
            f"  {r.label:34} {stage} ({share * 100:.0f}% of pipeline time)"
            + (
                f" · GPU {r.gpu_util_mean:.0f}% mean"
                if r.gpu_util_mean is not None else ""
            )
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = results_to_dict(results, env.__dict__ | {"problems": env.problems})
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(f"\n[green]Wrote[/] {out}")

    console.print(
        "\n[dim]Stream estimates assume the benchmark clip's motion load and keep "
        "30% headroom for bursts.\nReal sites are quieter overnight (single-digit "
        "gate rates) but spike at shift changes.\nTreat these as a floor, and "
        "re-measure on real footage when it arrives.[/]"
    )


def _benchmark_table(results: list, stream_fps: float) -> Table:
    """Only show GPU columns when something actually ran on a GPU — empty
    telemetry columns on a CPU run read as a broken GPU rather than no GPU."""
    any_gpu = any(r.gpu_util_mean is not None for r in results)

    table = Table(title="Throughput", title_justify="left", header_style="bold",
                  pad_edge=False)
    table.add_column("Configuration", no_wrap=True)
    table.add_column("Device", no_wrap=True)
    table.add_column("FPS", justify="right")
    table.add_column("ms/tile", justify="right")
    if any_gpu:
        table.add_column("GPU%", justify="right")
        table.add_column("VRAM", justify="right")
    table.add_column(f"Streams\n@{stream_fps:.0f}fps", justify="right")
    table.add_column("h per\ncam-night", justify="right")

    for r in results:
        row = [
            r.label,
            f"{r.device} {r.dtype}".replace(" n/a", "").strip(),
            f"{r.fps:.1f}",
            f"{r.detect_ms_per_tile:.1f}" if r.detect_ms_per_tile else "—",
        ]
        if any_gpu:
            row += [
                f"{r.gpu_util_mean:.0f}" if r.gpu_util_mean is not None else "—",
                f"{r.vram_peak_mb / 1000:.1f}G" if r.vram_peak_mb is not None else "—",
            ]
        row += [
            f"{r.streams_supported(stream_fps):.1f}",
            f"{r.camera_night_hours(stream_fps):.1f}",
        ]
        table.add_row(*row)
    return table


@app.command("import-verdicts")
def import_verdicts_cmd(
    path: Annotated[Path, typer.Argument(help="verdicts.json exported from the review sheet")],
    db: Annotated[Path, typer.Option(help="SQLite store path")] = Path("out/expressvision.db"),
    user: Annotated[str, typer.Option(help="Who reviewed these")] = "review-sheet",
) -> None:
    """Apply verdicts exported from the HTML review sheet."""
    from .review import import_verdicts

    if not path.exists():
        console.print(f"[red]No such file: {path}[/]")
        raise typer.Exit(1)

    with Store(db) as store:
        applied, skipped = import_verdicts(store, path, user)
        counts = store.counts()

    console.print(f"[green]Applied {applied}[/] verdicts" + (
        f", [yellow]skipped {skipped}[/]" if skipped else ""
    ))
    console.print(
        f"store now: {counts['confirmed']} confirmed, {counts['rejected']} rejected, "
        f"{counts['unverified']} unverified"
    )


@app.command("export-coco")
def export_coco_cmd(
    out: Annotated[Path, typer.Argument(help="Directory to write the dataset into")],
    db: Annotated[Path, typer.Option(help="SQLite store path")] = Path("out/expressvision.db"),
    all_events: Annotated[
        bool, typer.Option("--all", help="Include unverified events, not just confirmed")
    ] = False,
) -> None:
    """Export keyframes and boxes as COCO pre-annotations for CVAT / Label Studio."""
    from .review import export_coco

    with Store(db) as store:
        result = export_coco(store, out, only_confirmed=not all_events)

    if result["images"] == 0:
        console.print(
            "[yellow]Nothing exported.[/] Confirm some events first "
            "(exv review --html sheet.html), or pass --all."
        )
        return

    console.print(
        f"[green]Wrote[/] {out}  ({result['images']} images, "
        f"{result['annotations']} pre-annotations"
        + (f", {result['skipped']} skipped" if result["skipped"] else "") + ")"
    )
    console.print(
        "\n[yellow]These are pre-annotations, not ground truth.[/] The boxes come from "
        "motion ROIs\nor MegaDetector proposals — a human corrects them in CVAT or "
        "Label Studio."
    )


@app.command()
def serve(
    db: Annotated[Path, typer.Option(help="SQLite store path")] = Path("out/expressvision.db"),
    media_root: Annotated[
        Path | None,
        typer.Option(help="Confine clip/keyframe serving to this directory (default: db's dir)"),
    ] = None,
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
    dev: Annotated[
        bool, typer.Option("--dev", help="Allow the Vite dev server origin (CORS)")
    ] = False,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes")] = False,
) -> None:
    """Run the operator console — event feed, verdicts, pipeline health.

    Binds to localhost by default. There is no authentication yet, so exposing
    this on 0.0.0.0 would publish evidence clips to the whole network; that has
    to wait for the RBAC and view-auditing the architecture specifies in §10.
    """
    try:
        import uvicorn

        from .api import create_app
    except ImportError as exc:
        console.print(
            "[red]The console needs FastAPI and uvicorn.[/]\n"
            "Install them with: [bold]uv pip install -e \".[web]\"[/]"
        )
        raise typer.Exit(1) from exc

    static_dir = Path(__file__).resolve().parents[2] / "web" / "dist"
    built = (static_dir / "index.html").exists()

    if not db.exists():
        console.print(
            f"[yellow]No store at {db}.[/] The API will run and say so.\n"
            f"Collect some events with [bold]exv run data/fixture.mp4[/], or build a "
            f"demo store with [bold]exv seed-demo[/].\n"
        )
    else:
        with Store(db) as store:
            counts = store.counts()
            if store.is_demo:
                console.print(
                    "[bold yellow]This store is synthetic demo data.[/] "
                    "Every view will be bannered as such.\n"
                )
        console.print(
            f"store: [bold]{db}[/] · {counts['events']:,} events, "
            f"{counts['unverified']:,} awaiting a verdict\n"
        )

    url = f"http://{host}:{port}"
    if built:
        console.print(f"[green]Console[/]  {url}")
    else:
        console.print(
            f"[green]API[/]      {url}/api/overview   ·   docs at {url}/docs\n"
            f"[yellow]The console front end is not built.[/] Either:\n"
            f"  build it once  — [bold]cd web && npm install && npm run build[/]\n"
            f"  or develop it  — [bold]cd web && npm run dev[/] with "
            f"[bold]exv serve --dev[/] running here"
        )

    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            f"\n[red]Binding to {host} exposes evidence clips with no authentication.[/]"
        )

    if reload:
        # Reload re-imports the app in a fresh process, so it needs a module
        # path and the settings have to travel through the environment.
        import os

        os.environ["EXV_DB"] = str(db)
        os.environ["EXV_DEV"] = "1" if dev else "0"
        if media_root:
            os.environ["EXV_MEDIA_ROOT"] = str(media_root)
        if built:
            os.environ["EXV_STATIC_DIR"] = str(static_dir)
        uvicorn.run(
            "expressvision.api.devserver:app",
            host=host, port=port, reload=True, log_level="info",
        )
        return

    uvicorn.run(
        create_app(
            db_path=db,
            media_root=media_root,
            static_dir=static_dir if built else None,
            dev=dev,
        ),
        host=host,
        port=port,
        log_level="info",
    )


@app.command("seed-demo")
def seed_demo_cmd(
    db: Annotated[Path, typer.Option(help="Where to write the demo store")] = Path(
        "out/demo.db"
    ),
    out_dir: Annotated[Path, typer.Option(help="Where demo keyframes go")] = Path("out/demo"),
    nights: Annotated[int, typer.Option(help="Nights of history to generate")] = 14,
    no_media: Annotated[
        bool, typer.Option("--no-media", help="Skip rendering keyframes (much faster)")
    ] = False,
    seed: Annotated[int, typer.Option(help="RNG seed; the same seed rebuilds the same store")] = 41,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing store at this path")
    ] = False,
) -> None:
    """Build a synthetic store so the console can be demonstrated before footage.

    Everything it writes is invented. The store is stamped as demo data, the API
    reports the flag, the console banners every view, and each keyframe carries
    the words burned into the picture — because a screenshot of pest activity
    that is not visibly synthetic is a claim this system cannot yet support.
    """
    from .demo import CAMERAS, seed_demo

    if db.exists():
        if not force:
            console.print(
                f"[yellow]{db} already exists — not overwriting.[/] Pass --force to replace it."
            )
            raise typer.Exit(1)
        with Store(db) as store:
            if not store.is_demo:
                console.print(
                    f"[red]{db} holds real collected data, not a demo store.[/]\n"
                    f"Refusing to overwrite it even with --force. Choose another path."
                )
                raise typer.Exit(1)
        db.unlink()
        for suffix in ("-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)

    with console.status(f"generating {nights} nights across {len(CAMERAS)} cameras…"):
        written = seed_demo(
            db_path=db, out_dir=out_dir, nights=nights,
            write_keyframes=not no_media, seed=seed,
        )

    table = Table(title="Demo store", title_justify="left", header_style="bold")
    table.add_column("Wrote")
    table.add_column("Count", justify="right")
    for key, value in written.items():
        table.add_row(key, f"{value:,}")
    console.print(table)

    console.print(
        f"\n[bold yellow]This is synthetic data.[/] No model was trained and no real "
        f"footage exists yet.\nIt is stamped demo in the store, and the console says so "
        f"on every screen.\n\nView it: [bold]exv serve --db {db} --media-root {out_dir}[/]"
    )


@app.command("build-dataset")
def build_dataset_cmd(
    out: Annotated[Path, typer.Argument(help="Directory to write the dataset into")],
    work_dir: Annotated[
        Path, typer.Option(help="Where metadata and fetched source images are cached")
    ] = Path("data/lila"),
    frames: Annotated[int, typer.Option(help="How many training frames to generate")] = 400,
    sources: Annotated[
        int, typer.Option(help="How many source images to fetch per class")
    ] = 150,
    classes: Annotated[
        str, typer.Option(help="Comma-separated classes to include")
    ] = "rodent,bird",
    backgrounds_dir: Annotated[
        Path | None,
        typer.Option(help="Use your own overnight stills as backgrounds (strongly preferred)"),
    ] = None,
    night_only: Annotated[
        bool, typer.Option("--night-only/--any-light", help="Keep only infrared night backgrounds")
    ] = True,
    seed: Annotated[int, typer.Option(help="RNG seed; same seed rebuilds the same set")] = 11,
) -> None:
    """Build a scale-corrected training set from public camera-trap imagery.

    Replaces the client-footage dependency. Real rodents from the Channel
    Islands corpus are rescaled from their native 108-506 px down to the 24-140
    px range inherited CCTV actually delivers, then composited into night
    backgrounds with ground-truth boxes that are exact by construction.
    """
    import cv2

    from .dataset import lila
    from .dataset.compose import (
        ComposeConfig,
        Composer,
        is_night_ir,
        size_histogram,
        write_coco,
    )

    work_dir = Path(work_dir)
    wanted = [c.strip() for c in classes.split(",") if c.strip()]

    console.print(f"[dim]source: {lila.CITATION}[/]\n")

    with console.status("resolving metadata (18 MB on first run)…"):
        metadata = lila.download_metadata(work_dir)
    with console.status("building index (slow once, cached after)…"):
        index = lila.LilaIndex.build(metadata, cache=work_dir / "index.json")

    table = Table(title="Available annotations", title_justify="left", header_style="bold")
    table.add_column("Class")
    table.add_column("Boxes", justify="right")
    for name, count in index.summary().items():
        table.add_row(name, f"{count:,}")
    console.print(table)

    cfg = ComposeConfig(seed=seed)

    fetched: list = []
    for name in wanted:
        picks = index.select(
            name, min_px=cfg.source_min_px, limit=sources, seed=seed
        )
        if not picks:
            console.print(f"[yellow]no crops for {name!r} above {cfg.source_min_px}px[/]")
            continue
        with console.status(f"fetching {len(picks)} {name} sources…") as status:
            def progress(i: int, total: int, ok: int, _n=name) -> None:
                status.update(f"fetching {_n}: {i}/{total} ({ok} ok)")

            got = list(lila.fetch_many(picks, work_dir / "images", progress))
        console.print(f"  {name:8} {len(got):>4} source images")
        if name == "human" and not got:
            # Not a bug: LILA withholds frames containing people from the public
            # download, so the annotations exist and the images 404.
            console.print(
                "    [yellow]Human images are withheld from the public download for "
                "privacy.[/]\n"
                "    [dim]Collect this class yourself — walk past your own IR camera. "
                "It is the\n    easiest class to obtain and raises no privacy "
                "question.[/]"
            )
        fetched.extend(got)

    if not fetched:
        console.print("[red]No source imagery could be fetched. Check connectivity.[/]")
        raise typer.Exit(1)

    # Backgrounds are the half of this problem public wildlife data cannot solve.
    # A rodent looks like a rodent anywhere; a warehouse at night does not look
    # like a hillside at noon. Self-collected backgrounds need no animal in them,
    # so they are the cheapest real data this project can obtain.
    if backgrounds_dir is not None:
        bg_paths = sorted(
            p for p in Path(backgrounds_dir).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not bg_paths:
            console.print(f"[red]No images found in {backgrounds_dir}.[/]")
            raise typer.Exit(1)
        console.print(
            f"  {'local':8} {len(bg_paths):>4} backgrounds from {backgrounds_dir}\n"
        )
    else:
        wanted_bg = max(60, frames // 3)
        picks = index.backgrounds(limit=wanted_bg * 4, seed=seed)
        with console.status("fetching backgrounds…") as status:
            def bg_progress(i: int, total: int, ok: int) -> None:
                status.update(f"fetching backgrounds: {i}/{total} ({ok} kept)")

            bg_paths = []
            for _, path in lila.fetch_many(picks, work_dir / "images", bg_progress):
                img = cv2.imread(str(path))
                if img is not None and (not night_only or is_night_ir(img)):
                    bg_paths.append(path)
                if len(bg_paths) >= wanted_bg:
                    break

        if not bg_paths:
            console.print("[red]No usable backgrounds could be fetched.[/]")
            raise typer.Exit(1)
        label = "night IR" if night_only else "any"
        console.print(f"  {'empty':8} {len(bg_paths):>4} backgrounds ({label})\n")

        if night_only:
            console.print(
                "[yellow]Note:[/] these are outdoor wildland scenes. They give correct "
                "night-IR\nstatistics but the wrong furniture. Point [bold]--backgrounds-dir[/] "
                "at your own\novernight footage to close that gap — backgrounds need no "
                "animal in them.\n"
            )

    composer = Composer(cfg)
    samples: list[tuple[str, object]] = []
    with console.status("compositing…") as status:
        for i in range(frames):
            bg = bg_paths[i % len(bg_paths)]
            sample = composer.build(bg, fetched)
            if sample is not None:
                samples.append((f"exv_{i:06d}.jpg", sample))
            if i % 25 == 0:
                status.update(f"compositing {i}/{frames}")

    stats = write_coco(samples, Path(out), wanted, lila.CITATION)

    result = Table(title="Dataset written", title_justify="left", header_style="bold")
    result.add_column("Metric")
    result.add_column("Value", justify="right")
    for key, value in stats.items():
        result.add_row(key.replace("_", " "), f"{value:,}")
    console.print(result)

    sizes = [px for _, s in samples for px in s.target_px]
    console.print("\n[bold]Target size distribution[/]")
    console.print(size_histogram(sizes))
    console.print(
        f"\n[green]Wrote[/] {out}\n"
        f"[dim]Boxes are ground truth, not pre-annotations — the compositor "
        f"knows where it placed each animal.[/]"
    )


@app.command("train")
def train_cmd(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory from build-dataset")],
    out: Annotated[Path, typer.Option(help="Where checkpoints and the report go")] = Path(
        "out/model"
    ),
    epochs: Annotated[int, typer.Option(help="Training epochs")] = 30,
    batch_size: Annotated[int, typer.Option(help="Images per step, per GPU")] = 8,
    lr: Annotated[float, typer.Option(help="Peak learning rate")] = 1e-4,
    image_size: Annotated[
        int, typer.Option(help="Detector input size; 640 matches the stage-C tile")
    ] = 640,
    checkpoint: Annotated[
        str, typer.Option(help="rtdetr-v2-r18 | rtdetr-v2-r50 | rtdetr-r18 | rtdetr-r50")
    ] = "rtdetr-v2-r18",
    workers: Annotated[int, typer.Option(help="Dataloader workers")] = 4,
    no_amp: Annotated[bool, typer.Option("--no-amp", help="Disable mixed precision")] = False,
) -> None:
    """Fine-tune an Apache-2.0 detector on a scale-corrected dataset.

    Never an Ultralytics model: that family is AGPL-3.0, and shipping it inside
    software sold to clients triggers network copyleft. RT-DETR and the
    transformers library serving it are both Apache-2.0.
    """
    try:
        from .train.train import CHECKPOINTS, TrainConfig, train
    except ImportError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    if checkpoint not in CHECKPOINTS and "/" not in checkpoint:
        console.print(f"[red]Unknown checkpoint {checkpoint!r}.[/] Available:")
        for name, repo in CHECKPOINTS.items():
            console.print(f"  {name:16} {repo}")
        raise typer.Exit(1)

    from .train.data import CocoSet

    coco = CocoSet.load(dataset)
    profile = coco.size_profile()
    if not profile:
        console.print("[red]Dataset has no annotations.[/]")
        raise typer.Exit(1)

    table = Table(title="Dataset", title_justify="left", header_style="bold")
    table.add_column("Property")
    table.add_column("Value", justify="right")
    table.add_row("frames", f"{len(coco):,}")
    for name, count in coco.class_counts().items():
        table.add_row(f"  {name}", f"{count:,}")
    table.add_section()
    for key in ("p5", "median", "p95"):
        table.add_row(f"target px {key}", f"{profile[key]:.0f}")
    table.add_row("at or below 40 px", f"{profile['frac_at_or_below_40px']:.1%}")
    console.print(table)

    if profile["median"] > 120:
        console.print(
            "\n[yellow]Median target size is above 120 px.[/] That is drifting back "
            "toward the\nsource corpus bias this pipeline exists to correct — the "
            "model will be\ntrained on a regime the cameras do not deliver."
        )

    cfg = TrainConfig(
        epochs=epochs, batch_size=batch_size, lr=lr, image_size=image_size,
        checkpoint=checkpoint, num_workers=workers, amp=not no_amp,
    )
    console.print(f"\n[bold]{checkpoint}[/] · {epochs} epochs · batch {batch_size}\n")

    with console.status("training…") as status:
        def on_epoch(entry: dict) -> None:
            status.update(f"epoch {entry['epoch']}/{epochs} · loss {entry['loss']:.4f}")
            console.print(f"  epoch {entry['epoch']:>3}  loss {entry['loss']:.4f}")

        train(dataset, out, cfg, on_epoch=on_epoch)

    console.print("\n" + (Path(out) / "report.txt").read_text(encoding="utf-8"))
    console.print(f"\n[green]Wrote[/] {out}")


@app.command("evaluate")
def evaluate_cmd(
    model_dir: Annotated[Path, typer.Argument(help="Directory holding the checkpoint")],
    dataset: Annotated[Path, typer.Argument(help="Dataset to evaluate against")],
    score: Annotated[float, typer.Option(help="Confidence threshold")] = 0.35,
    min_recall: Annotated[
        float, typer.Option(help="Recall required for a band to count as usable")
    ] = 0.85,
) -> None:
    """Report recall by target pixel size — the metric that decides deployment.

    Not mAP. mAP averages over object sizes and hides the only question that
    matters here: does the detector work at the size inherited CCTV delivers?
    """
    try:
        from .train.data import CocoSet
        from .train.evaluate import report
        from .train.train import _require, predict_and_evaluate
    except ImportError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    torch = _require("torch")
    transformers = _require("transformers")

    checkpoint = model_dir / "checkpoint" if (model_dir / "checkpoint").exists() else model_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = transformers.AutoImageProcessor.from_pretrained(checkpoint)
    model = transformers.AutoModelForObjectDetection.from_pretrained(checkpoint).to(device)

    coco = CocoSet.load(dataset)
    with console.status(f"evaluating {len(coco):,} frames on {device}…"):
        result = predict_and_evaluate(model, processor, coco, device, score_threshold=score)

    console.print(report(result, min_recall=min_recall))


@app.command("init-config")
def init_config(
    out: Annotated[Path, typer.Option(help="Where to write the config")] = Path("cameras.yaml"),
) -> None:
    """Write a starter camera config."""
    if out.exists():
        console.print(f"[yellow]{out} already exists — not overwriting.[/]")
        raise typer.Exit(1)

    template = {
        "out_dir": "out",
        "db_path": "out/expressvision.db",
        "cameras": [
            {
                "id": "cam-001",
                "name": "Dock threshold",
                "site_id": "site-001",
                "source": "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101",
                "frame_stride": 2,
                "lens_mm": 6.0,
                "resolution_w": 1920,
                "max_detect_range_m": 12.0,
                "notes": "Main stream only — the sub-stream is D1 and loses small targets.",
            }
        ],
    }
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    console.print(f"[green]Wrote[/] {out}")
    console.print("Keep real credentials in cameras.local.yaml — it is gitignored.")


if __name__ == "__main__":
    app()
