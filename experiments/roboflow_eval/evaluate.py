"""Run the evaluation and report it in terms that answer the actual question.

Usage:
    $env:ROBOFLOW_API_KEY = "..."          # PowerShell
    uv run python -m experiments.roboflow_eval.evaluate

The report is organised by *condition*, not by image, because the decision does
not turn on how many images were hit — it turns on which conditions survive.
A model at 100% on daylight close-ups and 0% on near-IR is not a 50% model; it
is a model for a domain we do not have.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.roboflow_eval import testset
from experiments.roboflow_eval.client import (
    MissingApiKey,
    RoboflowDetector,
)

MODEL_ID = "thermal-rodent-detection-xkfep/1"
RODENT_WORDS = {"rat", "mouse", "mice", "rodent", "rats", "vermin", "pest"}


def is_rodent(label: str) -> bool:
    lowered = label.lower().strip()
    return any(word in lowered for word in RODENT_WORDS)


def run(
    model_id: str,
    out_dir: Path,
    confidence: float,
    rebuild: bool,
    fixture: Path | None,
) -> int:
    images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.json"

    print(f"model: {model_id}")
    print(f"output: {out_dir}\n")

    try:
        detector = RoboflowDetector(model_id, confidence=confidence)
    except MissingApiKey as exc:
        print(f"ERROR: {exc}")
        return 2

    # Resolve the model before spending time building a test set for it.
    print("probing model...")
    probe = detector.probe()
    if not probe.get("ok"):
        print(f"  model did not resolve: {json.dumps(probe, indent=2)[:900]}")
        print("\nIf this is a 404, the model ID or version is wrong, or the")
        print("project is private. If 401/403, the API key lacks access.")
        return 3
    print(
        f"  reachable · round trip {probe['latency_ms']:.0f} ms · "
        f"{probe['n_predictions_on_blank']} predictions on a blank image"
    )
    server_ms = (probe.get("raw") or {}).get("time")
    if server_ms is not None:
        print(
            f"  server-side compute {server_ms * 1000:.1f} ms — the rest is network"
        )
    print()

    if rebuild or not manifest_path.exists():
        print("building test set...")
        cases = testset.build(images_dir, fixture_video=fixture)
        testset.write_manifest(cases, manifest_path)
    else:
        cases = [
            testset.TestImage(
                name=c["name"], path=Path(c["path"]), subject=c["subject"],
                condition=c["condition"], modality=c["modality"],
                expect_rodent=c["expect_rodent"], notes=c.get("notes", ""),
                tags=c.get("tags", []),
            )
            for c in json.loads(manifest_path.read_text(encoding="utf-8"))
        ]
    print(f"  {len(cases)} test images\n")

    print("running inference...")
    records = []
    for i, case in enumerate(cases, 1):
        result = detector.infer(case.path)
        rodents = [p for p in result.predictions if is_rodent(p.label)]
        best = max(rodents, key=lambda p: p.confidence, default=None)

        records.append(
            {
                "name": case.name,
                "subject": case.subject,
                "condition": case.condition,
                "modality": case.modality,
                "expect_rodent": case.expect_rodent,
                "tags": case.tags,
                "notes": case.notes,
                "error": result.error,
                "latency_ms": round(result.latency_ms, 1),
                "n_predictions": len(result.predictions),
                "n_rodent": len(rodents),
                "best_conf": round(best.confidence, 4) if best else 0.0,
                "labels": sorted({p.label for p in result.predictions}),
                "boxes": [
                    {
                        "label": p.label,
                        "conf": round(p.confidence, 4),
                        "xyxy": [round(v, 1) for v in p.xyxy],
                    }
                    for p in sorted(
                        result.predictions, key=lambda p: -p.confidence
                    )[:5]
                ],
            }
        )

        mark = "!" if result.error else ("HIT " if best else "miss")
        conf = f"{best.confidence:.3f}" if best else "  -  "
        print(f"  [{i:>3}/{len(cases)}] {mark} {conf}  {case.name}")
        if result.error:
            print(f"          {result.error[:150]}")

    (out_dir / "results.json").write_text(
        json.dumps({"model_id": model_id, "records": records}, indent=2),
        encoding="utf-8",
    )
    report(records)
    print(f"\nfull results: {out_dir / 'results.json'}")
    return 0


def report(records: list[dict]) -> None:
    errors = [r for r in records if r["error"]]
    usable = [r for r in records if not r["error"]]

    print("\n" + "=" * 78)
    print("RESULTS BY MODALITY  — the question this evaluation exists to answer")
    print("=" * 78)

    by_modality: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        by_modality[r["modality"]].append(r)

    print(f"\n{'modality':<16}{'n':>4}{'detected':>10}{'rate':>8}   {'mean conf':>10}")
    print("-" * 60)
    for modality in ("visible", "nir_sim", "thermal_sim", "thermal_real",
                     "synthetic"):
        rows = [r for r in by_modality.get(modality, []) if r["expect_rodent"]]
        if not rows:
            continue
        hits = [r for r in rows if r["n_rodent"] > 0]
        confs = [r["best_conf"] for r in hits]
        rate = len(hits) / len(rows)
        mean_conf = statistics.fmean(confs) if confs else 0.0
        print(
            f"{modality:<16}{len(rows):>4}{len(hits):>10}{rate:>7.0%}   {mean_conf:>10.3f}"
        )

    print("\n" + "=" * 78)
    print("RESULTS BY CONDITION — where it breaks down")
    print("=" * 78)
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        if r["expect_rodent"]:
            by_condition[r["condition"]].append(r)

    rows = []
    for condition, group in by_condition.items():
        hits = [r for r in group if r["n_rodent"] > 0]
        rows.append((len(hits) / len(group), condition, len(hits), len(group),
                     statistics.fmean([r["best_conf"] for r in hits]) if hits else 0.0))
    rows.sort(reverse=True)

    print(f"\n{'condition':<38}{'hits':>10}{'rate':>8}{'conf':>8}")
    print("-" * 66)
    for rate, condition, hits, total, conf in rows:
        print(f"{condition:<38}{hits:>4}/{total:<5}{rate:>7.0%}{conf:>8.3f}")

    print("\n" + "=" * 78)
    print("FALSE POSITIVES — non-rodent images that produced rodent detections")
    print("=" * 78)
    controls = [r for r in usable if not r["expect_rodent"]]
    false_hits = [r for r in controls if r["n_rodent"] > 0]
    print(f"\n{len(false_hits)} of {len(controls)} control images produced a rodent box")
    for r in false_hits:
        print(f"  {r['name']:<40} conf {r['best_conf']:.3f}  ({r['subject']})")

    print("\n" + "=" * 78)
    print("CLASSES AND LATENCY")
    print("=" * 78)
    labels: set[str] = set()
    for r in usable:
        labels.update(r["labels"])
    print(f"\nclasses returned: {sorted(labels) or '(none — model detected nothing)'}")

    latencies = [r["latency_ms"] for r in usable]
    if latencies:
        ordered = sorted(latencies)
        print(
            f"latency: mean {statistics.fmean(latencies):.0f} ms · "
            f"median {statistics.median(ordered):.0f} ms · "
            f"p95 {ordered[int(0.95 * (len(ordered) - 1))]:.0f} ms · "
            f"max {ordered[-1]:.0f} ms"
        )
        per_night = statistics.fmean(latencies) * 25000 / 1000 / 3600
        print(
            f"at ~25k tiles per camera-night that is {per_night:.1f} hours of "
            f"round-trip latency per camera, before any compute"
        )

    if errors:
        print(f"\n{len(errors)} request(s) failed:")
        for r in errors[:5]:
            print(f"  {r['name']}: {r['error'][:120]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--out", type=Path, default=Path("out/roboflow_eval"))
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the test set")
    parser.add_argument(
        "--fixture", type=Path, default=Path("data/fixture.mp4"),
        help="Our own synthetic clip, for frames the pipeline actually sees",
    )
    args = parser.parse_args()

    if not os.environ.get("ROBOFLOW_API_KEY"):
        print("ROBOFLOW_API_KEY is not set.")
        print('  PowerShell:  $env:ROBOFLOW_API_KEY = "your-key"')
        print("  bash:        export ROBOFLOW_API_KEY=your-key")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    return run(
        args.model_id, args.out, args.confidence, args.rebuild,
        args.fixture if args.fixture.exists() else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
