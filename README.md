# ExpressVision

Passive video analysis for pest detection, identification and movement monitoring.
Built for Express Pesticides.

**New here?** Read [CONTEXT.md](CONTEXT.md) first — it carries the decisions and
their reasons, which the code cannot tell you.

- Short explainer, no background needed: [docs/how-it-works.html](docs/how-it-works.html)
- **The whole system in plain language, with flow diagrams**: [docs/architecture-explained.html](docs/architecture-explained.html)
- Why the project is shaped this way: [CONTEXT.md](CONTEXT.md)
- What happens next: [CHANGES.md](CHANGES.md)
- Formal architecture specification: [architecture.html](architecture.html)
- What we need from the client: [docs/footage-request.md](docs/footage-request.md)

---

## Where the project stands

Stages A–F of the cascade run end to end, verified against synthetic footage
with known ground truth. The bootstrap detector (MegaDetector v6) is wired in
and verified against real animal photographs. **No custom model has been
trained** — that waits on real footage.

| Stage | Component | State |
|---|---|---|
| A | Decode + downscale ([sources.py](src/expressvision/sources.py)) | file and RTSP with reconnect |
| B | Motion gate ([gate.py](src/expressvision/gate.py)) | incl. global-change and tamper guards |
| C | Native-resolution tiling ([tiling.py](src/expressvision/tiling.py)) | working |
| C | Detectors ([detect.py](src/expressvision/detect.py)) | `motion` passthrough + MegaDetector v6 |
| D | Tracking ([tracking.py](src/expressvision/tracking.py)) | IoU + velocity + distance fallback |
| E | Track validation ([validator.py](src/expressvision/validator.py)) | heuristic v1 |
| F | Event assembly ([assembler.py](src/expressvision/assembler.py)) | clips, keyframes, hashes |
| G | Cross-camera association | not started |
| — | Store + verdict loop ([store.py](src/expressvision/store.py)) | CLI + HTML review sheet |
| — | Benchmark ([benchmark.py](src/expressvision/benchmark.py)) | ready, awaiting GPU |

68 tests pass. `ruff` clean.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is installed by uv itself.

```bash
uv sync --extra dev          # collector, CLI, tests — no torch
uv sync --extra ml           # adds MegaDetector (torch, PytorchWildlife)
uv run pytest -q
```

### GPU setup

**The `ml` extra installs a CPU build of torch.** That is deliberate — it keeps
the default install small — but it means a GPU box needs one extra step, and
skipping it produces a pipeline that runs perfectly and silently uses no GPU.

```bash
# Check what you have first — this tells you exactly what is missing.
uv run exv doctor

# Then install torch built against your CUDA version.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Optional but recommended: real GPU telemetry during benchmarks.
uv pip install nvidia-ml-py

uv run exv doctor            # must now report "GPU ready"
```

Pick the index URL to match the driver: `cu121`, `cu124` or `cu126`. `nvidia-smi`
reports the maximum CUDA version the driver supports; the torch build must be
that or lower.

Requirements:

| Requirement | Why |
|---|---|
| NVIDIA driver ≥ 525 (for CUDA 12.x) | older drivers cannot load cu12 wheels |
| CUDA-built torch (`torch.version.cuda` not `None`) | a CPU wheel silently ignores `--device cuda` |
| ≥ 6 GB VRAM for `MDV6-yolov9-e` at 1280 | the compact variants fit in ~3 GB |
| `nvidia-ml-py` | utilisation and temperature; without it telemetry is omitted |

`exv doctor` checks all of these and prints the fix for whatever is missing.

---

## Assessing a camera

**Run this first on any footage from a new site.** It answers the question the
whole project hinges on — can this camera physically see a rodent — and splits
the problems into free fixes and expensive ones.

```bash
uv run exv survey "D:/footage/dock-01.mp4" \
    --hfov 78 --nearest-m 2 --furthest-m 14 --json out/survey.json
```

It reports resolution, focus, sensor noise, motion blur, IR illumination
evenness, compression, exposure stability and usable detection range, then a
verdict: **usable** / **usable after reconfiguration** / **replace or relocate**.
Everything failing carries a specific fix.

`--hfov` is preferred over `--lens-mm`: manufacturer field-of-view figures are
more reliable than computing one from focal length and an assumed sensor size.
Without either, image quality is still assessed but range is not.

---

## Running the collector

No footage yet? Generate a fixture whose answers are known — rodent crossings,
a swaying plant that must be rejected, a lighting change that must trip the
global-change guard.

```bash
uv run exv make-fixture --seconds 60
uv run exv run data/fixture.mp4
```

On real footage, with the bootstrap detector:

```bash
uv run exv run "D:/footage/night.mp4" --camera cam-004 \
    --detector megadetector --device cuda --image-size 640
```

On a live camera — use the **main** stream; sub-streams are typically D1 and
destroy small targets before the pipeline sees them:

```bash
uv run exv init-config       # writes cameras.yaml
uv run exv run-config cameras.yaml
```

### Detector modes

| Mode | Use when |
|---|---|
| `motion` (default) | Harvesting. Keeps every gated region, including the odd cases a camera-trap model would silently drop. |
| `megadetector` | Sorting a backlog. Filters to animals, making the labelling queue far denser. |

Both are exercised by the regression suite, so neither can rot.

---

## Reviewing and labelling

The verdict loop is how the system improves after delivery, so it has to be fast
enough that someone actually does it.

```bash
uv run exv review --html out/review.html --limit 200
```

Open it in a browser. `J`/`K` to move, `C` confirm, `R` reject, `1`–`7` to
reclassify, `U` to undo. Verdicts accumulate in the browser; **Export verdicts**
downloads a JSON file:

```bash
uv run exv import-verdicts verdicts.json
uv run exv stats
```

The sheet is self-contained — keyframes are inlined, no server, no network — so
it can be emailed to a technician who has no access to the collector's machine.

When there are enough confirmed events, export pre-annotations for correction in
CVAT or Label Studio:

```bash
uv run exv export-coco out/dataset
```

These are **pre-annotations, not ground truth**. The boxes come from motion ROIs
or MegaDetector proposals; a human corrects them. Starting from a roughly-right
box is several times faster than drawing each one cold.

---

## Benchmarking

```bash
uv run exv benchmark --device cuda --image-sizes 640,1280 --half
```

Reports FPS, per-tile latency, per-stage breakdown, GPU utilisation, VRAM, and
converts those into the number that matters: **streams per GPU**. Results are
written to `out/benchmark.json` — commit them so runs stay comparable.

Three properties make the numbers trustworthy:

- **CUDA work is synchronised around every timed section.** Without it a
  benchmark measures how fast Python enqueues work, not how fast the GPU
  finishes it, and reports impossibly good figures.
- **Warmup frames are excluded.** The first CUDA call pays for context creation
  and cuDNN autotuning — often seconds.
- **The device is read back from the model's own tensors.** PytorchWildlife
  accepts a `device` argument and never applies it (the line is commented out in
  its source), so the benchmark verifies where the weights actually are rather
  than trusting what was requested. A mismatch prints a red warning.

### Measured so far (CPU only — for comparison tomorrow)

i5-12500H, CPU torch, 1280×720 fixture:

| Configuration | Device | FPS | ms/tile | Streams @15fps | h per cam-night |
|---|---|---|---|---|---|
| motion-only | cpu | 186.4 | — | 8.7 | 1.0 |
| MDV6-yolov9-c @640 | cpu float32 | 2.7 | 630.4 | 0.1 | 67.4 |

Detection is 98% of pipeline time with the model attached; gating is 59% without
it. That is the expected shape, and it means GPU acceleration should move the
detector figure a long way before anything else becomes the bottleneck.

---

## Tomorrow: GPU runbook

Benchmark before changing anything. The point is a measured baseline, not a
faster pipeline.

```bash
# 1. Confirm the environment. Do not proceed if this does not say "GPU ready".
uv run exv doctor

# 2. Confirm behaviour is unchanged on the GPU box.
uv run pytest -q                      # 68 tests, no GPU required

# 3. Baseline. Motion-only first — it is the pipeline's ceiling.
uv run exv benchmark --device cuda --image-sizes 640,1280 --half \
    --max-frames 600 --out out/benchmark-gpu.json

# 4. Sanity-check the detector end to end on the GPU.
uv run exv make-fixture --seconds 60
uv run exv run data/fixture.mp4 --detector megadetector --device cuda
```

Step 4 should still produce **5 events** — one per crossing. If it does not, the
GPU path has changed behaviour, and that matters more than throughput.

What to read from the results:

- **If `detect` is still ~98% of pipeline time**, the GPU is the right lever and
  more/faster GPUs buy streams roughly linearly.
- **If `gate` or `decode` dominates**, adding GPUs buys nothing. The next work is
  hardware-accelerated decode (NVDEC) and moving the gate off the CPU.
- **If GPU utilisation is low (<40%) but FPS is flat**, the bottleneck is feeding
  the GPU, not the GPU itself — batch more tiles per call.
- **VRAM per instance** decides how many pipeline processes share one card.

Stream estimates assume the fixture's motion load and keep 30% headroom. Real
overnight footage gates in the single digits, so these are a floor — but real
sites also spike at shift changes. Re-measure on real footage when it arrives.

---

## Testing

```bash
uv run pytest -q                          # everything
uv run pytest tests/test_regression.py -q # end-to-end behaviour only
```

- `test_pipeline.py` — unit tests for each stage
- `test_regression.py` — end-to-end against synthetic footage with known ground
  truth. **These are the guard rail for GPU work**: they assert one event per
  rodent crossing, the plant rejected, the lighting change absorbed. Assertions
  are made against the fixture's own plan, so changing the fixture cannot
  silently weaken them.
- `test_review.py` — review sheet, verdict import, COCO export, GPU probing

---

## The bootstrap model

There is no public pretrained rodent detector worth building on:

| Option | Reality |
|---|---|
| **MegaDetector v6** (Microsoft AI for Good) | Animal / person / vehicle on camera-trap imagery, ~83% animal recall. **The only real option.** |
| Roboflow Universe rat/pest sets | A few hundred images each, mostly daylight or clip-art. Supplementary training data later; not a model. |
| Academic rodent-YOLO (Rat-YOLO, Cosevare) | Lab arenas — overhead camera, white floor, single animal, controlled light. Weights largely unpublished. |

MegaDetector detects `animal` / `person` / `vehicle`. It **cannot** tell a rat
from a cat — that is what fine-tuning on the client's own footage is for. What it
does is turn "all motion" into "mostly animals".

### Variants

Verified against **PytorchWildlife 1.3.0** by reading its source. The published
docs advertise a nine-variant zoo (`MDV6-apa-rtdetr-e`, `MDV6-mit-yolov9-c`) that
**the released package rejects**. These five actually load:

| Variant | Architecture | Note |
|---|---|---|
| `MDV6-yolov9-c` | YOLOv9 compact | default; fast |
| `MDV6-yolov9-e` | YOLOv9 extra @1280 | best recall, most VRAM |
| `MDV6-yolov10-c` | YOLOv10 compact | smallest |
| `MDV6-yolov10-e` | YOLOv10 extra @1280 | high recall |
| `MDV6-rtdetr-c` | RT-DETR compact | matches the production architecture |

---

## Where MegaDetector may run

**This matters more than the weights licence.** `PytorchWildlife` depends on
`ultralytics` and `yolov5` — both **AGPL-3.0** — whichever variant you select.
Picking permissive weights does not make the installed stack permissive.

- ✅ **Internal harvest and labelling.** Running it on our own machines to sort
  our own footage is internal use; copyleft triggers on distribution and on
  network interaction with third parties.
- ❌ **The edge node shipped to a client site.** Nothing from the `ml` extra may
  be installed there.

The production detector is a separately trained model exported to ONNX and served
through ONNX Runtime or TensorRT — no ultralytics, no PytorchWildlife, no torch.
The `Detector` protocol already keeps these interchangeable, so this is a
deployment-packaging rule, not a code change.

Ultralytics YOLO (v8/v11) is AGPL-3.0; RT-DETRv2, D-FINE and YOLOX are Apache-2.0
and comparable. Decide before the first production model goes in. See §11 of the
architecture.

---

## Next

Not started, and deliberately so — these wait on real footage:

1. **Train the custom detector.** Blocked on labelled data, which is blocked on
   footage. See [docs/footage-request.md](docs/footage-request.md).
2. **Cross-camera association** (stage G) — needs a real multi-camera site.
3. **Zones, homography, IoT correlation, analytics.** Building these against
   motion blobs would mean tuning analytics on noise.
