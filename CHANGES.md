# Changes

Planned work, in order, with the reason and the exit criterion for each. Read
[CONTEXT.md](CONTEXT.md) first for why the project is shaped the way it is.

Last updated: **2026-08-18**

**Convention:** move an item from *Next* to *Landed* only when its exit criterion
is met, not when the code is written. Add the date and keep the one-line reason —
in six months the reason is the part worth having.

---

## Next

### 0. Send the footage request — nothing else unblocks it

Still not sent as of 2026-08-18. Everything model-related waits on this, and no
amount of building substitutes.

A ready-to-send message is at
[docs/footage-request-email.md](docs/footage-request-email.md). It deliberately
asks for **two quick things first** — the camera list, and written permission —
because both take minutes and unblock planning without anyone exporting hundreds
of gigabytes.

- **Exit criterion:** camera list received for at least two sites.
- **If no reply in a week:** chase *only* the camera list. A follow-up asking for
  everything again is easy to defer; one asking for a single list is easy to
  answer.

### 1. GPU baseline — before any architecture change

The college GPU server becomes available **2026-08-14**. Benchmark first. The
point is a measured baseline, not a faster pipeline; capacity decisions should
follow numbers, not guesses.

```bash
uv run exv doctor          # must report "GPU ready" before proceeding
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install nvidia-ml-py
uv run pytest -q
uv run exv benchmark --device cuda --image-sizes 640,1280 --half \
    --max-frames 600 --out out/benchmark-gpu.json
uv run exv run data/fixture.mp4 --detector megadetector --device cuda
```

- **Exit criterion:** `out/benchmark-gpu.json` committed, and the fixture run
  still produces **5 events**. Behaviour matters more than throughput — if event
  count changed, the GPU path changed behaviour and that is the bug to chase.
- **Watch for:** `exv doctor` must not report a CPU-only torch build.
  PytorchWildlife accepts a `device` argument and never applies it (the line is
  commented out in its source), so the benchmark verifies where the weights
  actually are rather than trusting what was requested.
- **How to read it:** if `detect` is still ~98% of pipeline time, GPU is the
  right lever and streams scale roughly linearly. If `gate` or `decode`
  dominates, more GPUs buy nothing and the next work is NVDEC hardware decode.
  If GPU utilisation is low but FPS is flat, the bottleneck is feeding the GPU —
  batch more tiles per call.

CPU baseline for comparison: motion-only 186 FPS (gate 59% of time),
MegaDetector@640 2.7 FPS (detect 98%, 630 ms/tile).

### 2. Footage arrives → survey the cameras, *then* harvest

**Run `exv survey` on the first clips before anything else.** It answers the
project's highest-rated risk per camera — can this camera physically see a
rodent — and separates defects that are free to fix (night shutter, wrong
stream, dirty lens) from ones that cost money (too far, lens too wide).

```bash
uv run exv survey "D:/footage/site-041/*.mp4" \
    --hfov 78 --nearest-m 2 --furthest-m 14 --json out/survey-041.json
```

- **Exit criterion:** a usable / reconfigure / replace verdict per camera, and a
  supplementary-camera bill of materials for the ones that fail.
- **Expect some failures.** Most CCTV is aimed at people at head height. Saying
  so early, with numbers, is the point.

### 3. Harvest

Blocked on the client. See [docs/footage-request.md](docs/footage-request.md).

Run the collector in `motion` mode — deliberately over-triggering — so the
labelling queue contains the unusual cases a camera-trap model would discard.
Switch to `megadetector` only once the backlog makes labelling throughput the
bottleneck.

- **Exit criterion:** ≥ 3 nights from ≥ 2 contrasting sites processed, events in
  the store, funnel numbers recorded against real footage.
- **First thing to check:** the real gate pass rate. Single digits is expected
  overnight; much higher means the camera is seeing something constant and needs
  per-camera tuning.

### 4. Labelling

`exv review --html` → verdicts → `exv import-verdicts` → `exv export-coco` into
CVAT or Label Studio for correction.

- **Exit criterion:** 8,000–15,000 labelled instances for rodent, plus human.
- **Note:** exported boxes are *pre-annotations*, not ground truth. Say so to
  whoever labels, or they will trust them.

### 5. First custom detector

Rodent and human classes only. Apache-2.0 architecture (RT-DETRv2, D-FINE or
YOLOX) — **not** an Ultralytics model, see the licence boundary in CONTEXT.md.

- **Exit criterion:** ≥ 85% rodent recall at ≤ 1 false alert per camera-night on
  a held-out night from the pilot site.
- **Measure it on the same conditions** the Roboflow model was measured on —
  `experiments/roboflow_eval/testset.py` and `followup.py` are model-agnostic and
  the size sweep is the method. **40 px is the number to beat.**

### 6. Replace the heuristic validator with a learned one

Swap the hand-tuned gates in `validator.py` for a small gradient-boosted
classifier over the same features. The features are already computed and
persisted for every rejected track, so no new data collection is needed — this
was the reason for storing them.

- **Exit criterion:** false alerts per camera-night below the heuristic baseline
  on the same footage.

### 7. Stage G — cross-camera association

One rat crossing three overlapping views is currently three events, which
inflates every downstream analytic.

- **Blocked on:** a real multi-camera site. Cannot be built meaningfully against
  the single-camera fixture.

---

## Later

Deliberately after there is a working detector. Building these against motion
blobs would mean tuning analytics on noise.

- **Zones and homography** — operator draws polygons per camera view and marks
  which frame edges are physical boundaries. Turns "camera 4" into "the dock
  threshold" and makes entry-point inference trustworthy.
- **Entry-point inference** — DBSCAN over track origins in floor-plane
  coordinates. Directly answers the client's entry-point requirement, needs no
  extra hardware.
- **IoT** — temperature/humidity and door-contact sensors over LoRaWAN, then
  correlation. Door-contact is the highest-value one: it ties pest ingress to
  operational behaviour.
- **Analytics and reports** — heat maps, repeat-visit clustering, period
  comparison, IPM recommendation engine built with Express technicians.
- **ILT glue-board counting** — the reliable half of the flying-insect answer.
- **Production packaging** — ONNX export, TensorRT engines per device class, edge
  image with no torch, no ultralytics, no PytorchWildlife.
- **Multi-tenant hardening, mobile app, CRM integration.**

---

## Not doing, and why

Recorded so these are not reopened without new information.

- **`thermal-rodent-detection-xkfep/1`** — needs ~130–180 px on target against
  our 40 px floor, ≈14× more cameras for the same area. Calls a pigeon a rodent
  at 0.729. Cannot be fine-tuned from. Requires an Enterprise plan for deployment
  outside Roboflow's ecosystem, and needs the internet, which breaks
  edge-autonomy. Full writeup in
  [experiments/roboflow_eval/FINDINGS.md](experiments/roboflow_eval/FINDINGS.md).
- **Any hosted/cloud detector in production** — same reasons: breaks
  edge-autonomy, sends client footage to a third party, ~4 hours of network
  round-trip per camera-night.
- **Ultralytics YOLO in the shipped product** — AGPL-3.0. Fine internally, not in
  a client edge image.
- **Per-insect flying-insect detection in open air** — not reliably solvable.
  Reframed to activity index plus glue-board counting.
- **Separate day/night models** — the Roboflow evaluation showed near-IR transfer
  works without retraining. Revisit only if real footage contradicts it.

---

## Landed

### 2026-08-18

- **Camera assessment tool** (`exv survey`) — answers the project's top-rated
  risk per camera from a sample of its footage: resolution, focus, sensor noise,
  motion blur, IR illumination evenness, compression, exposure stability, and
  usable detection range. Separates free fixes (shutter, stream, dirty lens) from
  expensive ones (too far, wrong lens). Needs no client footage to build, and
  turns the first delivery into a per-camera answer within hours.
- **Ready-to-send footage request email** — the request had not gone out, and it
  is the actual blocker.
- **Plain-language explainer** (`docs/how-it-works.html`) — short non-technical
  guide with diagrams. Not part of the software.
- **Architecture in plain language** (`docs/architecture-explained.html`) — the
  complete system with system-architecture, dataflow and workflow diagrams, and
  a module-to-step map. Companion to the formal spec, not a replacement.

### 2026-08-13

- **Roboflow model evaluation** — 46 controlled images plus size-sweep and
  tiling-rescue follow-ups, and a live integration into the cascade. Verdict C/D.
  Produced one keeper: near-IR transfer works. Isolated in `experiments/`;
  nothing in `src/` imports it.
- **Review and labelling workflow** — self-contained keyboard-driven HTML sheet,
  verdict import, COCO pre-annotation export. Verified end to end.
- **GPU readiness** — `exv doctor`, `exv benchmark`, device/fp16 plumbing, and a
  runtime probe that reads the device back from the model's own tensors rather
  than trusting what was requested.
- **Tracker distance fallback** — small fast animals outran their own bounding
  box and formed no tracks at all. Silent failure.
- **Validator switched to trajectory extent** — false positives on the fixture
  went from 15 to 0.
- **Keyframe box fixed** — was taking the track's final position, producing 8 px
  slivers as pre-annotations.
- **End-to-end regression tests** — assert one event per rodent crossing against
  the fixture's own plan, so changing the fixture cannot silently weaken them.
  These are the guard rail for GPU work.
- **MegaDetector v6 integration** — verified against real animal photographs.
  Docs advertise variant names the released package rejects; the five that
  actually load are recorded in the README.
- **Collector pipeline, stages A–F** — the initial build.
- **Architecture and footage request** documents.
