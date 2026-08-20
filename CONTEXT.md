# Project context

Orientation for anyone — human or model — picking this up cold. It carries the
things that are **not** recoverable from the code: why decisions were made, what
was already tried and rejected, and what is currently blocking.

Last updated: **2026-08-13**

---

## What this is

**ExpressVision** — passive video analysis for pest detection, identification and
movement monitoring, built for **Express Pesticides**.

It turns existing CCTV into a continuous pest-surveillance instrument: detect
rats, mice and other pests on live or recorded streams, track their movement,
produce evidence clips, and generate analytics that drive IPM decisions.

**Assumed commercial shape:** a multi-tenant service Express Pesticides sells to
*their* clients — many sites per client, across food processing, warehousing,
hospitality, healthcare, residential, factories, agriculture and waste. Eight
named industries pointed that way. **This assumption is unconfirmed**; if it is
internal-only tooling, the tenancy layer and client-facing reporting collapse out
and roughly six weeks come off the roadmap.

### Where to find things

| Document | Contains |
|---|---|
| [architecture.html](architecture.html) | Formal specification, 16 sections. The reference. |
| [docs/architecture-explained.html](docs/architecture-explained.html) | Same system in plain language, with system / dataflow / workflow diagrams. Start here to explain it to anyone. |
| [docs/how-it-works.html](docs/how-it-works.html) | Short non-technical overview |
| [README.md](README.md) | Setup, running the collector, GPU runbook, testing |
| [docs/footage-request.md](docs/footage-request.md) | Exactly what we need from the client |
| [CHANGES.md](CHANGES.md) | What is planned next |
| [experiments/roboflow_eval/FINDINGS.md](experiments/roboflow_eval/FINDINGS.md) | Third-party model evaluation |
| [company_ans.txt](company_ans.txt) | The client's original answers — the source requirements |

---

## The framing that drives everything

**Almost nothing happens, and when it does it is small, fast, and dark.**

A warehouse camera sees ~650,000 frames in a twelve-hour night. Perhaps twenty
contain a rat. That rat is 25 cm long, moving at 1.5 m/s, lit only by an infrared
illuminator, and gone in four seconds.

Every significant design decision follows from that sentence. If a proposed
change does not respect it, the change is probably wrong.

Three consequences:

1. **Compute must be gated.** Running a detector on every frame is ~26× more GPU
   than the job needs. Cheap motion gating first, expensive inference only on
   candidate regions.
2. **Never alert on a single frame.** A frame-level detection is a coin flip. A
   *track* — same object, consistent class, plausible motion across many frames —
   is evidence. All alerting happens at track level.
3. **Pixels on target rule everything.** No model recovers a rat rendered at
   14 px. Camera placement, lens and shutter speed set the accuracy ceiling
   before any ML exists.

---

## Decisions that must not be quietly undone

These were expensive to arrive at. Several were bugs discovered the hard way.

### Vision pipeline

**Tile at native resolution; never downscale the whole frame.** The standard
pipeline resizes each frame to the detector's input size. That is fine for people
and fatal for pests — a 1920 px frame squeezed to 640 turns a 40 px rat into
13 px. Because the gate has already localised the region, we cut a 640 px tile
1:1 from the full-resolution frame instead. Same GPU work, three times the pixels
on target. This is the single most consequential CV decision in the system.

**Track *extent*, not net displacement, is the spatial gate.** Net displacement
looks reasonable and is wrong in both directions: a plant fragment drifting one
way for half a second has enough net displacement to look purposeful, while a rat
that works along a wall and returns has almost none. Extent — the diagonal of the
bounding box containing the whole trajectory — separates them correctly. This
change took false positives on the fixture from 15 to 0.

**The tracker needs a distance fallback, not just IoU.** A small fast animal
moves further than its own body length per frame, so consecutive detections have
*zero* IoU and no track ever forms. This is the normal case for a rat on a
low-frame-rate stream, not an edge case, and it failed silently — the gate fired,
tiles were inferred, detections were produced, and nothing reached the validator.
Association now falls back to centre distance gated on plausible travel.

**Evidence boxes come from the frame they are drawn on.** The keyframe is chosen
at the track's clearest moment; using the track's *final* box put an 8 px sliver
(the animal exiting frame) on top of a mid-track image. Those become
pre-annotations a human is told to trust, so a wrong box is worse than none.

**The gate needs two guards beyond background subtraction.** Global-change (a
light switch, IR-cut toggle at dawn, a bumped camera) and tamper/defocus (a
spider web across the lens). The second kills more outdoor analytics than any
other single cause, and must surface as a maintenance alert rather than as
silence that reads like "no pests".

### Data and models

**The default detector is a passthrough that keeps everything.** At this phase the
collector's job is to *harvest* candidates for labelling. A model trained on
someone else's domain would silently discard exactly the unusual examples worth
having.

**Rejected tracks are persisted with their full feature vectors.** Hard negatives
are worth more per example than positives when training the plausibility
classifier. Discarding them means paying for a second collection round later.

**The verify button is the training pipeline.** Operator Confirm / Reject /
Reclassify is not a side feature — it is the mechanism by which the system
improves after delivery. It shipped before the review UI did.

**Uptime normalisation is mandatory.** A camera offline for three nights produces
zero events, and an un-normalised dashboard renders that as *zero pest activity*
— the most dangerous possible false reassurance in a food plant. Every count is
divided by observed camera-hours.

### Deployment

**Sites must work with the uplink down.** Detection, tracking, validation,
evidence capture and local alerting all run on-site. The cloud aggregates and
presents; it is not required for a site to do its job. Any proposal that puts
inference behind a network call breaks this.

**Nothing AGPL ships to a client site.** MegaDetector runs through
`PytorchWildlife`, which depends on `ultralytics` and `yolov5` — both AGPL-3.0 —
whichever weights are selected. That is fine for internal harvesting and
labelling (internal use), but the production detector must be a separately
trained model exported to ONNX and served through ONNX Runtime or TensorRT, with
none of that stack in the edge image. The `Detector` protocol already makes this
a packaging rule rather than a code change.

---

## Hard constraints

| Constraint | Why |
|---|---|
| ≥ 40 px on target to detect, ≥ 80 px to identify species | Below that, no model recovers the animal |
| Night shutter floor 1/250 s | Slower turns a moving rat into a smear; most cameras default to 1/8 s |
| < 0.5 false alerts per camera-night | Above this the client mutes the channel and the product is dead |
| Main stream only, never sub-stream | Sub-streams are ~D1; a rat is ~12 px and unrecoverable |
| Face blur before any clip leaves site | Hospitals, hospitality, residential; DPDP Act 2023 |
| On-premises deployment option | Some healthcare and food clients will forbid video leaving the building |

---

## Current state

Stages A–F of the cascade run end to end, verified against synthetic footage with
known ground truth. **71 tests pass, `ruff` clean.**

| Stage | State |
|---|---|
| A Decode + downscale | file and RTSP with reconnect |
| B Motion gate | incl. global-change and tamper guards |
| C Tiling | native resolution |
| C Detector | `motion` passthrough + MegaDetector v6 adapter, verified |
| D Tracking | IoU + velocity + distance fallback |
| E Track validation | heuristic v1 |
| F Event assembly | clips, keyframes, SHA-256, sidecar geometry |
| G Cross-camera association | **not started** |
| — Store + verdict loop | CLI + self-contained HTML review sheet |
| — Benchmark | ready, awaiting GPU |

**No custom model has been trained.** That is deliberate and blocked on data.

---

## What is blocking

**Real overnight footage from Express Pesticides.** Everything model-related
waits on it. There is no public dataset of rats in Indian warehouses at 2 a.m. on
infrared CCTV, and the detector has to be taught what these sites look like.

The ask is specified in [docs/footage-request.md](docs/footage-request.md). Two
items are quick and unblock the rest: the camera list, and written permission to
share footage containing staff.

Every week without footage is a week added to the schedule, and it is not
recoverable by working harder on the software — the software is waiting.

---

## Things already evaluated and rejected

Recorded so they are not re-investigated.

**Public pretrained rodent detectors.** None worth building on. Roboflow Universe
rat/pest sets are a few hundred images each, mostly daylight or clip-art.
Academic rodent-YOLO work (Rat-YOLO, Cosevare) is trained on laboratory arenas —
overhead camera, white floor, single animal, controlled light — and weights are
largely unpublished. MegaDetector v6 is the only viable bootstrap.

**`thermal-rodent-detection-xkfep/1` (Roboflow).** Evaluated 2026-08-13 across 46
controlled images plus size-sweep and tiling-rescue follow-ups. Verdict: **C —
benchmarking only; D for production.** It needs ~130–180 px on target against our
40 px design floor (≈14× more cameras for the same floor area), calls a pigeon a
rodent at 0.729 confidence, emits a single `rodent` class, cannot be fine-tuned
from (Roboflow serves other users' Universe models via API only), and would
require an Enterprise plan for deployment outside their ecosystem. Full writeup:
[experiments/roboflow_eval/FINDINGS.md](experiments/roboflow_eval/FINDINGS.md).

**One genuinely useful result came out of it:** near-IR transfer works. A
competent rodent detector scored 3/3 at 0.790 mean confidence on simulated
near-IR, statistically indistinguishable from daylight. That suggests we will not
need separate day and night models, and that daylight rodent imagery has transfer
value. Worth keeping even though the model is not.

**Per-insect detection of flying insects in open air.** Not reliably solvable
with room cameras — near the lens an insect is a blurred smear, further away it
is sub-pixel. Reframed to a Flying Insect Activity Index plus camera-on-glue-board
counting at insect light traps, where the insects are static, in-plane and
countable. This should be agreed in the contract before build.

---

## Open questions for the client

Four change real decisions:

1. **Multi-tenant service, or internal Express tooling?** Assumed the former.
2. **Will any client require on-premises with no video leaving the building?**
   Assumed yes for healthcare and some food processors.
3. **Budget for supplementary cameras** where the survey finds existing ones
   cannot resolve pests.
4. **Pilot scale** — how many sites and cameras, across which industries.

Three are useful to know early: existing NVR makes and whether RTSP/ONVIF is
enabled; whether there is a pest-control CRM to integrate with; which sites
already run insect light traps or bait stations on a documented schedule.
