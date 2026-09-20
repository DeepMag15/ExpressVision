# Training data without the client

**Status: active.** Replaces `docs/footage-request.md`, which is now closed —
Express Pesticides will not be supplying footage.

The project was planned around client footage as the single source of rodent
imagery. That dependency is gone, and it was the critical path for everything
model-related. This document is the replacement, and the short version is:

> We do not need the client's footage to train a detector. We need **rodent
> appearance** and **our scene**, and those two things can be obtained
> separately, from different places, and combined.

---

## The insight that makes this work

A training image is two independent things:

| Component | Can we get it without the client? | How |
|---|---|---|
| **What a rodent looks like** in infrared, at night | **Yes** | 82,912 annotated rodents in public camera-trap data |
| **What our scene looks like** — a warehouse floor at 2 a.m. | **Yes** | Point our own IR camera at a dark room. No animal required. |

The second one is the part people assume is hard, and it is actually trivial:
**a background needs no rodent in it.** Anyone can film an empty storeroom
overnight. That is the half the client was never needed for.

The compositor in `src/expressvision/dataset/compose.py` joins them, and the
labels come out exact because we know precisely where we placed each animal.

---

## Source A — Channel Islands Camera Traps (public, permissive)

The Nature Conservancy, 2021, hosted by LILA BC. **CDLA-Permissive-1.0**, which
matters as much as the content: this project already refuses AGPL components in
anything shipped to a client, and a training corpus carries the same kind of
obligation. CDLA-Permissive imposes none that conflict with commercial use.

What it contains, all with bounding boxes, all 1920×1080 — the same resolution
as the CCTV this system reads:

| Class | Boxes | Why we want it |
|---|---|---|
| rodent | 82,899 | the target class, in infrared, at night |
| fox | 48,145 | four-legged non-target |
| bird | 11,082 | **the hard negative that matters most** |
| human | 5,929 | *unavailable — see below* |
| skunk | 1,071 | four-legged non-target |
| empty | 114,894 | real backgrounds, real sensor noise |

Bird earns its place specifically: the third-party detector we evaluated called
a pigeon a rodent at 0.729 confidence, and a corpus without birds cannot teach a
model not to.

**We never download the 86 GB image set.** The metadata is 18 MB and individual
images are addressable, so a build pulls only the few thousand frames it needs.

### The human class is withheld

Images containing people return HTTP 404 — LILA removes them from the public
download for privacy. The annotations exist; the pixels do not.

This is not a blocker, because human is the *easiest* class to collect
ourselves: walk past our own IR camera. No privacy question, exactly the right
domain, and it takes an evening.

---

## The problem with Source A, and the fix

Public camera-trap rodents are **the right appearance at the wrong scale**:

```
Channel Islands rodents:   p5 108 px   median 267 px   p95 506 px
Inherited CCTV delivers:                        40 px
```

A camera trap sits a metre from a burrow. Our cameras are mounted for people at
head height, six metres away. Training on the former and deploying on the latter
is precisely the mismatch we already measured in
`experiments/roboflow_eval/FINDINGS.md`, where an off-the-shelf rodent detector
held confidence to roughly 180 px and then collapsed to nothing.

**That evaluation now reads as an explanation, not just a rejection.** The model
probably failed below 130 px because its training data looked like this one —
biased to large, well-resolved subjects. The bias is in the data, not the
architecture.

So we correct the bias by resampling. `exv build-dataset` takes a real annotated
rodent, scales it **down** to a size drawn from our operating distribution, and
composites it into a night background.

Three rules are enforced in code and covered by tests:

- **Only ever downscale.** Upscaling invents sensor detail that will not exist
  at inference. `paste()` returns `None` rather than enlarge.
- **Concentrate on the decision boundary.** Uniform sampling over 24–140 px
  would spend most examples where detection is already easy. Roughly 45% of
  samples land in a tight band around the 40 px floor.
- **Degrade after compositing, not before.** Motion blur, sensor noise and JPEG
  artefacts belong to the camera, so they act on the assembled scene. Otherwise
  a suspiciously clean animal sits on a noisy background and the detector learns
  the seam.

Measured output from a real run:

```
  24-38  px |#########################         |  23
  38-53  px |##################################|  31   <- detection floor
  53-68  px |###########                       |  10
  68-82  px |#######                           |   6
  ...
  median 45 px   (source corpus median: 267 px)
```

---

## Source B — our own backgrounds (the part only we can supply)

Public wildlife data gives correct night-IR *statistics* and completely wrong
*furniture*. Channel Islands backgrounds are hillsides and scrub; our domain is
concrete, racking, pallets, drains and roller shutters.

Greyscaling a daylight frame does **not** produce a night frame either — daylight
has directional shadows and sky-lit fill; IR has point-source falloff from the
illuminator and a different noise floor. `is_night_ir()` rejects daylight frames
for exactly this reason.

So backgrounds come from us:

```bash
# Point the builder at your own overnight stills
exv build-dataset out/trainset --backgrounds-dir data/my-site-nights
```

**Shopping list:** one 4 MP IR bullet camera, roughly ₹2,000–4,000. Point it
along a wall–floor junction in a garage, stairwell, storeroom, back alley or
rubbish area. Leave it a week.

Even with no animal ever walking through, this delivers what the project is most
starved of: real negatives in our own domain. And if a rat, mouse, cat or bird
does pass, that is a genuine positive with our own ground truth.

---

## Source C — the existing synthetic generator

`src/expressvision/synth.py` stays. It is not training data and never was — it
is the regression fixture with ground truth known by construction, and it is the
guard rail that proves the cascade still behaves after a change.

---

## What this changes about the plan

| Was blocked on client footage | Now |
|---|---|
| Train a rodent detector | **Unblocked** — build the set, train, evaluate |
| Measure against the 40 px floor | **Unblocked** — the size sweep harness is model-agnostic |
| Validator training data | Partially — needs operator verdicts, which the console collects |
| Real gate pass rate | **Still needs real footage** — from our own camera, not the client's |
| Camera survey on real cameras | **Still needs cameras** — our own, or none |
| Stage G cross-camera | Still blocked — needs a genuine multi-camera site |

---

## Honest limitations

State these plainly; they are the first thing a reviewer will probe.

**Composited data is not field data.** It proves a detector can learn rodent
appearance at 40 px. It does not prove performance in a working food plant, and
a model trained only on composites will carry a domain gap of unknown size.

**The backgrounds are outdoor until we collect our own.** Until Source B exists,
the generated set has correct night-IR statistics over the wrong scenery.

**Blending is not physically exact.** Poisson cloning matches illumination
gradients, but it does not cast a shadow or reflect light onto the floor. At
40 px that is mostly below the noise floor; at 140 px it is visible.

**No held-out real test set exists yet.** Every number a model produces on this
data is train-distribution. The first real footage collected must be reserved as
a test set and never trained on.

---

## Commands

```bash
# Build a scale-corrected set from public data (first run fetches 18 MB metadata)
exv build-dataset out/trainset --frames 2000 --sources 600

# Preferred: use your own overnight footage as backgrounds
exv build-dataset out/trainset --frames 2000 --backgrounds-dir data/my-nights

# Include the hard negatives that break naive detectors
exv build-dataset out/trainset --classes rodent,bird,fox
```

**Attribution is mandatory** and is written into every `annotations.json`:

> The Nature Conservancy (2021): Channel Islands Camera Traps 1.0. The Nature
> Conservancy. Dataset. Licensed CDLA-Permissive-1.0 via LILA BC.
