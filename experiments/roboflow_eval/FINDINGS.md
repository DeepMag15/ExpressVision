# Evaluation: `thermal-rodent-detection-xkfep/1`

**Verdict: C — useful only for benchmarking/comparison.** For the production
pipeline specifically, treat it as D.

Evaluated 2026-08-13 against 46 controlled test images plus two follow-up
experiments (size sweep, tiling rescue) and a live integration into our cascade.
Raw results: `out/roboflow_eval/results.json`, `out/roboflow_eval/followup/`.

---

## What it actually is

| Property | Observed |
|---|---|
| Classes | **`rodent` only** — a single class, no rat/mouse distinction |
| Reachable | Yes, via `serverless.roboflow.com` |
| Server-side compute | ~8 ms per image |
| Round-trip latency | mean 1,344 ms · median 612 ms · p95 3,587 ms · max 4,128 ms |
| Weights | Not downloadable (see fine-tuning, below) |

The name says *thermal*. **It is not a thermal-only model** — see below. No public
Universe page, dataset size, or training-set description could be found for this
slug; the `-xkfep` suffix is Roboflow's auto-disambiguator for duplicate project
names, which usually indicates a fork.

---

## The hypothesis I had, and why it was wrong

Going in, the expectation was that a "thermal" model would fail on our imagery,
because thermal (LWIR, passive heat, bright blob on cool background) and near-IR
CCTV night mode (active IR LEDs, greyscale photograph with texture and shadows)
are different sensing modalities, not different lighting conditions.

**That prediction was wrong, and the result is good news.**

| Modality | n | detected | rate | mean conf |
|---|---|---|---|---|
| visible | 24 | 16 | 67% | 0.768 |
| near-IR simulated | 6 | 3 | 50% | 0.790 |
| thermal simulated | 6 | 4 | 67% | 0.694 |
| our synthetic fixture | 3 | 0 | 0% | — |

Broken out by condition, near-IR is **100%** (3/3, mean confidence 0.790) —
statistically indistinguishable from daylight (3/3, 0.799). Colour removal,
motion blur, low contrast and a ceiling-mounted viewpoint were also all 100%.

**The useful finding: a competent rodent detector transfers to near-IR
greyscale without retraining.** That de-risks our own training plan — it
suggests we will not need separate day and night models, and that daylight
rodent imagery has some transfer value to night footage.

---

## The actual blocker: it needs 3.75× more pixels than we can give it

The failure is not modality. It is **size**. A controlled size sweep on an
identical canvas, varying only the subject:

| Subject width | Rat | Mouse |
|---|---|---|
| 576 px (45%) | 0.855 | 0.892 |
| 384 px (30%) | 0.886 | 0.907 |
| 256 px (20%) | 0.748 | 0.893 |
| 179 px (14%) | 0.073 ⚠ | 0.826 |
| 128 px (10%) | **miss** | 0.570 |
| 89 px (7%) | **miss** | **miss** |
| 64 px (5%) | **miss** | **miss** |
| 25 px (2%) | **miss** | **miss** |

**Detection floor: roughly 130–180 px on target.** Our architecture (§2) is
designed around a 40 px floor. Working that through the same optics maths:

| Camera | Lens | Our design (40 px) | This model (~150 px) | Cameras needed |
|---|---|---|---|---|
| 2 MP | 2.8 mm | 5.0 m | 1.3 m | 14× |
| 2 MP | 6 mm | 12.0 m | 3.2 m | 14× |
| 4 MP | 4 mm | 9.9 m | 2.6 m | 14× |
| 4 MP | 8 mm | 22.0 m | 5.9 m | 14× |
| 8 MP | 4 mm | 14.8 m | 4.0 m | 14× |

Range scales inversely with the pixel floor and coverage area with its square,
so **the same floor area would need roughly 14× more cameras**. That is not a
tuning problem; it is a different product.

### Our tiling does not rescue it

Stage C crops a 640 px tile at native resolution around each motion region, so a
subject that is 5% of a 1280 px frame becomes 10% of a tile. That helps with
*fraction*, but the model needs absolute pixels, and cropping creates none.

| Subject | Whole frame | 640 px tile | Result |
|---|---|---|---|
| mouse 10% | 0.570 | 0.803 | both hit |
| mouse 7% | miss | 0.466 | **rescued** |
| mouse 5% | miss | miss | both miss |
| rat 10% → 2% | miss | miss | both miss (all sizes) |

**Tiling rescued 1 of 10 cases.** It moves the floor from ~128 px to ~89 px —
still more than double what we need.

For contrast, **our own motion gate emitted an ROI at every size tested, down to
25 px.** The gate is more sensitive than this detector by a wide margin, so the
detector — not our pipeline — is the limiting stage.

---

## False positives are severe

**6 of 7 control images produced a `rodent` box.**

| Control image | Confidence | Actually |
|---|---|---|
| Pigeon (daylight) | **0.729** | a bird |
| Pigeon (near-IR) | 0.673 | a bird |
| Cat (near-IR) | 0.570 | a cat |
| Cat (daylight) | 0.249 | a cat |
| Real thermogram ×2 | 0.129 | no animal at all |
| Empty warehouse floor | — | correctly clean |

A pigeon scoring **0.729** is higher than most of the true rodent detections in
this run. Our architecture requires `bird` as a separate L2 class and treats
species confusion as a trust problem; a detector that calls every bird a rodent
at high confidence would generate exactly the false pest alerts §9 is built to
prevent.

The empty floor was correctly clean, so it is not indiscriminate — it is
specifically confusing other animals for rodents.

### Box localisation is unstable

Same subject, same image, different transformation:

| Image | Box as % of frame |
|---|---|
| `brown_rat__visible` | 51.8% |
| `brown_rat__nir_night` | 74.9% |
| `brown_rat__thermal_whitehot` | 89.1% |

The box inflates as the image degrades. Our tracker associates on IoU and our
validator rejects tracks whose box area swings (`area_cv`), so unstable boxes
would both weaken association and trigger spurious `unstable_size` rejections.

### Multiple rodents

0 of 3 detected in every multi-rodent frame. This is confounded with size —
each animal was 7% of frame width, already below the floor — so it is not
independent evidence of a multi-object weakness.

---

## Fine-tuning: not possible from this model

Roboflow serves other users' Universe models **via hosted API only**. The
"download trained model weights" feature applies to models trained in *your own*
workspace on a premium plan. There is no route to obtain these weights as a
fine-tuning starting point.

What *is* possible is forking or downloading the underlying **dataset**, licence
permitting — but that makes it a data source, not a base model. Given the model
name, that dataset is likely thermal, which is not our modality.

**So the premise "use it as a base model" does not hold, independent of accuracy.**

---

## Licensing and API restrictions

Roboflow's commercial licence tiers govern deployment method:

| Plan | Covers |
|---|---|
| Public | Roboflow Managed Cloud only |
| Core | + self-hosted Roboflow Inference Server |
| **Enterprise** | deployment outside the Roboflow ecosystem |

Edge nodes at client sites are, by definition, outside their ecosystem, so
production use would require an **Enterprise plan**. The underlying architecture
also carries its own licence, which Roboflow's commercial licence covers only
within their platform.

Separately, the individual Universe project's own licence would need checking —
and it could not be located for this slug.

---

## Inside our actual pipeline

Plugged into the real cascade at stage C via `adapter.py`, on 200 frames of our
fixture (nothing in `src/expressvision` was changed):

```
frames processed      200
passed motion gate    123
tiles inferred        139
API calls             139
detections              1     <- out of 139 tiles
tracks confirmed        0
events                  0

throughput           2.53 fps
mean call latency     553 ms
```

One detection across 139 tiles, and it never survived to a track. Our fixture's
rodent is ~32 px wide — well under the measured 130–180 px floor — so this is
exactly what the size sweep predicts, now confirmed end to end.

At ~25,000 tiles per camera-night that is **3.8 hours of round-trip latency per
camera per night**, before any compute, scaling linearly with camera count.

---

## Architectural incompatibility, independent of accuracy

Even a perfect hosted model would not fit the design:

- **It needs the internet.** Architecture §3 requires each site to keep
  detecting, recording and alerting with its uplink down. A cloud detector
  breaks that outright.
- **Client footage leaves the site.** §10 commits to keeping hospital,
  food-processing and residential video on-premises where required.
- **Latency.** At ~25,000 tiles per camera-night and a 612 ms median round trip,
  that is **~4 hours of pure network wait per camera per night**, scaling
  linearly with cameras. Server-side compute was 8 ms — 98% of the cost is the
  network.

---

## Assessment

**C — useful only for benchmarking/comparison**, and its benchmarking value has
largely already been extracted by this evaluation.

Not **A**: fails at our operating distances, single class, severe false
positives on other animals.

Not **B**: "base detector for fine-tuning" is not available — the weights cannot
be obtained. And as a temporary detector it is worse for us than MegaDetector,
which runs locally, covers animal/person/vehicle, and has no per-tile network
cost.

Not quite **D**, only because the evaluation produced one genuinely useful
result: **near-IR transfer works**. That finding is worth keeping even though
the model is not.

### What to do

1. **Do not integrate it**, even experimentally, beyond what is in
   `experiments/roboflow_eval/`. Nothing in `src/expressvision` imports it.
2. **Keep the finding, not the model.** Near-IR needing no separate model is a
   real de-risking of the training plan.
3. **Reuse the test harness.** The size sweep and modality ablation in
   `testset.py` and `followup.py` are model-agnostic. Run our own trained
   detector through exactly the same conditions — the 40 px floor is the number
   to beat, and this evaluation established the method for measuring it.
4. **Nothing changes about the plan.** Real footage from Express Pesticides is
   still the blocker; see `docs/footage-request.md`.
