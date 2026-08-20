# What we need from Express Pesticides

**Purpose:** everything below exists to train and validate the pest-detection
system on your actual sites. The software is built and running; it currently has
nothing real to look at.

**The short version:** a few nights of overnight CCTV recordings from two or
three contrasting sites, plus a one-page form per camera. Nothing else is
blocking.

---

## Why this is the critical path

There is no public dataset of rats in Indian warehouses at 2 a.m. on infrared
CCTV. The detector has to be taught what your sites look like, and that takes
labelled examples from those sites.

Every week without footage is a week added to the schedule, and the delay is not
recoverable by working harder on the software — the software is waiting.

---

## 1. Overnight footage — the priority

### What

| | |
|---|---|
| **Sites** | 2–3, deliberately different (see below) |
| **Cameras per site** | 2–4, chosen with us — not all of them |
| **Nights** | 3–5 consecutive, per site |
| **Hours** | 22:00 to 06:00 (or dusk to dawn) |
| **Stream** | **Main / primary stream.** Not the sub-stream |
| **Format** | Native NVR export, or MP4 (H.264) |

**Which sites.** The most useful pair is one food-processing plant or warehouse
and one hospitality or healthcare site. They differ in lighting, layout, clutter
and pest species, and a system that works on both will generalise. Two warehouses
teach us much less than one warehouse and one hotel kitchen.

**Which cameras.** We would like to choose these together after seeing a camera
list — the best ones look along floors and walls at loading docks, storage
aisles, waste areas and kitchens. A camera pointed at a car park teaches us
nothing.

### Critical: main stream, not sub-stream

Most NVRs record two streams per camera. The sub-stream is low resolution
(typically 704×576) and is what many export tools default to.

A rat at that resolution is roughly 12 pixels long. **It is not recoverable —
no software can find it.** Please confirm exports are the main/primary stream.

### Please do not

- ❌ Record a monitor with a phone. Compression and screen glare destroy exactly
  the detail we need.
- ❌ Trim to short clips around known sightings. We need the ordinary hours too —
  the empty footage is what teaches the system not to raise false alarms.
- ❌ Re-encode, downscale, or "compress for email".
- ❌ Convert to GIF or WhatsApp video.

Original files, exactly as the NVR wrote them, are ideal.

### Data volume — please plan for this

A main-stream camera produces roughly **10–20 GB per night**.

| Scope | Approximate size |
|---|---|
| 1 camera, 1 night | 10–20 GB |
| 4 cameras, 5 nights, one site | 200–400 GB |
| Two sites | 400–800 GB |

This is too large to email or upload over a typical site connection. **A portable
USB hard drive is the practical option** — we can supply one, or collect it.

If that is difficult, a good fallback is fewer cameras for more nights rather
than more cameras for fewer nights. Continuity matters more than coverage.

---

## 2. Camera information

One row per camera included in the export. Most of this is on a label on the
camera or in the NVR's device list; where it is not, an estimate is genuinely
useful — please do not leave a row blank because a figure is approximate.

| Field | Example | Why we need it |
|---|---|---|
| Camera ID / name in NVR | `Dock-01` | To match footage to a physical location |
| Make and model | `Hikvision DS-2CD2143G2` | Determines resolution and lens options |
| Resolution | `4 MP (2560×1440)` | Sets the smallest animal we can detect |
| Lens focal length | `4 mm` | With resolution, sets detection range |
| Mounting height | `3.2 m` | Steep angles compress the floor and hide animals |
| Tilt angle | `~30° down` | Same |
| Distance to nearest floor in view | `2 m` | Defines the usable detection band |
| Distance to furthest floor in view | `14 m` | Beyond a certain range, animals are too small |
| Infrared at night? | `Yes, built-in` | Almost all pest activity is after dark |
| External IR lamp? | `No` | Determines whether we can raise shutter speed |
| What it looks at | `Loading dock, inside, facing east wall` | Tells us whether it can see pest routes |
| Known pest activity here? | `Droppings found monthly` | Ground truth for validation |

A blank spreadsheet with these columns is at the end of this document.

### One setting worth checking now

If the NVR exposes **shutter speed** or **exposure** in night mode, please note
it. Many cameras drop to 1/8 second at night to keep the picture bright, which
turns any moving animal into a blur.

A rat crossing at normal speed becomes a smear roughly a quarter of its own
length. This is the single most common reason pest detection fails on existing
CCTV, and it is usually fixable with a settings change and costs nothing.

---

## 3. Site information

Per site, ideally:

- **Floor plan or hand sketch.** Rough is fine. We use it to turn "camera 4" into
  "the dock threshold", which is what makes reports actionable.
- **Camera positions marked on it.** Even approximate.
- **Known problem areas** — where staff have seen pests, or where your technicians
  already focus.
- **Existing bait stations and insect light traps**, with locations. If these are
  already on a service schedule, their historical counts are extremely valuable.
- **Operating hours** — when staff are present, when deliveries arrive, when
  cleaning happens. Activity at 02:00 in an empty building means something quite
  different from activity at 18:00 during a shift.
- **Pest sighting log**, if one exists — dates, locations, species. This is the
  only independent ground truth available for checking early detections.

---

## 4. Permissions

Overnight CCTV may contain staff, contractors, and in hospitality or healthcare
sites, members of the public.

We need written confirmation that Express Pesticides has the authority to share
this footage with us for the purpose of developing this system, and that the site
owner is aware. The system blurs faces automatically before any clip leaves a
site, but that protection applies to the finished product — the raw training
footage we receive will not be blurred.

Please also tell us if any site has restrictions we must honour: healthcare
sites in particular may require the footage never leaves their premises, which is
supported but changes how we work.

---

## 5. Sequencing

| Order | Item | Blocks |
|---|---|---|
| 1 | Camera list per site | Choosing which cameras to export |
| 2 | Permissions confirmation | Everything |
| 3 | Overnight footage | Model training — the long pole |
| 4 | Camera information form | Knowing what each camera can physically see |
| 5 | Floor plans, sighting logs, trap records | Reports and validation |

Items 1 and 2 are quick and unblock the rest. If footage takes time to arrange,
please send the camera list first — we can plan around it.

---

## What happens next

1. We review the camera list and agree which cameras to export.
2. Footage arrives; we run it through the system in collection mode. It over-
   detects deliberately — everything that moves is kept for review.
3. Our team labels what the system found. This is where the detector learns what
   a rat looks like at your sites, at night, on your cameras.
4. We report back with what the footage revealed, including an honest assessment
   of which cameras are and are not usable for this purpose. **Some will not be**,
   and it is far better to know that in month one than in month six.

---

## Appendix: camera information form

Copy one row per camera.

```
Site name:
Site type (warehouse / food processing / hotel / hospital / other):
Completed by:                          Date:

| Camera ID | Make & model | Resolution | Lens (mm) | Mount height (m) |
|-----------|--------------|------------|-----------|------------------|
|           |              |            |           |                  |
|           |              |            |           |                  |
|           |              |            |           |                  |

| Camera ID | Tilt | Nearest floor (m) | Furthest floor (m) | IR at night? |
|-----------|------|-------------------|--------------------|--------------|
|           |      |                   |                    |              |
|           |      |                   |                    |              |
|           |      |                   |                    |              |

| Camera ID | External IR lamp? | Night shutter (if known) | What it looks at |
|-----------|-------------------|--------------------------|------------------|
|           |                   |                          |                  |
|           |                   |                          |                  |
|           |                   |                          |                  |

| Camera ID | Known pest activity in this area? | Notes |
|-----------|-----------------------------------|-------|
|           |                                   |       |
|           |                                   |       |
|           |                                   |       |

NVR make and model:
Can it export main-stream footage?           Yes / No / Unsure
Approximate free space for exports:          ______ GB
Preferred handover (USB drive / other):
```
