# Roboflow model evaluation (experimental)

Evaluating `thermal-rodent-detection-xkfep/1` as a candidate detector.

**Nothing here is production code.** It lives outside `src/expressvision`, and
nothing in `src/` imports it. The existing collector pipeline is untouched.

## Running it

```powershell
$env:ROBOFLOW_API_KEY = "your-key"      # PowerShell
uv pip install inference-sdk
uv run python -m experiments.roboflow_eval.evaluate
```

```bash
export ROBOFLOW_API_KEY=your-key        # bash
```

The key is read from the environment only. It is never written to disk, never
logged, and `.gitignore` covers `.env`, `*.key` and `secrets.*`.

Options:

```
--model-id    default thermal-rodent-detection-xkfep/1
--confidence  default 0.10 — deliberately low, see below
--rebuild     regenerate the test set
--out         default out/roboflow_eval
```

## Why the test set looks like this

The question is not "can it find a rat in a photo" — any rodent model can. It is
whether it works on **greyscale, near-IR-illuminated CCTV at night, with the
animal small, distant, motion-blurred and low-contrast**.

So each subject is rendered under single-variable transformations, and results
are reported by *condition*. A model at 100% on daylight close-ups and 0% on
near-IR is not a 50% model — it is a model for a domain we do not have.

The confidence threshold defaults to **0.10**, much lower than a production
0.5, because weak responses are informative here. A model that fires at 0.12 on
our domain and 0.90 on its own is telling us something a 0.5 cutoff would hide.

### The distinction that decides this

| | Thermal (LWIR, 8–14 µm) | Near-IR CCTV night mode (~850 nm) |
|---|---|---|
| Physics | Passive heat emission | Active IR LED illumination, reflected |
| Sensor | Microbolometer, typically 320×240 | Normal CMOS, IR-cut filter removed, 1080p+ |
| Rodent looks like | Bright hot blob, no texture, no shadow | Mid-grey animal on grey floor, textured, casts a shadow |
| Background | Cool, dark, near-uniform | Cluttered, textured, IR hot-spot in the centre |
| Camera cost | 10–50× a CCTV camera | Standard |

These are **different sensing modalities, not different lighting conditions**.
A detector trained on one has no particular reason to transfer to the other.
Express Pesticides' sites have the right-hand column; thermal cameras are
essentially never installed for general security in warehouses or food plants.

### Test conditions

| Condition | What it isolates |
|---|---|
| `visible` | Baseline — daylight close-up |
| `grayscale` | Colour dependence |
| `nir_night` | **Our actual domain** — greyscale, flat contrast, IR falloff, noise |
| `thermal_colour` / `thermal_whitehot` | Hot-blob-on-cool-background |
| `small_10pct` / `small_3pct` | Distance — 3% is ~10 m on a 2 MP camera |
| `small_nir` | The realistic combination: small *and* near-IR |
| `motion_blur` | Slow night shutter on a moving animal |
| `low_contrast` | Compressed dynamic range |
| `high_angle` | Ceiling-mounted viewpoint |
| `multiple` | Several rodents in one frame |
| `thermal_real` | Genuine LWIR thermogram (no rodent) |
| `*_negative`, `empty_floor` | False-positive controls |
| `fixture_*` | Frames from our own synthetic clip |

**Caveat, stated plainly:** simulated thermal is not real thermal. A real
microbolometer image has no shadows, no reflectance texture, and intensity
encodes temperature. The simulation reproduces the *property* a thermal-trained
detector most likely keys on — a bright compact blob on a dark background — so a
hit there is evidence about that property, not proof the model saw real LWIR.
Real thermograms are included as a separate control.

## Files

| File | Purpose |
|---|---|
| `testset.py` | Builds the controlled ablation on disk |
| `client.py` | Roboflow hosted-API wrapper; key from env only |
| `evaluate.py` | Runs it and reports by modality and condition |
| `adapter.py` | `Detector`-protocol adapter, to plug into our cascade experimentally |

## Why a hosted model cannot be the production detector

Independent of accuracy:

- **It needs the internet.** The architecture requires each site to keep
  detecting, recording and alerting with its uplink down. A cloud detector
  breaks that outright.
- **Client footage leaves the site.** Hospital, food-processing and residential
  deployments have constraints on that which we have committed to honouring.
- **Latency is network-bound.** At ~25k tiles per camera-night, a 200 ms round
  trip is ~1.4 hours of pure waiting per camera per night, scaling linearly with
  cameras.
- **Commercial deployment outside Roboflow's ecosystem requires their Enterprise
  plan.** Edge nodes at client sites are, by definition, outside it.

That does not make the model useless — it makes it unusable *as a production
detector*, which is a different and narrower claim.
