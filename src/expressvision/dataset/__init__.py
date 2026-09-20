"""Training-data acquisition and synthesis.

The client footage this project was planned around is not coming. That removed
the only planned source of rodent imagery, so this package replaces it with one
that depends on nobody:

* :mod:`.lila` indexes and selectively fetches public camera-trap imagery —
  real rodents, real infrared, real night, permissively licensed.
* :mod:`.compose` corrects the one thing that public data gets wrong for us, and
  gets wrong badly: **scale**.

The scale problem is the whole reason this package exists. Camera-trap rodents
in the Channel Islands corpus have a median long axis of 267 px and a 5th
percentile of 108 px, because a camera trap is mounted a metre from a burrow.
Inherited CCTV sees a rodent at 40 px. A detector trained on that corpus learns
the appearance of a large, well-resolved animal and collapses on a small one —
which is precisely the failure measured in ``experiments/roboflow_eval``, where
an off-the-shelf rodent detector held up to ~180 px and then fell off a cliff.

So we resample rather than retrain blind: real rodent crops, rescaled to the
pixel sizes our optics actually deliver, composited into night backgrounds, with
ground-truth boxes that are exact by construction.
"""

from __future__ import annotations

__all__ = ["compose", "lila"]
