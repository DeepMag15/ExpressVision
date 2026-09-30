"""Size-stratified detection evaluation.

**Why not mAP.** COCO mAP averages precision over all object sizes, which is
exactly the wrong summary for this project. A detector can post a respectable
mAP while failing completely below 60 px, because the large objects carry the
average. That is precisely the failure measured in
``experiments/roboflow_eval``: confidence held to roughly 180 px and then
collapsed, and a single averaged number would have concealed it.

So this module reports recall **as a function of target size**, in pixel bands,
and names the smallest band that clears a recall threshold. That number — the
detector's own pixel floor — is directly comparable to the 40 px design target
and to the 130-180 px floor measured on the third-party model.

The matching rule is deliberately plain: greedy IoU at a fixed threshold, no
score-sorted precision-recall integration. Recall at a fixed operating point is
what a deployment actually experiences, and it is a number an operator can be
told without a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DETECT_FLOOR_PX = 40

# Bands straddle the design floor deliberately: 32-48 px is the decision
# boundary, so it gets its own band rather than being averaged into neighbours.
DEFAULT_BANDS: tuple[tuple[int, int], ...] = (
    (0, 32),
    (32, 48),
    (48, 64),
    (64, 96),
    (96, 144),
    (144, 10_000),
)

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Prediction:
    box: Box
    label: str
    score: float


@dataclass
class GroundTruth:
    box: Box
    label: str
    size_px: float


@dataclass
class BandResult:
    lo: int
    hi: int
    total: int = 0
    hit: int = 0

    @property
    def recall(self) -> float | None:
        """None, not zero, when the band has no examples.

        A band with no ground truth has an undefined recall. Reporting 0.0 would
        read as total failure and would drag any average down — the same
        unknown-versus-zero distinction the console enforces for camera uptime.
        """
        return (self.hit / self.total) if self.total else None

    @property
    def label(self) -> str:
        return f"{self.lo}-{self.hi} px" if self.hi < 10_000 else f"{self.lo}+ px"


@dataclass
class EvalResult:
    bands: list[BandResult]
    matched: int = 0
    missed: int = 0
    false_positives: int = 0
    frames: int = 0
    negative_frames: int = 0
    fp_on_negatives: int = 0
    per_class: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def recall(self) -> float | None:
        total = self.matched + self.missed
        return (self.matched / total) if total else None

    @property
    def precision(self) -> float | None:
        total = self.matched + self.false_positives
        return (self.matched / total) if total else None

    @property
    def fp_per_frame(self) -> float | None:
        return (self.false_positives / self.frames) if self.frames else None

    def pixel_floor(self, min_recall: float = 0.85) -> int | None:
        """Smallest band whose recall clears the threshold, and stays clear.

        A single band scraping past the threshold by luck is not a floor. The
        floor is the point above which the detector is reliable, so every band
        from here up must also clear it.
        """
        ordered = sorted(self.bands, key=lambda b: b.lo)
        for i, band in enumerate(ordered):
            r = band.recall
            if r is None or r < min_recall:
                continue
            if all(
                (later.recall is None or later.recall >= min_recall)
                for later in ordered[i + 1:]
            ):
                return band.lo
        return None

    def verdict(self, min_recall: float = 0.85) -> str:
        floor = self.pixel_floor(min_recall)
        if floor is None:
            return (
                f"No pixel size reaches {min_recall:.0%} recall. "
                f"The detector is not usable at any range."
            )
        if floor <= DETECT_FLOOR_PX:
            return (
                f"Pixel floor {floor} px — meets the {DETECT_FLOOR_PX} px design "
                f"target. Range predictions in the architecture hold."
            )
        ratio = (floor / DETECT_FLOOR_PX) ** 2
        return (
            f"Pixel floor {floor} px against a {DETECT_FLOOR_PX} px target. "
            f"Equivalent coverage would need ~{ratio:.1f}x more cameras."
        )


def evaluate(
    frames: list[tuple[list[GroundTruth], list[Prediction]]],
    iou_threshold: float = 0.4,
    score_threshold: float = 0.35,
    bands: tuple[tuple[int, int], ...] = DEFAULT_BANDS,
    class_agnostic: bool = False,
) -> EvalResult:
    """Match predictions to ground truth and stratify recall by target size.

    ``iou_threshold`` is 0.4 rather than the COCO-conventional 0.5 on purpose.
    At 40 px a box is roughly 40x20, so a four-pixel offset costs a large slice
    of IoU while still being a perfectly useful detection — the operator sees
    the animal either way. Demanding 0.5 at this scale measures box regression
    precision, not whether the pest was found.
    """
    result = EvalResult(bands=[BandResult(lo, hi) for lo, hi in bands])

    for truths, preds in frames:
        result.frames += 1
        kept = sorted(
            (p for p in preds if p.score >= score_threshold),
            key=lambda p: -p.score,
        )

        if not truths:
            result.negative_frames += 1
            result.fp_on_negatives += len(kept)

        used: set[int] = set()
        for truth in truths:
            band = next(
                (b for b in result.bands if b.lo <= truth.size_px < b.hi),
                None,
            )
            if band is not None:
                band.total += 1

            best_i, best_iou = -1, 0.0
            for i, pred in enumerate(kept):
                if i in used:
                    continue
                if not class_agnostic and pred.label != truth.label:
                    continue
                overlap = iou(pred.box, truth.box)
                if overlap > best_iou:
                    best_i, best_iou = i, overlap

            seen, missed = result.per_class.get(truth.label, (0, 0))
            if best_i >= 0 and best_iou >= iou_threshold:
                used.add(best_i)
                result.matched += 1
                result.per_class[truth.label] = (seen + 1, missed)
                if band is not None:
                    band.hit += 1
            else:
                result.missed += 1
                result.per_class[truth.label] = (seen, missed + 1)

        result.false_positives += len(kept) - len(used)

    return result


def report(result: EvalResult, min_recall: float = 0.85) -> str:
    """A text report. The size table is the part that matters."""
    lines: list[str] = []
    lines.append("Recall by target size")
    lines.append(f"  {'band':>12}  {'n':>7}  {'recall':>7}")
    for band in sorted(result.bands, key=lambda b: b.lo):
        r = band.recall
        shown = "  --  " if r is None else f"{r:6.1%}"
        mark = "  <- design floor" if band.lo <= DETECT_FLOOR_PX < band.hi else ""
        lines.append(f"  {band.label:>12}  {band.total:>7,}  {shown}{mark}")

    lines.append("")
    overall = "n/a" if result.recall is None else f"{result.recall:.1%}"
    prec = "n/a" if result.precision is None else f"{result.precision:.1%}"
    fpf = "n/a" if result.fp_per_frame is None else f"{result.fp_per_frame:.2f}"
    lines.append(f"  overall recall     {overall}")
    lines.append(f"  precision          {prec}")
    lines.append(f"  false pos / frame  {fpf}")
    if result.negative_frames:
        lines.append(
            f"  FPs on {result.negative_frames:,} empty frames: "
            f"{result.fp_on_negatives:,}"
        )

    if result.per_class:
        lines.append("")
        lines.append("Per class")
        for name, (hit, miss) in sorted(result.per_class.items()):
            total = hit + miss
            rate = f"{hit / total:.1%}" if total else "n/a"
            lines.append(f"  {name:<12} {hit:>6,}/{total:<6,} {rate:>7}")

    lines.append("")
    lines.append(result.verdict(min_recall))
    return "\n".join(lines)
