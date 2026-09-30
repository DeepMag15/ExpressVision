"""COCO loading for training, with the size metadata the evaluation needs.

The dataset written by ``exv build-dataset`` carries something a normal COCO set
does not: every annotation records ``target_long_axis_px``, the size the
compositor scaled that animal to. That number is what makes size-stratified
evaluation possible without re-deriving it from boxes, and it is exact rather
than inferred.

This module has no torch dependency so it can be unit-tested on any machine.
The torch ``Dataset`` wrapper lives in :mod:`.train`, which does.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Annotation:
    """One labelled box."""

    category: str
    x: float
    y: float
    w: float
    h: float
    target_px: int | None = None

    @property
    def long_axis(self) -> float:
        """Observed box size. Falls back to this when target_px is absent.

        Real (human-labelled) data has no target_px, so evaluation must be able
        to stratify on the box itself. The two agree closely on composited data;
        target_px is preferred only because it is exact.
        """
        return max(self.w, self.h)

    @property
    def size_px(self) -> float:
        return float(self.target_px) if self.target_px else self.long_axis

    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class Frame:
    """One image and everything labelled in it."""

    path: Path
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def is_negative(self) -> bool:
        return not self.annotations


@dataclass
class CocoSet:
    frames: list[Frame]
    classes: list[str]
    citation: str = ""

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    @classmethod
    def load(cls, root: Path) -> CocoSet:
        """Read a dataset directory written by ``exv build-dataset``."""
        root = Path(root)
        payload = json.loads((root / "annotations.json").read_text(encoding="utf-8"))

        categories = {c["id"]: c["name"] for c in payload["categories"]}
        classes = [categories[k] for k in sorted(categories)]

        by_image: dict[int, list[Annotation]] = {}
        for ann in payload.get("annotations", []):
            x, y, w, h = ann["bbox"]
            attrs = ann.get("attributes") or {}
            by_image.setdefault(ann["image_id"], []).append(
                Annotation(
                    category=categories.get(ann["category_id"], "unknown"),
                    x=float(x), y=float(y), w=float(w), h=float(h),
                    target_px=attrs.get("target_long_axis_px"),
                )
            )

        frames = [
            Frame(
                path=root / "images" / img["file_name"],
                width=int(img["width"]),
                height=int(img["height"]),
                annotations=by_image.get(img["id"], []),
            )
            for img in payload["images"]
        ]
        return cls(
            frames=frames,
            classes=classes,
            citation=payload.get("info", {}).get("source_attribution", ""),
        )

    def split(self, val_frac: float = 0.15, seed: int = 13) -> tuple[CocoSet, CocoSet]:
        """Hold out a validation split.

        **This is a train-distribution split, not a test set.** Both halves come
        from the same compositor with the same backgrounds, so a score here
        measures fit, not field performance. The only honest test set is real
        footage that the compositor never touched, and it must never be trained
        on. Nothing in this module can enforce that; a person has to.
        """
        order = list(range(len(self.frames)))
        random.Random(seed).shuffle(order)
        cut = int(len(order) * (1 - val_frac))
        train = [self.frames[i] for i in order[:cut]]
        val = [self.frames[i] for i in order[cut:]]
        return (
            CocoSet(train, self.classes, self.citation),
            CocoSet(val, self.classes, self.citation),
        )

    def size_profile(self) -> dict[str, float]:
        """Where this dataset's objects actually sit, in pixels.

        Worth printing before every run. If the median drifts up toward the
        source corpus's 267 px, the compositor's size weighting has regressed
        and the model will be trained on the wrong regime without anything
        failing loudly.
        """
        sizes = sorted(a.size_px for f in self.frames for a in f.annotations)
        if not sizes:
            return {}
        q = lambda p: sizes[min(len(sizes) - 1, int(len(sizes) * p))]
        return {
            "n": len(sizes),
            "p5": q(0.05), "p25": q(0.25), "median": q(0.50),
            "p75": q(0.75), "p95": q(0.95),
            "frac_at_or_below_40px": sum(1 for s in sizes if s <= 40) / len(sizes),
        }

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for frame in self.frames:
            for ann in frame.annotations:
                counts[ann.category] = counts.get(ann.category, 0) + 1
        counts["(negative frames)"] = sum(1 for f in self.frames if f.is_negative)
        return counts
