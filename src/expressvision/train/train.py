"""Fine-tune an Apache-2.0 detector on the scale-corrected dataset.

RT-DETR is the default, for the reason recorded in the architecture: the
Ultralytics YOLO family is AGPL-3.0, and shipping it inside software sold to
clients triggers network copyleft. RT-DETR and the ``transformers`` library that
serves it are both Apache-2.0, so nothing here constrains what we can deploy.

Imports of torch and transformers are deferred into the functions that need
them. The collector, the console and the test suite must keep working on a
machine with neither installed, and a module-level import would break that.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import CocoSet, Frame
from .evaluate import GroundTruth, Prediction, evaluate, report

# Pretrained COCO checkpoints, all Apache-2.0.
CHECKPOINTS = {
    "rtdetr-r18": "PekingU/rtdetr_r18vd_coco_o365",
    "rtdetr-r50": "PekingU/rtdetr_r50vd_coco_o365",
    "rtdetr-v2-r18": "PekingU/rtdetr_v2_r18vd",
    "rtdetr-v2-r50": "PekingU/rtdetr_v2_r50vd",
}
DEFAULT_CHECKPOINT = "rtdetr-v2-r18"


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    # Detector input size. 640 matches the tile size stage C cuts, so the model
    # sees objects at the same scale in training and in deployment.
    image_size: int = 640
    num_workers: int = 4
    seed: int = 17
    checkpoint: str = DEFAULT_CHECKPOINT
    amp: bool = True
    val_frac: float = 0.15


def _require(module: str, extra: str = "train"):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"{module!r} is needed for training. Install the optional extra:\n"
            f'    uv pip install -e ".[{extra}]"'
        ) from exc


def build_torch_dataset(coco: CocoSet, processor, image_size: int):
    """Wrap a CocoSet as a torch Dataset. Imported lazily by design."""
    torch = _require("torch")
    import cv2
    import numpy as np

    class _Set(torch.utils.data.Dataset):
        def __init__(self, frames: list[Frame], classes: list[str]) -> None:
            self.frames = frames
            self.class_to_id = {name: i for i, name in enumerate(classes)}

        def __len__(self) -> int:
            return len(self.frames)

        def __getitem__(self, idx: int):
            frame = self.frames[idx]
            image = cv2.imread(str(frame.path))
            if image is None:
                image = np.zeros((image_size, image_size, 3), np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            annotations = [
                {
                    "bbox": [a.x, a.y, a.w, a.h],
                    "category_id": self.class_to_id.get(a.category, 0),
                    "area": a.w * a.h,
                    "iscrowd": 0,
                }
                for a in frame.annotations
            ]
            encoding = processor(
                images=image,
                annotations={"image_id": idx, "annotations": annotations},
                return_tensors="pt",
            )
            return {
                "pixel_values": encoding["pixel_values"][0],
                "labels": encoding["labels"][0],
            }

    return _Set(coco.frames, coco.classes)


def collate(batch):
    torch = _require("torch")
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


def train(
    dataset_dir: Path,
    out_dir: Path,
    cfg: TrainConfig | None = None,
    on_epoch=None,
) -> dict:
    """Fine-tune and return a summary. Writes checkpoints into ``out_dir``."""
    cfg = cfg or TrainConfig()
    torch = _require("torch")
    transformers = _require("transformers")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = CocoSet.load(Path(dataset_dir))
    train_set, val_set = coco.split(val_frac=cfg.val_frac, seed=cfg.seed)

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo = CHECKPOINTS.get(cfg.checkpoint, cfg.checkpoint)
    processor = transformers.AutoImageProcessor.from_pretrained(
        repo, size={"width": cfg.image_size, "height": cfg.image_size}
    )
    model = transformers.AutoModelForObjectDetection.from_pretrained(
        repo,
        num_labels=len(coco.classes),
        ignore_mismatched_sizes=True,
    ).to(device)

    loader = torch.utils.data.DataLoader(
        build_torch_dataset(train_set, processor, cfg.image_size),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps = max(1, len(loader) * cfg.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=cfg.lr, total_steps=steps, pct_start=cfg.warmup_frac
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device == "cuda"):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            running += float(loss.detach()) * pixel_values.size(0)
            seen += pixel_values.size(0)

        entry = {"epoch": epoch, "loss": running / max(1, seen)}
        history.append(entry)
        if on_epoch:
            on_epoch(entry)

        model.save_pretrained(out_dir / "checkpoint")
        processor.save_pretrained(out_dir / "checkpoint")

    result = predict_and_evaluate(model, processor, val_set, device)
    summary = {
        "config": asdict(cfg),
        "classes": coco.classes,
        "train_frames": len(train_set),
        "val_frames": len(val_set),
        "size_profile": coco.size_profile(),
        "history": history,
        "pixel_floor": result.pixel_floor(),
        "recall": result.recall,
        "precision": result.precision,
        "fp_per_frame": result.fp_per_frame,
        "citation": coco.citation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(report(result), encoding="utf-8")
    return summary


def predict_and_evaluate(model, processor, coco: CocoSet, device: str,
                         score_threshold: float = 0.35):
    """Run the model over a split and stratify recall by target size."""
    torch = _require("torch")
    import cv2

    model.eval()
    frames: list[tuple[list[GroundTruth], list[Prediction]]] = []

    with torch.no_grad():
        for frame in coco.frames:
            image = cv2.imread(str(frame.path))
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            outputs = model(**inputs)

            target_sizes = torch.tensor([[frame.height, frame.width]]).to(device)
            detections = processor.post_process_object_detection(
                outputs, threshold=score_threshold, target_sizes=target_sizes
            )[0]

            preds = [
                Prediction(
                    box=tuple(float(v) for v in box),
                    label=coco.classes[int(label)] if int(label) < len(coco.classes) else "unknown",
                    score=float(score),
                )
                for score, label, box in zip(
                    detections["scores"], detections["labels"], detections["boxes"]
                )
            ]
            truths = [
                GroundTruth(box=a.xyxy(), label=a.category, size_px=a.size_px)
                for a in frame.annotations
            ]
            frames.append((truths, preds))

    return evaluate(frames, score_threshold=score_threshold)
