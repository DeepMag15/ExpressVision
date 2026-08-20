"""Review and labelling workflow.

The verdict loop is the mechanism by which the system improves after delivery,
so it has to be fast enough that someone actually does it. Typing a CLI command
per event does not scale past about twenty; a night at a busy site produces
hundreds.

This module produces a self-contained HTML sheet: keyframes embedded as data
URIs, keyboard-driven, no server and no network. Verdicts accumulate in the
browser and export as a JSON file that :func:`import_verdicts` applies to the
store. That avoids standing up a web service for a job that is fundamentally
one person looking at pictures for an hour.

It also exports COCO, because the boxes here are *pre-annotations* — motion ROIs
or MegaDetector proposals, not human-verified ground truth. They go into CVAT or
Label Studio for correction, which is far faster than drawing every box cold.
"""

from __future__ import annotations

import base64
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2

from .store import Store

# The L2 taxonomy from the architecture. Species (rat vs mouse) is deliberately
# absent: it is only emitted when the pixels support it, and a reviewer looking
# at a 40 px animal in infrared cannot reliably make that call either.
REVIEW_LABELS = [
    ("rodent", "1"),
    ("bird", "2"),
    ("reptile", "3"),
    ("carnivore", "4"),
    ("insect", "5"),
    ("human", "6"),
    ("other", "7"),
]


@dataclass
class ReviewStats:
    total: int
    written: int
    missing_keyframes: int


def build_review_sheet(
    store: Store,
    out_path: Path,
    limit: int = 200,
    thumb_width: int = 480,
    include_verified: bool = False,
) -> ReviewStats:
    """Write a self-contained HTML review sheet."""
    query = "SELECT * FROM event"
    if not include_verified:
        query += " WHERE verdict IS NULL"
    query += " ORDER BY started_at LIMIT ?"
    rows = list(store.conn.execute(query, (limit,)))

    cards: list[str] = []
    missing = 0
    for row in rows:
        thumb = _thumbnail_data_uri(row["keyframe_path"], thumb_width)
        if thumb is None:
            missing += 1
        cards.append(_render_card(row, thumb))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_page(cards, len(rows)), encoding="utf-8")
    return ReviewStats(total=len(rows), written=len(cards), missing_keyframes=missing)


def _thumbnail_data_uri(path: str | None, width: int) -> str | None:
    """Downscale a keyframe and inline it.

    Embedding keeps the sheet portable — it can be emailed to a technician who
    has no access to the collector's filesystem — and downscaling keeps a
    200-event sheet in the low megabytes rather than the tens.
    """
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None

    img = cv2.imread(str(src))
    if img is None:
        return None

    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _render_card(row: sqlite3.Row, thumb: str | None) -> str:
    when = (row["started_at"] or "")[11:19]
    clip_link = _clip_link(row["clip_path"])
    image = (
        f'<img loading="lazy" src="{thumb}" alt="event keyframe">'
        if thumb
        else '<div class="noimg">keyframe missing</div>'
    )
    return f"""
<article class="card" data-id="{_escape(row['id'])}" data-verdict="">
  <div class="thumb">{image}<span class="badge"></span></div>
  <div class="meta">
    <b>{_escape(row['camera_id'])}</b> · {when}
    <span class="muted">{row['duration_s']:.1f}s · {row['n_detections']} det ·
    straight {row['straightness']:.2f} · {_escape(row['label'] or '')}</span>
    {clip_link}
  </div>
</article>"""


def _clip_link(clip_path: str | None) -> str:
    """Link to the clip on disk.

    Paths are stored as written, which is usually relative to the working
    directory, and a relative path cannot be expressed as a file:// URI —
    resolve first.
    """
    if not clip_path:
        return '<span class="clip muted">no clip</span>'
    try:
        resolved = Path(clip_path).resolve(strict=True)
    except (OSError, ValueError):
        return '<span class="clip muted">clip missing</span>'
    return f'<a class="clip" href="{_escape(resolved.as_uri())}">open clip</a>'


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_page(cards: list[str], total: int) -> str:
    keys = "".join(
        f'<kbd>{key}</kbd> {name} ' for name, key in REVIEW_LABELS
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ExpressVision review</title>
<style>
:root {{
  --ground:#eef1ee; --surface:#fff; --ink:#101715; --muted:#6b7772;
  --line:#ccd4ca; --ok:#2b6656; --no:#a2382a; --accent:#a9640f;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --ground:#0c1211; --surface:#151d1b; --ink:#e1e7e2; --muted:#7d8a84;
    --line:#26302d; --ok:#5fa994; --no:#db7461; --accent:#e3a34a;
  }}
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--ground); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:10; background:var(--surface);
  border-bottom:1px solid var(--line); padding:12px 20px;
  display:flex; gap:20px; align-items:center; flex-wrap:wrap; }}
h1 {{ font-size:16px; margin:0; letter-spacing:-.01em }}
.help {{ font-size:12.5px; color:var(--muted) }}
kbd {{ background:var(--ground); border:1px solid var(--line); border-radius:3px;
  padding:1px 5px; font:11px ui-monospace,monospace; }}
.counts {{ margin-left:auto; font:12px ui-monospace,monospace; color:var(--muted) }}
button {{ font:inherit; padding:6px 12px; border-radius:4px; cursor:pointer;
  border:1px solid var(--line); background:var(--surface); color:var(--ink); }}
button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff }}
main {{ display:grid; gap:14px; padding:20px;
  grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:6px;
  overflow:hidden; scroll-margin:90px; }}
.card.sel {{ outline:2px solid var(--accent); outline-offset:1px }}
.card[data-verdict="confirmed"] {{ border-color:var(--ok) }}
.card[data-verdict="rejected"] {{ border-color:var(--no); opacity:.55 }}
.thumb {{ position:relative; background:#000; aspect-ratio:16/9 }}
.thumb img {{ width:100%; height:100%; object-fit:contain; display:block }}
.noimg {{ display:grid; place-items:center; height:100%; color:var(--muted); font-size:13px }}
.badge {{ position:absolute; top:8px; left:8px; padding:2px 8px; border-radius:3px;
  font:11px ui-monospace,monospace; color:#fff; display:none }}
.card[data-verdict] .badge:not(:empty) {{ display:block }}
.card[data-verdict="confirmed"] .badge {{ background:var(--ok) }}
.card[data-verdict="rejected"] .badge {{ background:var(--no) }}
.card[data-verdict="reclassified"] .badge {{ background:var(--accent) }}
.meta {{ padding:10px 12px; font-size:13px; display:flex; flex-direction:column; gap:3px }}
.muted {{ color:var(--muted); font-size:12px }}
.clip {{ font-size:12px; color:var(--accent) }}
</style></head><body>
<header>
  <h1>ExpressVision review</h1>
  <div class="help">
    <kbd>J</kbd>/<kbd>K</kbd> move <kbd>C</kbd> confirm <kbd>R</kbd> reject
    <kbd>U</kbd> undo &nbsp;|&nbsp; reclassify: {keys}
  </div>
  <div class="counts"><span id="done">0</span>/{total} · <span id="conf">0</span> confirmed</div>
  <button class="primary" id="export">Export verdicts</button>
</header>
<main id="grid">
{"".join(cards)}
</main>
<script>
const LABELS = {json.dumps([name for name, _ in REVIEW_LABELS])};
const KEYS = {json.dumps({key: name for name, key in REVIEW_LABELS})};
const cards = [...document.querySelectorAll('.card')];
const KEY = 'expressvision-verdicts';
let store = JSON.parse(localStorage.getItem(KEY) || '{{}}');
let i = 0;

function paint(card) {{
  const v = store[card.dataset.id];
  card.dataset.verdict = v ? v.verdict : '';
  card.querySelector('.badge').textContent =
    v ? (v.corrected_label ? v.verdict + ': ' + v.corrected_label : v.verdict) : '';
}}
function counts() {{
  const vals = Object.values(store);
  document.getElementById('done').textContent = vals.length;
  document.getElementById('conf').textContent =
    vals.filter(v => v.verdict === 'confirmed').length;
}}
function select(n) {{
  if (!cards.length) return;
  cards[i]?.classList.remove('sel');
  i = Math.max(0, Math.min(cards.length - 1, n));
  cards[i].classList.add('sel');
  cards[i].scrollIntoView({{block:'nearest', behavior:'smooth'}});
}}
function setVerdict(verdict, label) {{
  const card = cards[i];
  if (!card) return;
  store[card.dataset.id] = {{verdict, corrected_label: label || null}};
  localStorage.setItem(KEY, JSON.stringify(store));
  paint(card); counts();
  if (i < cards.length - 1) select(i + 1);
}}
document.addEventListener('keydown', e => {{
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'j' || k === 'arrowdown') {{ select(i + 1); e.preventDefault(); }}
  else if (k === 'k' || k === 'arrowup') {{ select(i - 1); e.preventDefault(); }}
  else if (k === 'c') setVerdict('confirmed');
  else if (k === 'r') setVerdict('rejected');
  else if (k === 'u') {{
    delete store[cards[i].dataset.id];
    localStorage.setItem(KEY, JSON.stringify(store));
    paint(cards[i]); counts();
  }}
  else if (KEYS[k]) setVerdict('reclassified', KEYS[k]);
}});
cards.forEach((c, n) => c.addEventListener('click', () => select(n)));
document.getElementById('export').addEventListener('click', () => {{
  const rows = Object.entries(store).map(([id, v]) => ({{event_id: id, ...v}}));
  if (!rows.length) {{ alert('Nothing to export yet.'); return; }}
  const blob = new Blob([JSON.stringify({{verdicts: rows}}, null, 2)],
                        {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'verdicts.json';
  a.click();
}});
cards.forEach(paint); counts(); select(0);
</script></body></html>"""


def import_verdicts(store: Store, path: Path, user: str = "review-sheet") -> tuple[int, int]:
    """Apply an exported verdicts file. Returns (applied, skipped)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("verdicts", payload if isinstance(payload, list) else [])

    applied = skipped = 0
    for row in rows:
        event_id = row.get("event_id")
        verdict = row.get("verdict")
        if not event_id or verdict not in {"confirmed", "rejected", "reclassified"}:
            skipped += 1
            continue
        if store.set_verdict(event_id, verdict, row.get("corrected_label"), user):
            applied += 1
        else:
            skipped += 1
    return applied, skipped


def export_coco(
    store: Store,
    out_dir: Path,
    only_confirmed: bool = True,
    copy_images: bool = True,
) -> dict[str, int]:
    """Export keyframes and boxes as COCO, for import into CVAT or Label Studio.

    These are **pre-annotations**, not ground truth. The boxes come from motion
    ROIs or MegaDetector proposals; a human corrects them. Starting from a
    roughly-right box is several times faster than drawing each one cold, which
    is the whole reason to export rather than hand over raw frames.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    query = "SELECT * FROM event WHERE keyframe_path IS NOT NULL"
    if only_confirmed:
        query += " AND verdict IN ('confirmed','reclassified')"
    query += " ORDER BY started_at"
    rows = list(store.conn.execute(query))

    categories = [
        {"id": idx + 1, "name": name, "supercategory": "pest"}
        for idx, (name, _) in enumerate(REVIEW_LABELS)
    ]
    by_name = {c["name"]: c["id"] for c in categories}

    images: list[dict] = []
    annotations: list[dict] = []
    skipped = 0

    for image_id, row in enumerate(rows, start=1):
        src = Path(row["keyframe_path"])
        sidecar = src.with_suffix(".json")
        if not src.exists() or not sidecar.exists():
            skipped += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            skipped += 1
            continue
        height, width = img.shape[:2]

        file_name = f"{row['id']}.jpg"
        if copy_images:
            shutil.copy2(src, images_dir / file_name)

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "date_captured": row["started_at"],
                "expressvision": {
                    "event_id": row["id"],
                    "camera_id": row["camera_id"],
                    "site_id": row["site_id"],
                    "model_version": row["model_version"],
                },
            }
        )

        geometry = json.loads(sidecar.read_text(encoding="utf-8"))
        x1, y1, x2, y2 = geometry.get("box", [0, 0, 0, 0])
        label = row["corrected_label"] or "other"
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": by_name.get(label, by_name["other"]),
                "bbox": [x1, y1, max(0, x2 - x1), max(0, y2 - y1)],
                "area": max(0, x2 - x1) * max(0, y2 - y1),
                "iscrowd": 0,
                "attributes": {"pre_annotation": True, "source": row["model_version"]},
            }
        )

    coco = {
        "info": {
            "description": "ExpressVision pre-annotations — corrections required",
            "version": "1.0",
            "date_created": datetime.now(UTC).isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    (out_dir / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")

    return {"images": len(images), "annotations": len(annotations), "skipped": skipped}
