"""Native-resolution tiling — stage C of the cascade.

The standard pipeline resizes a whole frame to the detector's input size. That
is fine for people and vehicles and fatal for pests: a 1920px frame squeezed
into 640px costs three linear factors, turning a 40px rat into 13px — below the
detection floor.

Because the gate has already localised the interesting regions, we can afford to
cut crops at 1:1 from the full-resolution frame instead. Same GPU work, three
times the pixels on target.

Regions larger than the tile (a person, a forklift) are downscaled to fit, which
is safe: large objects survive downscaling, which is precisely why small ones
get the native-pixel budget.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import TileConfig
from .types import Box, Tile


class Tiler:
    def __init__(self, cfg: TileConfig, frame_width: int, frame_height: int) -> None:
        self.cfg = cfg
        self.frame_width = frame_width
        self.frame_height = frame_height

    def plan(self, rois: list[Box]) -> list[Tile]:
        """Choose the smallest set of tiles that covers every ROI.

        ROIs are taken largest-first so a big region claims its tile before
        smaller neighbours, then every remaining ROI that already falls inside
        that tile is absorbed into it rather than triggering another inference.
        """
        if not rois:
            return []

        cfg = self.cfg
        order = sorted(range(len(rois)), key=lambda i: rois[i].area, reverse=True)
        assigned: set[int] = set()
        tiles: list[Tile] = []

        for i in order:
            if i in assigned:
                continue
            if len(tiles) >= cfg.max_tiles_per_frame:
                break

            padded = self._pad(rois[i])
            size = self._tile_extent(padded)
            x, y = self._place(padded, size)
            tile = Tile(x=x, y=y, size=size, roi_indices=[i])
            assigned.add(i)

            # Absorb any other pending ROI fully inside this tile.
            for j in order:
                if j in assigned:
                    continue
                r = rois[j]
                if x <= r.x1 and y <= r.y1 and r.x2 <= x + size and r.y2 <= y + size:
                    tile.roi_indices.append(j)
                    assigned.add(j)

            tiles.append(tile)

        return tiles

    def _pad(self, roi: Box) -> Box:
        pad_x = roi.w * self.cfg.roi_padding * 0.5
        pad_y = roi.h * self.cfg.roi_padding * 0.5
        return Box(roi.x1 - pad_x, roi.y1 - pad_y, roi.x2 + pad_x, roi.y2 + pad_y)

    def _tile_extent(self, roi: Box) -> int:
        """Source-space square that the tile covers.

        ``tile_size`` when the ROI fits — that is the native-resolution case and
        the one that matters. Larger only when the ROI cannot fit, and never
        larger than the frame's shorter side.
        """
        need = int(np.ceil(max(roi.w, roi.h)))
        limit = max(1, min(self.frame_width, self.frame_height))
        return int(min(limit, max(self.cfg.tile_size, need)))

    def _place(self, roi: Box, size: int) -> tuple[int, int]:
        """Centre the tile on the ROI, then slide it inside the frame bounds."""
        x = round(roi.cx - size / 2.0)
        y = round(roi.cy - size / 2.0)
        x = max(0, min(x, self.frame_width - size))
        y = max(0, min(y, self.frame_height - size))
        return x, y

    def crop(self, frame: np.ndarray, tile: Tile) -> np.ndarray:
        """Extract a tile, resized to the detector input only if oversized."""
        patch = frame[tile.y : tile.y + tile.size, tile.x : tile.x + tile.size]
        if tile.size != self.cfg.tile_size:
            patch = cv2.resize(
                patch,
                (self.cfg.tile_size, self.cfg.tile_size),
                interpolation=cv2.INTER_AREA,
            )
        return patch

    def to_frame_coords(self, tile: Tile, box: Box) -> Box:
        """Map a box from detector-input space back to full-frame coordinates."""
        return map_to_frame(
            tile, box, self.cfg.tile_size, self.frame_width, self.frame_height
        )


def map_to_frame(
    tile: Tile, box: Box, tile_size: int, frame_width: int, frame_height: int
) -> Box:
    """Detector-input coordinates to full-frame coordinates.

    Module level because the detector needs it too, and a detector that maps
    boxes back with slightly different arithmetic than the tiler used to cut
    them produces detections that are subtly, silently in the wrong place.
    """
    factor = tile.size / tile_size
    return Box(
        tile.x + box.x1 * factor,
        tile.y + box.y1 * factor,
        tile.x + box.x2 * factor,
        tile.y + box.y2 * factor,
    ).clipped(frame_width, frame_height)


def nms(detections: list, iou_threshold: float = 0.55) -> list:
    """Greedy non-maximum suppression in full-frame coordinates.

    Adjacent tiles overlap, so one animal near a shared boundary can be
    reported twice. Suppression has to happen after mapping back to the frame —
    doing it per tile cannot see the duplicate.
    """
    if len(detections) < 2:
        return list(detections)

    keep: list = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(det.box.iou(kept.box) < iou_threshold for kept in keep):
            keep.append(det)
    return keep
