"""Shared utility functions for OSInsert."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_bbox_txt(path: str | Path) -> Tuple[int, int, int, int]:
    """Load a simple `x1 y1 x2 y2` bbox txt file."""
    path = Path(path)
    with path.open("r") as f:
        line = f.readline().strip()
    x1, y1, x2, y2 = map(int, line.split())
    return x1, y1, x2, y2


def make_rect_mask_from_bbox(h: int, w: int, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Create a uint8 mask (0/255) from bbox coordinates."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("degenerate bbox")

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def read_image_bgr(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img
