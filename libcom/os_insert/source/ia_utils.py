"""Local reimplementation of InsertAnything utility functions.

Moved from `osinsert/ia_utils.py` so that all OSInsert logic lives under
`libcom/os_insert` and does not depend on external repositories.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import math


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get (y1, y2, x1, x2) bounding box from a binary mask.

    This mirrors the original InsertAnything implementation: when the mask is
    nearly empty, fall back to the full image.
    """

    h, w = mask.shape[:2]

    if mask.sum() < 10:
        return 0, h, 0, w

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(y1), int(y2), int(x1), int(x2)


def expand_bbox(mask: np.ndarray, yyxx, ratio: float, min_crop: int = 0):
    """Expand a bbox using the same heuristic as the original repo.

    `mask` is only used for its shape; the expansion ratio is adapted based on
    the relative area of the bbox.
    """

    y1, y2, x1, x2 = yyxx
    H, W = mask.shape[0], mask.shape[1]

    yyxx_area = (y2 - y1 + 1) * (x2 - x1 + 1)
    r1 = yyxx_area / (H * W)

    def _f(r, T=0.6, beta=0.1):
        return np.where(r < T, beta + (1 - beta) / T * r, 1)

    r2 = _f(r1)
    ratio = math.sqrt(r2 / r1)

    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2 - y1 + 1)
    w = ratio * (x2 - x1 + 1)
    h = max(h, min_crop)
    w = max(w, min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)
    return int(y1), int(y2), int(x1), int(x2)


def pad_to_square(image: np.ndarray, pad_value: int = 255, random: bool = False) -> np.ndarray:
    """Pad to square using the original InsertAnything convention.

    Only the shorter side is padded, and padding is applied with `np.pad`.
    """

    H, W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0, padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if image.ndim == 2:
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2))
        else:
            pad_param = ((padd_1, padd_2), (0, 0))
    else:
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
        else:
            pad_param = ((padd_1, padd_2), (0, 0), (0, 0))

    image = np.pad(image, pad_param, "constant", constant_values=pad_value)
    return image


def box2squre(image: np.ndarray, box) -> Tuple[int, int, int, int]:  # spelling kept for compatibility
    """Convert a bbox to square, matching the original InsertAnything logic."""

    H, W = image.shape[0], image.shape[1]
    y1, y2, x1, x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h = y2 - y1
    w = x2 - x1

    if h >= w:
        x1 = cx - h // 2
        x2 = cx + h // 2
    else:
        y1 = cy - w // 2
        y2 = cy + w // 2
    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)
    return int(y1), int(y2), int(x1), int(x2)


def crop_back(edited: np.ndarray, old_tar: np.ndarray, hw_vec, box_crop) -> np.ndarray:
    """Paste the edited crop back into the original target image.

    This is a faithful port of the original InsertAnything geometry, including
    margin and de-padding logic. It is used by both conservative and
    aggressive modes so they share identical edge behavior.
    """

    H1, W1, H2, W2 = map(int, hw_vec)
    y1, y2, x1, x2 = map(int, box_crop)

    # Resize prediction back to the padded crop size.
    pred = cv2.resize(edited, (W2, H2))
    m = 2  # margin pixels

    # Case 1: original crop was square, simple margin paste.
    if W1 == H1:
        if m != 0:
            old_tar[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
        else:
            old_tar[y1:y2, x1:x2, :] = pred[:, :]
        return old_tar

    # Case 2: handle padding introduced by pad_to_square along one dimension.
    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]

    gen_image = old_tar.copy()
    if m != 0:
        gen_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
    else:
        gen_image[y1:y2, x1:x2, :] = pred[:, :]

    return gen_image


def expand_image_mask(image: np.ndarray, mask: np.ndarray, ratio: float = 1.4):
    """Expand image and mask by symmetric padding (original behavior)."""

    h, w = image.shape[0], image.shape[1]
    H, W = int(h * ratio), int(w * ratio)
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W - w) // 2)
    w2 = W - w - w1

    pad_param_image = ((h1, h2), (w1, w2), (0, 0))
    pad_param_mask = ((h1, h2), (w1, w2))
    image = np.pad(image, pad_param_image, "constant", constant_values=255)
    mask = np.pad(mask, pad_param_mask, "constant", constant_values=0)
    return image, mask
