"""SAM wrapper for ObjectStitch composites (skeleton).

This module provides a libcom-style home for running SAM on top of
ObjectStitch outputs, using a bbox prompt to focus on the insertion region.

The actual SAM model loading (segment-anything) will be implemented in a
later phase. For now only the public API is defined so that `OSInsertModel`
can depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SAM_PRETRAINED_ROOT = REPO_ROOT / "pretrained_models" / "sam"


@dataclass
class SamOnObjectStitchConfig:
    """Configuration for SAM-on-ObjectStitch inference."""

    sam_checkpoint: Path | str = SAM_PRETRAINED_ROOT / "sam_vit_h_4b8939.pth"
    model_type: str = "vit_h"
    device: str = "cuda:0"


def _run_sam_on_image(predictor, image_bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    """Run SAM with a box prompt and return a binary mask (uint8 0/255).

    Ported from `run_sam_on_objectstitch.py::run_sam_on_image`.
    """

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM returned no masks")

    best_idx = int(np.argmax(scores))
    m = masks[best_idx].astype(np.uint8)

    x1, y1, x2, y2 = box_xyxy
    box_w = max(int(x2 - x1), 1)
    box_h = max(int(y2 - y1), 1)
    box_area = float(box_w * box_h)
    area = float(int(m.sum()))
    if area / box_area < 0.01:
        raise RuntimeError("SAM mask area too small")

    m = m.astype(np.uint8) * 255
    return m


def run_sam_on_objectstitch_single(
    os_image_path: Path,
    bg_shape_hw: Tuple[int, int],
    bbox_xyxy_bg: Tuple[int, int, int, int],
    *,
    config: SamOnObjectStitchConfig,
    out_path: Path | None = None,
) -> Path:
    """Run SAM on a single ObjectStitch composite and return a binary mask path.

    Parameters
    ----------
    os_image_path:
        Path to the ObjectStitch composite image (BGR when read by OpenCV).
    bg_shape_hw:
        Original background image shape as (H, W). The bbox is defined in this
        coordinate system.
    bbox_xyxy_bg:
        Bbox (x1, y1, x2, y2) in the original background coordinates.
    config:
        SAM configuration including checkpoint and device.
    out_path:
        Optional output path for the SAM mask PNG. If None, will create a
        sibling file next to the ObjectStitch input with suffix `_sam_mask.png`.
    """

    os_image_path = Path(os_image_path)
    if out_path is None:
        out_path = os_image_path.with_name(os_image_path.stem + "_sam_mask.png")
    out_path = Path(out_path)

    if not os_image_path.exists():
        raise FileNotFoundError(os_image_path)
    if not config.sam_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {config.sam_checkpoint}. "
            f"Expected at {SAM_PRETRAINED_ROOT}."
        )

    os_image_bgr = cv2.imread(str(os_image_path))
    if os_image_bgr is None:
        raise RuntimeError(f"Failed to read ObjectStitch image: {os_image_path}")

    os_h, os_w = os_image_bgr.shape[:2]
    bg_h, bg_w = bg_shape_hw

    bx1, by1, bx2, by2 = bbox_xyxy_bg

    scale_x = os_w / float(bg_w) if bg_w > 0 else 1.0
    scale_y = os_h / float(bg_h) if bg_h > 0 else 1.0

    x1 = int(bx1 * scale_x)
    y1 = int(by1 * scale_y)
    x2 = int(bx2 * scale_x)
    y2 = int(by2 * scale_y)

    x1 = max(0, min(x1, os_w - 1))
    y1 = max(0, min(y1, os_h - 1))
    x2 = max(x1 + 1, min(x2, os_w))
    y2 = max(y1 + 1, min(y2, os_h))

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as e:
        raise ImportError(
            "segment_anything package is required for SAM aggressive mode. "
            "Install from https://github.com/facebookresearch/segment-anything"
        ) from e

    sam = sam_model_registry[config.model_type](checkpoint=str(config.sam_checkpoint))
    sam.to(device=config.device)
    predictor = SamPredictor(sam)

    os_mask = _run_sam_on_image(predictor, os_image_bgr, (x1, y1, x2, y2))
    cv2.imwrite(str(out_path), os_mask)

    return out_path


def run_sam_on_objectstitch(
    os_image: np.ndarray,
    bg_shape_hw: Tuple[int, int],
    bbox_xyxy_bg: Tuple[int, int, int, int],
    *,
    config: SamOnObjectStitchConfig,
) -> np.ndarray:
    os_image_bgr = os_image
    if os_image_bgr.ndim != 3 or os_image_bgr.shape[2] != 3:
        raise ValueError("Expected HxWx3 image")

    os_h, os_w = os_image_bgr.shape[:2]
    bg_h, bg_w = bg_shape_hw

    bx1, by1, bx2, by2 = bbox_xyxy_bg

    scale_x = os_w / float(bg_w) if bg_w > 0 else 1.0
    scale_y = os_h / float(bg_h) if bg_h > 0 else 1.0

    x1 = int(bx1 * scale_x)
    y1 = int(by1 * scale_y)
    x2 = int(bx2 * scale_x)
    y2 = int(by2 * scale_y)

    x1 = max(0, min(x1, os_w - 1))
    y1 = max(0, min(y1, os_h - 1))
    x2 = max(x1 + 1, min(x2, os_w))
    y2 = max(y1 + 1, min(y2, os_h))

    if not config.sam_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {config.sam_checkpoint}. "
            f"Expected at {SAM_PRETRAINED_ROOT}."
        )

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as e:
        raise ImportError(
            "segment_anything package is required for SAM aggressive mode. "
            "Install from https://github.com/facebookresearch/segment-anything"
        ) from e

    sam = sam_model_registry[config.model_type](checkpoint=str(config.sam_checkpoint))
    sam.to(device=config.device)
    predictor = SamPredictor(sam)

    return _run_sam_on_image(predictor, os_image_bgr, (x1, y1, x2, y2))


def build_sam_predictor(*, config: SamOnObjectStitchConfig):
    # 检查 sam_checkpoint 是本地路径还是 HuggingFace 模型 ID
    if isinstance(config.sam_checkpoint, Path):
        if not config.sam_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {config.sam_checkpoint}. "
                f"Expected at {SAM_PRETRAINED_ROOT}."
            )
        sam_checkpoint = str(config.sam_checkpoint)
    else:
        # 如果是字符串，可能是 HuggingFace 模型 ID，直接使用
        sam_checkpoint = config.sam_checkpoint

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as e:
        raise ImportError(
            "segment_anything package is required for SAM aggressive mode. "
            "Install from https://github.com/facebookresearch/segment-anything"
        ) from e
    
    sam = sam_model_registry[config.model_type](checkpoint=sam_checkpoint)
    sam.to(device=config.device)
    return SamPredictor(sam)


def run_sam_on_objectstitch_with_predictor(
    *,
    predictor,
    os_image: np.ndarray,
    bg_shape_hw: Tuple[int, int],
    bbox_xyxy_bg: Tuple[int, int, int, int],
) -> np.ndarray:
    os_image_bgr = os_image
    if os_image_bgr.ndim != 3 or os_image_bgr.shape[2] != 3:
        raise ValueError("Expected HxWx3 image")

    os_h, os_w = os_image_bgr.shape[:2]
    bg_h, bg_w = bg_shape_hw

    bx1, by1, bx2, by2 = bbox_xyxy_bg

    scale_x = os_w / float(bg_w) if bg_w > 0 else 1.0
    scale_y = os_h / float(bg_h) if bg_h > 0 else 1.0

    x1 = int(bx1 * scale_x)
    y1 = int(by1 * scale_y)
    x2 = int(bx2 * scale_x)
    y2 = int(by2 * scale_y)

    x1 = max(0, min(x1, os_w - 1))
    y1 = max(0, min(y1, os_h - 1))
    x2 = max(x1 + 1, min(x2, os_w))
    y2 = max(y1 + 1, min(y2, os_h))

    return _run_sam_on_image(predictor, os_image_bgr, (x1, y1, x2, y2))
