from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torchvision


def get_tensor(normalize: bool = True, to_tensor: bool = True, resize: bool = True, image_size: Tuple[int, int] = (512, 512)):
    """Basic image -> tensor transform used by ObjectStitch (Stable Diffusion style).

    This is a minimal copy of the preprocessing from ObjectStitch's
    `ldm.data.open_images.get_tensor`, without any dataset-specific logic.
    """

    transform_list: List[torch.nn.Module] = []
    if resize:
        transform_list.append(torchvision.transforms.Resize(image_size))
    if to_tensor:
        transform_list.append(torchvision.transforms.ToTensor())
    if normalize:
        transform_list.append(
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            )
        )
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize: bool = True, to_tensor: bool = True, resize: bool = True, image_size: Tuple[int, int] = (224, 224)):
    """Foreground (CLIP-style) image transform from ObjectStitch."""

    transform_list: List[torch.nn.Module] = []
    if resize:
        transform_list.append(torchvision.transforms.Resize(image_size))
    if to_tensor:
        transform_list.append(torchvision.transforms.ToTensor())
    if normalize:
        transform_list.append(
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
        )
    return torchvision.transforms.Compose(transform_list)


def bbox2mask(bbox: Iterable[int], mask_w: int, mask_h: int) -> np.ndarray:
    """Create a binary mask (uint8, 0/255) from an (x1,y1,x2,y2) bbox."""

    x1, y1, x2, y2 = map(int, bbox)
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def get_bbox_tensor(bbox: Iterable[float], width: int, height: int) -> torch.Tensor:
    """Normalize bbox coordinates to [0,1] as in ObjectStitch."""

    norm_bbox = torch.tensor(list(bbox), dtype=torch.float32).reshape(-1)
    norm_bbox[0::2] /= float(width)
    norm_bbox[1::2] /= float(height)
    return norm_bbox
