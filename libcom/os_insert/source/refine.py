from __future__ import annotations

from pathlib import Path

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

from .ia_utils import (
    get_bbox_from_mask,
    expand_bbox,
    pad_to_square,
    box2squre,
    crop_back,
    expand_image_mask,
)

dtype = torch.bfloat16
size = (768, 768)


# ---------------------------------------------------------------------------
# 模型路径：对齐 libcom 的 `pretrained_models/` 目录约定。
# ---------------------------------------------------------------------------

_PIPE: FluxFillPipeline | None = None
_REDUX: FluxPriorReduxPipeline | None = None
_PIPE_KEY: tuple[str, str, str, str] | None = None


def _resolve_ia_paths(
    model_dir: str | Path | None,
    *,
    flux_fill_path: str | Path | None = None,
    flux_redux_path: str | Path | None = None,
    ia_lora_path: str | Path | None = None,
):
    if model_dir is None:
        base = None
    else:
        base = Path(model_dir)

    # Priority: explicit args > env vars > model_dir convention
    flux_fill = str(flux_fill_path) if flux_fill_path is not None else os.getenv("FLUX_FILL_PATH")
    flux_redux = str(flux_redux_path) if flux_redux_path is not None else os.getenv("FLUX_REDUX_PATH")
    ia_lora = str(ia_lora_path) if ia_lora_path is not None else os.getenv("IA_LORA_PATH")

    if flux_fill is None:
        if base is None:
            raise ValueError("model_dir must be provided when FLUX_FILL_PATH is not set")
        flux_fill = str(base / "flux" / "FLUX.1-Fill-dev")
    if flux_redux is None:
        if base is None:
            raise ValueError("model_dir must be provided when FLUX_REDUX_PATH is not set")
        flux_redux = str(base / "flux" / "FLUX.1-Redux-dev")
    if ia_lora is None:
        if base is None:
            raise ValueError("model_dir must be provided when IA_LORA_PATH is not set")
        ia_lora = str(base / "insert_anything" / "20250321_steps5000_pytorch_lora_weights.safetensors")

    return Path(flux_fill), Path(flux_redux), Path(ia_lora)


def _get_pipes(
    model_dir: str | Path | None,
    *,
    flux_fill_path: str | Path | None = None,
    flux_redux_path: str | Path | None = None,
    ia_lora_path: str | Path | None = None,
    device: str | torch.device | None = None,
):
    global _PIPE, _REDUX, _PIPE_KEY

    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)

    flux_fill_path, flux_redux_path, ia_lora_path = _resolve_ia_paths(
        model_dir,
        flux_fill_path=flux_fill_path,
        flux_redux_path=flux_redux_path,
        ia_lora_path=ia_lora_path,
    )
    device_key = str(device) if device is not None else "<diffusers_default>"
    key = (str(flux_fill_path), str(flux_redux_path), str(ia_lora_path), device_key)

    if _PIPE is not None and _REDUX is not None and _PIPE_KEY == key:
        return _PIPE, _REDUX

    pipe = FluxFillPipeline.from_pretrained(
        str(flux_fill_path),
        torch_dtype=dtype,
    )
    pipe.load_lora_weights(str(ia_lora_path))
    redux = FluxPriorReduxPipeline.from_pretrained(str(flux_redux_path)).to(dtype=dtype)

    # Prefer explicit device. If device is None, fall back to diffusers defaults.
    if device is None:
        pipe.enable_model_cpu_offload()
        redux.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
    elif device.type == "cuda":
        gpu_id = int(device.index or 0)
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        redux.enable_model_cpu_offload(gpu_id=gpu_id)
        pipe.enable_vae_slicing()
    else:
        # CPU mode: keep everything on CPU.
        pipe.to(device)
        redux.to(device)

    _PIPE = pipe
    _REDUX = redux
    _PIPE_KEY = key
    return pipe, redux



def _to_rgb_numpy(image: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(str(image))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if isinstance(image, Image.Image):
        arr = np.array(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected RGB PIL image")
        return arr
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected HxWx3 numpy image")
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_mask_u8(mask: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        m = cv2.imread(str(mask))
        if m is None:
            raise FileNotFoundError(str(mask))
        m = (m > 128).astype(np.uint8)[:, :, 0]
        return m
    if isinstance(mask, Image.Image):
        m = np.array(mask)
    elif isinstance(mask, np.ndarray):
        m = mask
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    if m.ndim == 3:
        m = m[:, :, 0]
    if m.ndim != 2:
        raise ValueError("Expected HxW mask")
    return (m > 128).astype(np.uint8)


def _basename_no_ext(x: str | Path | None, fallback: str) -> str:
    if isinstance(x, (str, Path)):
        base = os.path.basename(str(x))
        return os.path.splitext(base)[0]
    return fallback


def _normalize_suffix(s: str) -> str:
    if not s:
        return ""
    return s[1:] if s.startswith("_") else s


class InsertAnythingModel:
    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        flux_fill_path: str | Path | None = None,
        flux_redux_path: str | Path | None = None,
        ia_lora_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        self.device = device
        self.pipe, self.redux = _get_pipes(
            model_dir,
            flux_fill_path=flux_fill_path,
            flux_redux_path=flux_redux_path,
            ia_lora_path=ia_lora_path,
            device=device,
        )

    def __call__(
        self,
        *,
        source_image: str | Path | np.ndarray | Image.Image,
        mask_image: str | Path | np.ndarray | Image.Image,
        ref_image: str | Path | np.ndarray | Image.Image,
        ref_mask: str | Path | np.ndarray | Image.Image,
        sam_mask: str | Path | np.ndarray | Image.Image | None = None,
        seeds=123,
        strength: float | None = None,
        split_ratio: float = 0.5,
        save_path: str = "./result",
        filename_suffix: str = "",
        return_image: bool = False,
    ):
        return _run_insertanything_with_pipes(
            source_image=source_image,
            mask_image=mask_image,
            ref_image=ref_image,
            ref_mask=ref_mask,
            sam_mask=sam_mask,
            seeds=seeds,
            strength=strength,
            split_ratio=split_ratio,
            save_path=save_path,
            filename_suffix=filename_suffix,
            return_image=return_image,
            pipe=self.pipe,
            redux=self.redux,
            device=self.device,
        )


def _run_insertanything_with_pipes(
    *,
    source_image: str | Path | np.ndarray | Image.Image,
    mask_image: str | Path | np.ndarray | Image.Image,
    ref_image: str | Path | np.ndarray | Image.Image,
    ref_mask: str | Path | np.ndarray | Image.Image,
    sam_mask: str | Path | np.ndarray | Image.Image | None = None,
    seeds=123,
    strength: float | None = None,
    split_ratio: float = 0.5,
    save_path: str = "./result",
    filename_suffix: str = "",
    return_image: bool = False,
    pipe: FluxFillPipeline,
    redux: FluxPriorReduxPipeline,
    device: torch.device,
):
    if seeds is None:
        seeds = [666]

    ref_image_np = _to_rgb_numpy(ref_image)
    tar_image = _to_rgb_numpy(source_image)

    ref_mask_np = _to_mask_u8(ref_mask)
    tar_mask = _to_mask_u8(mask_image)
    tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

    sam_mask_np = None
    if sam_mask is not None:
        sam_mask_np = _to_mask_u8(sam_mask)
        sam_mask_np = cv2.resize(sam_mask_np, (tar_image.shape[1], tar_image.shape[0]))

    # Remove the background information of the reference picture
    ref_box_yyxx = get_bbox_from_mask(ref_mask_np)
    ref_mask_3 = np.stack([ref_mask_np, ref_mask_np, ref_mask_np], -1)
    masked_ref_image = ref_image_np * ref_mask_3 + np.ones_like(ref_image_np) * 255 * (1 - ref_mask_3)

    # Extract the box where the reference image is located, and place the reference
    # object at the center of the image
    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask_crop = ref_mask_np[y1:y2, x1:x2]
    ratio = 1.3
    masked_ref_image, ref_mask_crop = expand_image_mask(masked_ref_image, ref_mask_crop, ratio=ratio)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)

    # Dilate the mask
    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

    # zoom in
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2)
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop

    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]
    if sam_mask_np is not None:
        sam_mask_np = sam_mask_np[y1:y2, x1:x2]

    H1, W1 = tar_image.shape[0], tar_image.shape[1]

    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)

    if sam_mask_np is not None:
        sam_mask_np = pad_to_square(sam_mask_np, pad_value=0)
        sam_mask_np = cv2.resize(sam_mask_np, size)

    # Extract the features of the reference image
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    pipe_prior_output = redux(Image.fromarray(masked_ref_image))

    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2 = tar_image.shape[0], tar_image.shape[1]

    tar_image = cv2.resize(tar_image, size)
    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)

    tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
    mask_black = np.ones_like(tar_image) * 0
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

    # BBox-based mask for the second half of denoising (bbox_mask)
    bbox_mask_diptych = mask_diptych.copy()

    # SAM/ObjectStitch-based mask for the first half of denoising (sam_mask)
    sam_mask_diptych = None
    if sam_mask_np is not None:
        sam_mask_rgb = np.stack([sam_mask_np, sam_mask_np, sam_mask_np], -1)
        sam_mask_diptych = np.concatenate([mask_black, sam_mask_rgb], axis=1)

    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)

    if bbox_mask_diptych is not None:
        bbox_mask_diptych[bbox_mask_diptych == 1] = 255
        bbox_mask_diptych = Image.fromarray(bbox_mask_diptych)

    if sam_mask_diptych is not None:
        sam_mask_diptych[sam_mask_diptych == 1] = 255
        sam_mask_diptych = Image.fromarray(sam_mask_diptych)

    os.makedirs(save_path, exist_ok=True)

    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        from ..diffusers_osinsert import patch_context

        with patch_context():
            edited_image = pipe(
                image=diptych_ref_tar,
                mask_image=mask_diptych,
                height=mask_diptych.size[1],
                width=mask_diptych.size[0],
                max_sequence_length=512,
                generator=generator,
                strength=strength if strength is not None else 1.0,
                sam_mask=sam_mask_diptych,
                bbox_mask=bbox_mask_diptych,
                split_ratio=split_ratio,
                **pipe_prior_output,
            ).images[0]

        width, height = edited_image.size
        left = width // 2
        right = width
        top = 0
        bottom = height
        edited_image = edited_image.crop((left, top, right, bottom))

        edited_image = np.array(edited_image)
        edited_image = crop_back(
            edited_image,
            old_tar_image,
            np.array([H1, W1, H2, W2]),
            np.array(tar_box_yyxx_crop),
        )
        edited_pil = Image.fromarray(edited_image)

        ref_without_ext = _basename_no_ext(ref_mask if isinstance(ref_mask, (str, Path)) else None, "ref")
        tar_without_ext = _basename_no_ext(mask_image if isinstance(mask_image, (str, Path)) else None, "tar")

        suffix_norm = _normalize_suffix(filename_suffix)
        if ref_without_ext == "ref" and tar_without_ext == "tar" and suffix_norm:
            base_name = suffix_norm
            seed_token = f"seed{seed}"
            needs_seed_suffix = (len(seeds) > 1) and (seed_token not in base_name)
            if needs_seed_suffix:
                edited_image_save_path = os.path.join(save_path, f"{base_name}_seed{seed}.png")
            else:
                edited_image_save_path = os.path.join(save_path, f"{base_name}.png")
        else:
            suffix = f"_{suffix_norm}" if suffix_norm else ""
            edited_image_save_path = os.path.join(
                save_path,
                f"{ref_without_ext}_to_{tar_without_ext}_seed{seed}{suffix}.png",
            )
        edited_pil.save(edited_image_save_path)

        if return_image:
            return np.array(edited_pil)

    return None


def run_insertanything(
    source_image_path: str | Path | np.ndarray | Image.Image,
    mask_image_path: str | Path | np.ndarray | Image.Image,
    ref_image_path: str | Path | np.ndarray | Image.Image,
    ref_mask_path: str | Path | np.ndarray | Image.Image,
    sam_mask_path: str | Path | np.ndarray | Image.Image | None = None,
    seeds=123,
    strength: float | None = None,
    split_ratio: float = 0.5,
    save_path: str = "./result",
    filename_suffix: str = "",
    model_dir: str | Path | None = None,
    flux_fill_path: str | Path | None = None,
    flux_redux_path: str | Path | None = None,
    ia_lora_path: str | Path | None = None,
    device: str | torch.device | None = None,
    net: InsertAnythingModel | None = None,
    *,
    return_image: bool = False,
):
    """Single-image InsertAnything inference following the original diptych pipeline."""

    if net is None:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        pipe, redux = _get_pipes(
            model_dir,
            flux_fill_path=flux_fill_path,
            flux_redux_path=flux_redux_path,
            ia_lora_path=ia_lora_path,
            device=device,
        )
    else:
        pipe, redux, device = net.pipe, net.redux, net.device

    return _run_insertanything_with_pipes(
        source_image=source_image_path,
        mask_image=mask_image_path,
        ref_image=ref_image_path,
        ref_mask=ref_mask_path,
        sam_mask=sam_mask_path,
        seeds=seeds,
        strength=strength,
        split_ratio=split_ratio,
        save_path=save_path,
        filename_suffix=filename_suffix,
        return_image=return_image,
        pipe=pipe,
        redux=redux,
        device=device,
    )


def insertanything_infer(
    *,
    source_image: str | Path | np.ndarray | Image.Image,
    mask_image: str | Path | np.ndarray | Image.Image,
    ref_image: str | Path | np.ndarray | Image.Image,
    ref_mask: str | Path | np.ndarray | Image.Image,
    sam_mask: str | Path | np.ndarray | Image.Image | None = None,
    seeds=123,
    strength: float | None = None,
    split_ratio: float = 0.5,
    save_path: str = "./result",
    filename_suffix: str = "",
    net: InsertAnythingModel | None = None,
    model_dir: str | Path | None = None,
    flux_fill_path: str | Path | None = None,
    flux_redux_path: str | Path | None = None,
    ia_lora_path: str | Path | None = None,
    device: str | torch.device | None = None,
    return_image: bool = True,
):
    if net is None:
        net = InsertAnythingModel(
            model_dir=model_dir,
            flux_fill_path=flux_fill_path,
            flux_redux_path=flux_redux_path,
            ia_lora_path=ia_lora_path,
            device=device,
        )

    return net(
        source_image=source_image,
        mask_image=mask_image,
        ref_image=ref_image,
        ref_mask=ref_mask,
        sam_mask=sam_mask,
        seeds=seeds,
        strength=strength,
        split_ratio=split_ratio,
        save_path=save_path,
        filename_suffix=filename_suffix,
        return_image=return_image,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Single-image InsertAnything inference using diptych pipeline",
    )
    parser.add_argument("--source_image", default="examples/source_image/1.png")
    parser.add_argument("--source_mask", default="examples/source_mask/1.png")
    parser.add_argument("--ref_image", default="examples/ref_image/1.png")
    parser.add_argument("--ref_mask", default="examples/ref_mask/1.png")
    parser.add_argument("--outdir", default="./result")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    run_insertanything(
        source_image_path=args.source_image,
        mask_image_path=args.source_mask,
        ref_image_path=args.ref_image,
        ref_mask_path=args.ref_mask,
        seeds=args.seeds,
        strength=args.strength,
        save_path=args.outdir,
    )


if __name__ == "__main__":
    main()
