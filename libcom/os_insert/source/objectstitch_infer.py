"""ObjectStitch inference wrapper for OSInsert (skeleton).

This module is the libcom-style home for ObjectStitch inference. It is
responsible for taking a single sample (background, foreground, masks, bbox)
and producing a coarse composite image that will later be refined by SAM and
InsertAnything.

The actual network definition and weight loading will be implemented in a
later phase. For now this file only defines the public API so that
`OSInsertModel` can depend on a stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import importlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import Resize

from .objectstitch_data import bbox2mask, get_bbox_tensor, get_tensor, get_tensor_clip

REPO_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_ROOT = REPO_ROOT / "pretrained_models" / "objectstitch"

from libcom.os_insert.source.ldm.models.diffusion.ddim import DDIMSampler


@dataclass
class ObjectStitchConfig:
    """Configuration for ObjectStitch inference.

    Parameters are intentionally minimal; more options can be added when we
    hook up the real model implementation.
    """

    ckpt_path: Path = PRETRAINED_ROOT / "v1" / "model.ckpt"
    config_path: Path = PRETRAINED_ROOT / "v1" / "configs" / "v1.yaml"
    clip_dir: Path | None = None
    device: str = "cuda:0"


def instantiate_from_config(config):
    """Instantiate a module from an OmegaConf-style config.

    This is a local copy of the utility used in ObjectStitch's ldm.util.
    """

    if "target" not in config:
        if config in {"__is_first_stage__", "__is_unconditional__"}:
            return None
        raise KeyError("Expected key `target` to instantiate.")

    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        # 确保使用本地的 ldm 模块
        if module.startswith("ldm"):
            # 使用绝对导入路径
            local_module = f"libcom.os_insert.source.{module}"
            if reload:
                module_imp = importlib.import_module(local_module)
                importlib.reload(module_imp)
            return getattr(importlib.import_module(local_module, package=None), cls)
        else:
            # 其他模块使用标准导入
            if reload:
                module_imp = importlib.import_module(module)
                importlib.reload(module_imp)
            return getattr(importlib.import_module(module, package=None), cls)

    return get_obj_from_str(config["target"])(**config.get("params", {}))


def load_model_from_config(cfg_path: Path, ckpt_path: Path, verbose: bool = False) -> torch.nn.Module:
    """Load ObjectStitch model from config and checkpoint.

    This mirrors the `load_model_from_config` helper in ObjectStitch's
    `scripts/inference.py`, but is rooted in this repository's
    `pretrained_models/objectstitch` directory.
    """

    cfg_path = Path(cfg_path)
    ckpt_path = Path(ckpt_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"ObjectStitch config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ObjectStitch checkpoint not found: {ckpt_path}")

    print(f"[ObjectStitch] load config {cfg_path}")
    config = OmegaConf.load(str(cfg_path))

    print(f"[ObjectStitch] load checkpoint {ckpt_path}")
    pl_sd = torch.load(str(ckpt_path), map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd

    model = instantiate_from_config(config["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if verbose:
        if missing:
            print("[ObjectStitch] missing keys:")
            print(missing)
        if unexpected:
            print("[ObjectStitch] unexpected keys:")
            print(unexpected)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data transforms & helpers (ported from scripts/inference.py and
# ldm.data.open_images).
# ---------------------------------------------------------------------------

clip_transform = get_tensor_clip(image_size=(224, 224))
sd_transform = get_tensor(image_size=(512, 512))
mask_transform = get_tensor(normalize=False, image_size=(512, 512))


def clip2sd(x: torch.Tensor) -> torch.Tensor:
    """Convert CLIP-normalized tensor to Stable Diffusion space.

    Copied from ObjectStitch's `scripts/inference.py`.
    """

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    denorm = x * std + mean
    sd_x = denorm * 2 - 1
    return sd_x


def generate_image_batch(
    bg_path: Path,
    fg_path: Path,
    bbox_xyxy: Tuple[int, int, int, int],
    fg_mask_path: Path,
) -> dict:
    """Prepare a single-sample batch for ObjectStitch.

    This is a slightly simplified version of `generate_image_batch` from
    ObjectStitch's `scripts/inference.py`, fixed to always use bbox and fg_mask
    (which matches the OSInsert aggressive pipeline).
    """

    bg_img = Image.open(bg_path).convert("RGB")
    bg_w, bg_h = bg_img.size
    bg_t = sd_transform(bg_img)

    fg_img = Image.open(fg_path).convert("RGB")
    fg_mask = Image.open(fg_mask_path).convert("RGB")
    fg_mask = fg_mask.resize((fg_img.width, fg_img.height))

    black = np.zeros_like(np.asarray(fg_mask))
    fg_mask_np = np.asarray(fg_mask)
    fg_img_np = np.asarray(fg_img)
    fg_img_np = np.where(fg_mask_np > 127, fg_img_np, black)
    fg_img = Image.fromarray(fg_img_np)

    fg_t = clip_transform(fg_img)

    mask_np = bbox2mask(bbox_xyxy, bg_w, bg_h)
    mask = Image.fromarray(mask_np)
    mask_t = mask_transform(mask)
    mask_t = torch.where(mask_t > 0.5, 1, 0).float()

    inpaint_t = bg_t * (1 - mask_t)
    bbox_t = get_bbox_tensor(bbox_xyxy, bg_w, bg_h)

    return {
        "bg_img": inpaint_t.unsqueeze(0),
        "bg_mask": mask_t.unsqueeze(0),
        "fg_img": fg_t.unsqueeze(0),
        "bbox": bbox_t.unsqueeze(0),
    }


def _ensure_pil_rgb(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not isinstance(image, np.ndarray):
        raise TypeError("Expected numpy array or PIL.Image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HxWx3 image")
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


def _ensure_mask_uint8(mask: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask_np = np.asarray(mask)
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        raise TypeError("Expected numpy array or PIL.Image for mask")

    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.ndim != 2:
        raise ValueError("Expected HxW mask")
    return mask_np.astype(np.uint8)


def generate_image_batch_from_images(
    bg_image: np.ndarray | Image.Image,
    fg_image: np.ndarray | Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    fg_mask: np.ndarray | Image.Image,
) -> dict:
    """In-memory version of :func:`generate_image_batch`.

    Parameters are identical in semantics, but accept already-loaded images.
    """

    bg_img = _ensure_pil_rgb(bg_image)
    bg_w, bg_h = bg_img.size
    bg_t = sd_transform(bg_img)

    fg_img = _ensure_pil_rgb(fg_image)
    fg_mask_np = _ensure_mask_uint8(fg_mask)

    if fg_mask_np.shape[0] != fg_img.height or fg_mask_np.shape[1] != fg_img.width:
        fg_mask_pil = Image.fromarray(fg_mask_np)
        fg_mask_pil = fg_mask_pil.resize((fg_img.width, fg_img.height))
        fg_mask_np = np.asarray(fg_mask_pil).astype(np.uint8)

    black = np.zeros((fg_img.height, fg_img.width, 3), dtype=np.uint8)
    fg_img_np = np.asarray(fg_img).astype(np.uint8)
    fg_img_np = np.where(fg_mask_np[:, :, None] > 127, fg_img_np, black)
    fg_img = Image.fromarray(fg_img_np)

    fg_t = clip_transform(fg_img)

    mask_np = bbox2mask(bbox_xyxy, bg_w, bg_h)
    mask = Image.fromarray(mask_np)
    mask_t = mask_transform(mask)
    mask_t = torch.where(mask_t > 0.5, 1, 0).float()

    inpaint_t = bg_t * (1 - mask_t)
    bbox_t = get_bbox_tensor(bbox_xyxy, bg_w, bg_h)

    return {
        "bg_img": inpaint_t.unsqueeze(0),
        "bg_mask": mask_t.unsqueeze(0),
        "fg_img": fg_t.unsqueeze(0),
        "bbox": bbox_t.unsqueeze(0),
    }


def prepare_input(
    batch: dict,
    model: torch.nn.Module,
    latent_shape: Tuple[int, int, int],
    device: torch.device,
    num_samples: int = 1,
):
    """Prepare model kwargs and conditioning vectors for sampling.

    Port of ObjectStitch's `prepare_input` function, adapted to our config.
    """

    if num_samples > 1:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = torch.cat([v] * num_samples, dim=0)

    test_model_kwargs: dict = {}

    bg_img = batch["bg_img"].to(device)
    bg_latent = model.encode_first_stage(bg_img)
    bg_latent = model.get_first_stage_encoding(bg_latent).detach()
    test_model_kwargs["bg_latent"] = bg_latent

    rs_mask = F.interpolate(batch["bg_mask"].to(device), latent_shape[-2:])
    rs_mask = torch.where(rs_mask > 0.5, 1.0, 0.0)
    test_model_kwargs["bg_mask"] = rs_mask

    test_model_kwargs["bbox"] = batch["bbox"].to(device)

    fg_tensor = batch["fg_img"].to(device)
    c = model.get_learned_conditioning(fg_tensor)
    c = model.proj_out(c)
    uc = model.learnable_vector.repeat(c.shape[0], c.shape[1], 1)

    return test_model_kwargs, c, uc


def tensor2numpy(image: torch.Tensor, normalized: bool = False, image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Convert a BCHW tensor in [-1,1] or [0,1] to uint8 numpy images.

    Port of ObjectStitch's `tensor2numpy` utility.
    """

    image = Resize(image_size, antialias=True)(image)
    if not normalized:
        image = (image + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 2, 3, 1)
    image = image.detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image


def run_objectstitch_single(
    bg_path: Path,
    fg_path: Path,
    fg_mask_path: Path,
    bbox_xyxy: Tuple[int, int, int, int],
    *,
    config: ObjectStitchConfig,
    out_dir: Path | None = None,
    seed: int | None = None,
) -> Path:
    """Run ObjectStitch for a single sample and return the coarse composite path.

    This wraps the original ObjectStitch `scripts/inference.py` logic into a
    single-call API for OSInsert aggressive mode.
    """

    if out_dir is None:
        out_dir = bg_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load OmegaConf config and patch CLIP path to local directory if present.
    cfg = OmegaConf.load(str(config.config_path))
    clip_dir = config.clip_dir
    if clip_dir is None:
        clip_dir = config.config_path.parent / "openai-clip-vit-large-patch14"
        if not clip_dir.exists():
            clip_dir = config.ckpt_path.parent / "openai-clip-vit-large-patch14"
    if clip_dir is not None and Path(clip_dir).exists():
        cfg.model.params.cond_stage_config.params.version = str(clip_dir)

    # Instantiate model and sampler.
    model = instantiate_from_config(cfg["model"])
    pl_sd = torch.load(str(config.ckpt_path), map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[ObjectStitch] missing keys:", missing)
    if unexpected:
        print("[ObjectStitch] unexpected keys:", unexpected)

    device = torch.device(config.device)
    model = model.to(device)

    img_size = (512, 512)
    latent_shape = (4, img_size[1] // 8, img_size[0] // 8)
    sample_steps = 50
    num_samples = 1
    guidance_scale = 5.0

    sampler = DDIMSampler(model)

    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    start_code = torch.randn((num_samples, *latent_shape), device=device)

    batch = generate_image_batch(bg_path, fg_path, bbox_xyxy, fg_mask_path)
    test_model_kwargs, c, uc = prepare_input(batch, model, latent_shape, device, num_samples)

    samples_ddim, _ = sampler.sample(
        S=sample_steps,
        conditioning=c,
        batch_size=num_samples,
        shape=list(latent_shape),
        verbose=False,
        eta=0.0,
        x_T=start_code,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=uc,
        test_model_kwargs=test_model_kwargs,
    )

    x_samples_ddim = model.decode_first_stage(samples_ddim[:, :4]).cpu().float()
    comp_img = tensor2numpy(x_samples_ddim, image_size=img_size)
    out_path = out_dir / "objectstitch_coarse.png"
    Image.fromarray(comp_img[0]).save(out_path)

    return out_path


def run_objectstitch_single_image(
    bg_path: Path,
    fg_path: Path,
    fg_mask_path: Path,
    bbox_xyxy: Tuple[int, int, int, int],
    *,
    config: ObjectStitchConfig,
    seed: int | None = None,
) -> np.ndarray:
    """Run ObjectStitch for a single sample and return the coarse composite image (RGB uint8)."""

    # Load OmegaConf config and patch CLIP path to local directory if present.
    cfg = OmegaConf.load(str(config.config_path))
    clip_dir = config.clip_dir
    if clip_dir is None:
        clip_dir = config.config_path.parent / "openai-clip-vit-large-patch14"
        if not clip_dir.exists():
            clip_dir = config.ckpt_path.parent / "openai-clip-vit-large-patch14"
    if clip_dir is not None and Path(clip_dir).exists():
        cfg.model.params.cond_stage_config.params.version = str(clip_dir)

    model = instantiate_from_config(cfg["model"])
    pl_sd = torch.load(str(config.ckpt_path), map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[ObjectStitch] missing keys:", missing)
    if unexpected:
        print("[ObjectStitch] unexpected keys:", unexpected)

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    img_size = (512, 512)
    latent_shape = (4, img_size[1] // 8, img_size[0] // 8)
    sample_steps = 50
    num_samples = 1
    guidance_scale = 5.0

    sampler = DDIMSampler(model)

    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
    start_code = torch.randn((num_samples, *latent_shape), device=device)

    batch = generate_image_batch(bg_path, fg_path, bbox_xyxy, fg_mask_path)
    test_model_kwargs, c, uc = prepare_input(batch, model, latent_shape, device, num_samples)

    samples_ddim, _ = sampler.sample(
        S=sample_steps,
        conditioning=c,
        batch_size=num_samples,
        shape=list(latent_shape),
        verbose=False,
        eta=0.0,
        x_T=start_code,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=uc,
        test_model_kwargs=test_model_kwargs,
    )

    x_samples_ddim = model.decode_first_stage(samples_ddim[:, :4]).cpu().float()
    comp_img = tensor2numpy(x_samples_ddim, image_size=img_size)
    return comp_img[0]


def load_objectstitch_model_and_sampler(*, config: ObjectStitchConfig) -> tuple[torch.nn.Module, DDIMSampler]:
    """Load ObjectStitch model + sampler once.

    The returned model is moved to ``config.device`` and set to ``eval()``.
    """

    cfg = OmegaConf.load(str(config.config_path))
    clip_dir = config.clip_dir
    if clip_dir is None:
        clip_dir = config.config_path.parent / "openai-clip-vit-large-patch14"
        if not clip_dir.exists():
            clip_dir = config.ckpt_path.parent / "openai-clip-vit-large-patch14"
    if clip_dir is not None and Path(clip_dir).exists():
        cfg.model.params.cond_stage_config.params.version = str(clip_dir)

    model = instantiate_from_config(cfg["model"])
    pl_sd = torch.load(str(config.ckpt_path), map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[ObjectStitch] missing keys:", missing)
    if unexpected:
        print("[ObjectStitch] unexpected keys:", unexpected)

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    return model, sampler


def run_objectstitch_single_image_from_images_cached(
    *,
    background: np.ndarray | Image.Image,
    foreground: np.ndarray | Image.Image,
    foreground_mask: np.ndarray | Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    model: torch.nn.Module,
    sampler: DDIMSampler,
    device: str | torch.device,
    seed: int | None = None,
    split_steps: int = 50,
) -> np.ndarray:
    """Cached in-memory ObjectStitch inference.

    This avoids re-loading weights for every call.
    """

    img_size = (512, 512)
    latent_shape = (4, img_size[1] // 8, img_size[0] // 8)
    num_samples = 1
    guidance_scale = 5.0

    device_t = torch.device(device)

    if seed is not None:
        torch.manual_seed(int(seed))
        if device_t.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
    start_code = torch.randn((num_samples, *latent_shape), device=device_t)

    batch = generate_image_batch_from_images(background, foreground, bbox_xyxy, foreground_mask)
    test_model_kwargs, c, uc = prepare_input(batch, model, latent_shape, device_t, num_samples)

    samples_ddim, _ = sampler.sample(
        S=int(split_steps),
        conditioning=c,
        batch_size=num_samples,
        shape=list(latent_shape),
        verbose=False,
        eta=0.0,
        x_T=start_code,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=uc,
        test_model_kwargs=test_model_kwargs,
    )

    x_samples_ddim = model.decode_first_stage(samples_ddim[:, :4]).cpu().float()
    comp_img = tensor2numpy(x_samples_ddim, image_size=img_size)
    return comp_img[0]


def run_objectstitch_single_image_from_images(
    *,
    background: np.ndarray | Image.Image,
    foreground: np.ndarray | Image.Image,
    foreground_mask: np.ndarray | Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    config: ObjectStitchConfig,
    seed: int | None = None,
) -> np.ndarray:
    """Run ObjectStitch for a single sample and return the coarse composite image (RGB uint8).

    This is the in-memory variant of :func:`run_objectstitch_single_image`.
    """

    cfg = OmegaConf.load(str(config.config_path))
    clip_dir = config.clip_dir
    if clip_dir is None:
        clip_dir = config.config_path.parent / "openai-clip-vit-large-patch14"
        if not clip_dir.exists():
            clip_dir = config.ckpt_path.parent / "openai-clip-vit-large-patch14"
    if clip_dir is not None and Path(clip_dir).exists():
        cfg.model.params.cond_stage_config.params.version = str(clip_dir)

    model = instantiate_from_config(cfg["model"])
    pl_sd = torch.load(str(config.ckpt_path), map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[ObjectStitch] missing keys:", missing)
    if unexpected:
        print("[ObjectStitch] unexpected keys:", unexpected)

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    img_size = (512, 512)
    latent_shape = (4, img_size[1] // 8, img_size[0] // 8)
    sample_steps = 50
    num_samples = 1
    guidance_scale = 5.0

    sampler = DDIMSampler(model)

    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
    start_code = torch.randn((num_samples, *latent_shape), device=device)

    batch = generate_image_batch_from_images(background, foreground, bbox_xyxy, foreground_mask)
    test_model_kwargs, c, uc = prepare_input(batch, model, latent_shape, device, num_samples)

    samples_ddim, _ = sampler.sample(
        S=sample_steps,
        conditioning=c,
        batch_size=num_samples,
        shape=list(latent_shape),
        verbose=False,
        eta=0.0,
        x_T=start_code,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=uc,
        test_model_kwargs=test_model_kwargs,
    )

    x_samples_ddim = model.decode_first_stage(samples_ddim[:, :4]).cpu().float()
    comp_img = tensor2numpy(x_samples_ddim, image_size=img_size)
    return comp_img[0]
