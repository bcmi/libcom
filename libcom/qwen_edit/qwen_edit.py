"""libcom-style Qwen-Image-2.1 composition using Diffusers."""

import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .lora import load_lora


PROMPTS = {
    "photorealistic": (
        "Place the object from Image 2 inside the red bounding box in Image 1. "
        "Remove the red bounding box and harmonize the object to match the "
        "photorealistic background. Generate shadow/reflection for the object accordingly."
    ),
    "artistic": (
        "Place the object from Image 2 inside the red bounding box in Image 1. "
        "Remove the red bounding box and stylize the object to match the artistic "
        "background. Generate shadow/reflection for the object accordingly."
    ),
}

# External public base model: fetched via from_pretrained when no local path is given.
QWEN21_BASE_HF_ID = "Qwen/Qwen-Image-2.1"

# Default LoRA checkpoints, downloaded through the libcom pretrained-model
# mechanism (BCMIZB/Libcom_pretrained_models) when missing locally.
DEFAULT_LORA_FILES = {
    "artistic": "artistic-e15.safetensors",
    "photorealistic": "photorealistic-e15.safetensors",
}


def _resolve_lora_path(model_type, lora_path, lora_paths):
    """Choose the adapter for one mode without silently reusing the other mode's LoRA."""
    if lora_paths is not None:
        if lora_path is not None:
            raise ValueError("Pass either lora_path or lora_paths, not both")
        if not isinstance(lora_paths, Mapping):
            raise TypeError("lora_paths must map model_type to a checkpoint path")
        unknown = set(lora_paths) - set(PROMPTS)
        if unknown:
            raise ValueError(f"Unknown lora_paths modes: {sorted(unknown)}")
        selected = lora_paths.get(model_type)
        if not selected:
            raise ValueError(f"lora_paths has no checkpoint for {model_type}")
        return selected
    if lora_path is not None:
        if not lora_path:
            raise ValueError("lora_path must be nonempty")
        return lora_path
    variable = f"LIBCOM_QWEN21_{model_type.upper()}_LORA_PATH"
    selected = os.environ.get(variable)
    if not selected:
        # Fall back to the libcom pretrained-model downloader (BCMIZB repo).
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'pretrained_models',
            DEFAULT_LORA_FILES[model_type],
        )
        try:
            from libcom.utils.model_download import download_pretrained_model
            return download_pretrained_model(candidate)
        except Exception as error:
            raise ValueError(
                f"Set lora_path, lora_paths[{model_type!r}], or {variable}; "
                f"automatic download of {DEFAULT_LORA_FILES[model_type]} also failed: {error}"
            ) from error
    return selected


def _rgb(value):
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as source:
            return source.convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError("Image arrays must be uint8 H×W×3 in BGR order")
        return Image.fromarray(np.ascontiguousarray(value[:, :, ::-1]), "RGB")
    raise TypeError("Image must be a path, PIL image, or BGR uint8 array")


def _mask(value, size):
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as source:
            mask = source.convert("L")
    elif isinstance(value, Image.Image):
        mask = value.convert("L")
    elif isinstance(value, np.ndarray):
        if value.ndim == 3:
            value = value[:, :, 0]
        if value.ndim != 2:
            raise ValueError("Mask must have shape H×W")
        if value.dtype == np.bool_ or (value.size and value.max() <= 1):
            value = value.astype(np.uint8) * 255
        mask = Image.fromarray(np.clip(value, 0, 255).astype(np.uint8), "L")
    else:
        raise TypeError("Mask must be a path, PIL image, or array")
    return mask.resize(size, Image.Resampling.NEAREST)


def _bucket(image):
    """Match the training operator's at-most-1MP, 32-pixel geometry."""
    width, height = image.size
    if width * height > 1_048_576:
        factor = math.sqrt(width * height / 1_048_576)
        target_width, target_height = int(width / factor), int(height / factor)
    else:
        target_width, target_height = width, height
    target_width = target_width // 32 * 32
    target_height = target_height // 32 * 32
    if min(target_width, target_height) < 32:
        raise ValueError(f"Image too small for training bucket: {image.size}")
    scale = max(target_width / width, target_height / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    left = round((resized_width - target_width) / 2)
    top = round((resized_height - target_height) / 2)
    return resized.crop((left, top, left + target_width, top + target_height)), (scale, left, top)


def _background_target_size(width, height):
    """Match the MureCom preprocessing size near 1MP without changing aspect ratio."""
    ratio = width / height
    ideal_height = math.sqrt(1_048_576 / ratio)
    center = max(1, round(ideal_height / 16))
    candidates = []
    for height_units in range(max(1, center - 5), center + 6):
        target_height = height_units * 16
        ideal_width_units = target_height * ratio / 16
        for width_units in {max(1, math.floor(ideal_width_units)), max(1, math.ceil(ideal_width_units))}:
            target_width = width_units * 16
            area_error = abs(math.log(target_width * target_height / 1_048_576))
            ratio_error = abs(math.log((target_width / target_height) / ratio))
            candidates.append((area_error + 4.0 * ratio_error, target_width, target_height))
    _, target_width, target_height = min(candidates)
    return target_width, target_height


def _background_with_box(image, bbox):
    if len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (int(v) for v in bbox)
    width, height = image.size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox is outside the background image: {bbox}")
    target_size = _background_target_size(width, height)
    normalized = image.resize(target_size, Image.Resampling.BICUBIC)
    x1 = math.floor(x1 * target_size[0] / width)
    x2 = math.ceil(x2 * target_size[0] / width)
    y1 = math.floor(y1 * target_size[1] / height)
    y2 = math.ceil(y2 * target_size[1] / height)
    processed, (scale, left, top) = _bucket(normalized)
    x1, x2 = round(x1 * scale) - left, round(x2 * scale) - left
    y1, y2 = round(y1 * scale) - top, round(y2 * scale) - top
    x1, x2 = max(0, x1), min(processed.width, x2)
    y1, y2 = max(0, y1), min(processed.height, y2)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox vanished after the training crop")
    line_width = max(3, round(min(processed.size) / 256))
    pixels = np.asarray(processed)
    red_channel = pixels[:, :, 0].astype(np.int16)
    green_channel = pixels[:, :, 1].astype(np.int16)
    blue_channel = pixels[:, :, 2].astype(np.int16)
    red = ((red_channel >= 220) & (green_channel <= 60) &
           (blue_channel <= 60) & (red_channel >= green_channel + 120) &
           (red_channel >= blue_channel + 120))
    strip = line_width + 2
    top = red[y1:min(y2, y1 + strip), x1:x2].any(axis=0).mean()
    bottom = red[max(y1, y2 - strip):y2, x1:x2].any(axis=0).mean()
    left_edge = red[y1:y2, x1:min(x2, x1 + strip)].any(axis=1).mean()
    right_edge = red[y1:y2, max(x1, x2 - strip):x2].any(axis=1).mean()
    # A training crop can clip one rectangle edge at the image boundary.
    if x1 == 0:
        left_edge = 1.0
    if y1 == 0:
        top = 1.0
    if x2 == processed.width:
        right_edge = 1.0
    if y2 == processed.height:
        bottom = 1.0
    if min(top, bottom, left_edge, right_edge) < 0.6:
        ImageDraw.Draw(processed).rectangle(
            (x1, y1, x2 - 1, y2 - 1), outline=(255, 0, 0), width=line_width
        )
    return processed


def _white_foreground(image, mask):
    if mask is None:
        if image.size != (1024, 1024):
            raise ValueError("Without foreground_mask, pass a prepared white-background 1024×1024 foreground")
        return image
    bounds = mask.point(lambda value: 255 if value > 127 else 0).getbbox()
    if bounds is None:
        raise ValueError("foreground_mask has no foreground")
    crop = image.crop(bounds)
    crop_mask = mask.crop(bounds)
    scale = 1024 / max(crop.size)
    size = (min(1024, max(1, round(crop.width * scale))),
            min(1024, max(1, round(crop.height * scale))))
    resized = crop.resize(size, Image.Resampling.BICUBIC)
    resized_mask = crop_mask.resize(size, Image.Resampling.BICUBIC)
    white_object = Image.composite(resized, Image.new("RGB", size, "white"), resized_mask)
    canvas = Image.new("RGB", (1024, 1024), "white")
    canvas.paste(white_object, ((1024 - size[0]) // 2, (1024 - size[1]) // 2))
    return canvas


class QwenEditModel:
    """
    Qwen-Image-2.1 based object insertion with mode-specific LoRA adapters.

    The model inserts a foreground object into a specified background region and
    can generate matching shadows and reflections. Use a clean background: the
    model does not remove objects already present in the bounding box.

    Args:
        device (int | str): CUDA device index or device string, such as 0 or
            ``"cuda:0"``. Default: 0.
        model_type (str): ``"photorealistic"`` or ``"artistic"``. Default:
            ``"photorealistic"``.
        kwargs: Optional model settings:

            - ``model_path`` (str | Path): Local base model path. Defaults to
              ``LIBCOM_QWEN21_MODEL_PATH``, or the Hugging Face model
              ``Qwen/Qwen-Image-2.1`` when unset.
            - ``lora_path`` (str | Path): LoRA checkpoint for the selected mode.
            - ``lora_paths`` (mapping): Maps model types to LoRA checkpoints.
              Cannot be used together with ``lora_path``. Without either option,
              the mode-specific ``LIBCOM_QWEN21_<MODE>_LORA_PATH`` environment
              variable or the default downloaded checkpoint is used.
            - ``lora_scale`` (float): LoRA scale. Default: 1.0.
            - ``local_files_only`` (bool): Whether to use only local base model
              files. Defaults to True for a local path and False otherwise.
            - ``cpu_offload`` (bool): Offload model components to CPU when not
              in use. Default: False.

    Examples:
        >>> import cv2
        >>> from libcom import QwenEditModel
        >>> from libcom.utils.process_image import draw_bbox_on_image, make_image_grid
        >>> net = QwenEditModel(device=0, model_type="photorealistic")
        >>> sample = "../tests/qwen_edit_source/photorealistic/1/"
        >>> background = sample + "background.png"
        >>> foreground = sample + "foreground.png"
        >>> mask = sample + "foreground_mask.png"
        >>> bbox = [153, 563, 749, 870]
        >>> result = net(background, foreground, bbox, foreground_mask=mask, seed=2511)
        >>> background_with_box = draw_bbox_on_image(background, bbox, color=(0, 0, 255))
        >>> grid_img = make_image_grid([background_with_box, foreground, result[0]])
        >>> cv2.imshow("Qwen Edit", grid_img)


    Expected result:

    The grid shows the background with the red bounding box, the foreground,
    and the generated composite from left to right. Generated details may vary.

    .. image:: _static/image/qwen_edit_result1.png
    """

    def __init__(self, device=0, model_type="photorealistic", **kwargs):
        if model_type not in PROMPTS:
            raise ValueError(f"model_type must be one of {tuple(PROMPTS)}")
        local_model_path = kwargs.pop("model_path", None) or os.environ.get("LIBCOM_QWEN21_MODEL_PATH")
        lora_path = _resolve_lora_path(model_type, kwargs.pop("lora_path", None),
                                       kwargs.pop("lora_paths", None))
        scale = kwargs.pop("lora_scale", 1.0)
        if local_model_path:
            model_path = local_model_path
            default_local_only = True
        else:
            # No local base model: fetch the official public checkpoint from HuggingFace.
            print(f"[QwenEditModel] No local base model found, falling back to "
                  f"HuggingFace id {QWEN21_BASE_HF_ID} (~31GB, downloaded on first use)")
            model_path = QWEN21_BASE_HF_ID
            default_local_only = False
        local_files_only = kwargs.pop("local_files_only", default_local_only)
        cpu_offload = kwargs.pop("cpu_offload", False)
        if kwargs:
            raise TypeError(f"Unknown model options: {sorted(kwargs)}")
        import torch
        from diffusers import QwenImage21Pipeline

        self.model_type = model_type
        self.device = f"cuda:{device}" if isinstance(device, int) else str(device)
        if not self.device.startswith("cuda"):
            raise ValueError("QwenEditModel requires a CUDA device")
        self.pipeline = QwenImage21Pipeline.from_pretrained(
            str(model_path), dtype=torch.bfloat16, local_files_only=local_files_only
        )
        self.lora_report = load_lora(self.pipeline, lora_path, scale=scale,
                                     adapter_name=f"qwen_edit_{model_type}")
        self.lora_report["model_type"] = model_type
        if cpu_offload:
            gpu_id = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            self.pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            self.pipeline.to(self.device)

    def inputs_preprocess(self, background_image, foreground_image, bbox, foreground_mask=None):
        background = _rgb(background_image)
        foreground = _rgb(foreground_image)
        mask = _mask(foreground_mask, foreground.size)
        return _background_with_box(background, bbox), _white_foreground(foreground, mask)

    def __call__(self, background_image, foreground_image, bbox,
                 foreground_mask=None, prompt=None, num_samples=1,
                 sample_steps=20, guidance_scale=1.0, seed=2511):
        """
        Insert the foreground object into the background.

        Args:
            background_image (str | Path | PIL.Image | numpy.ndarray): Clean
                background image. NumPy images must be uint8 in BGR order.
            foreground_image (str | Path | PIL.Image | numpy.ndarray): Foreground
                image, with NumPy images in uint8 BGR order.
            bbox (list | tuple): In-bounds background coordinates
                ``[x1, y1, x2, y2]``.
            foreground_mask (str | Path | PIL.Image | numpy.ndarray | None):
                Mask identifying the foreground object. If omitted, the
                foreground must already be a 1024x1024 image on white.
            prompt (str | None): Override the mode-specific prompt.
            num_samples (int): Number of images to generate. Default: 1.
            sample_steps (int): Number of inference steps; only 20 is supported.
            guidance_scale (float): Guidance scale; only 1.0 is supported.
            seed (int): Initial random seed; successive samples use seed + index.

        Returns:
            numpy.ndarray: Generated uint8 BGR images of shape ``(N, H, W, 3)``.
            The height and width depend on background preprocessing and may
            differ from the input dimensions.
        """
        background, foreground = self.inputs_preprocess(
            background_image, foreground_image, bbox, foreground_mask
        )
        return self.infer_prepared(
            background, foreground, prompt=prompt, num_samples=num_samples,
            sample_steps=sample_steps, guidance_scale=guidance_scale, seed=seed
        )

    def infer_prepared(self, background_bbox, white_foreground, *,
                       prompt=None, num_samples=1, sample_steps=20,
                       guidance_scale=1.0, seed=2511):
        """Evaluate already prepared condition images without rebuilding them."""
        import torch

        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if sample_steps != 20:
            raise ValueError("QwenEditModel supports 20 inference steps")
        if guidance_scale != 1.0:
            raise ValueError("Qwen-Image-2.1 was evaluated with guidance_scale=1.0")
        background = _bucket(_rgb(background_bbox))[0]
        foreground = _bucket(_rgb(white_foreground))[0]
        results = []
        for index in range(num_samples):
            generator = torch.Generator(device=self.pipeline._execution_device).manual_seed(seed + index)
            output = self.pipeline(
                prompt=prompt if prompt is not None else PROMPTS[self.model_type],
                image=[background, foreground],
                height=background.height,
                width=background.width,
                num_inference_steps=sample_steps,
                true_cfg_scale=guidance_scale,
                generator=generator,
                use_kv_cache=True,
            ).images[0]
            results.append(np.ascontiguousarray(np.asarray(output.convert("RGB"))[:, :, ::-1]))
        return np.stack(results)
