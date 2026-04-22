from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Literal

import os

import cv2
import numpy as np

from libcom.utils.model_download import download_pretrained_model, download_entire_folder
from .source.insertanything_infer import InsertAnythingModel, insertanything_infer, run_insertanything
from .source.objectstitch_infer import (
    ObjectStitchConfig,
    run_objectstitch_single_image,
    run_objectstitch_single_image_from_images,
    load_objectstitch_model_and_sampler,
    run_objectstitch_single_image_from_images_cached,
)
from .source.sam_on_objectstitch import (
    SamOnObjectStitchConfig,
    build_sam_predictor,
    run_sam_on_objectstitch,
    run_sam_on_objectstitch_with_predictor,
)
from .source.utils import load_bbox_txt, make_rect_mask_from_bbox


@dataclass
class OSInsertConfig:
    model_dir: Path
    device: str = "cuda:0"
    objectstitch_ckpt_path: Path | None = None
    objectstitch_config_path: Path | None = None
    objectstitch_clip_dir: Path | None = None
    sam_checkpoint: Path | None = None
    flux_fill_path: Path | None = None
    flux_redux_path: Path | None = None
    ia_lora_path: Path | None = None

class OSInsertModel:
    """
    High-level OSInsert interface.

    This model provides a unified interface for object insertion with two modes:
    conservative and aggressive. It internally combines multiple sub-models such
    as InsertAnything, ObjectStitch, and SAM.

    Modes
    -----
    - ``aggressive``:
        ObjectStitch + SAM + InsertAnything pipeline.
        Suitable for more complex and flexible compositions.

    - ``conservative``:
        Directly uses background + bbox to generate mask,
        then performs insertion via InsertAnything.
        Faster and more stable.

    Args:
        device (str): Device to run the model on (e.g., "cuda:0", "cpu").
        model_dir (str | Path | None): Root directory of all model checkpoints.
        eager_aggressive_init (bool): 
            If True, preload ObjectStitch and SAM models at initialization.
            Otherwise, they will be lazily loaded when first used.
        objectstitch_ckpt_path (str | Path | None):
            Path to ObjectStitch checkpoint.
        objectstitch_config_path (str | Path | None):
            Path to ObjectStitch config file.
        objectstitch_clip_dir (str | Path | None):
            Path to CLIP model directory used by ObjectStitch.
        sam_checkpoint (str | Path | None):
            Path to SAM (Segment Anything Model) checkpoint.
        flux_fill_path (str | Path | None):
            Path to Flux Fill model directory.
        flux_redux_path (str | Path | None):
            Path to Flux Redux model directory.
        ia_lora_path (str | Path | None):
            Path to LoRA weights for InsertAnything.

    Notes
    -----
    - InsertAnything is initialized during class construction.
    - ObjectStitch and SAM are lazily initialized (unless
      ``eager_aggressive_init=True``), and then cached for reuse.
    - Conservative mode does not require ObjectStitch.

    Examples:
        >>> import cv2
        >>> from libcom import OSInsertModel

        >>> model = OSInsertModel(
        >>>     device="cuda:0"
        >>> )

        >>> bg = cv2.imread("tests/osinsert/background/Demo_0.png")
        >>> fg = cv2.imread("tests/osinsert/foreground/Demo_0.png")
        >>> fg_mask = cv2.imread(
        >>>     "tests/osinsert/foreground_mask/Demo_0.png",
        >>>     cv2.IMREAD_GRAYSCALE
        >>> )

        >>> bbox = (175, 184, 363, 372)

        >>> result = model.infer_images(
        >>>     background=bg,
        >>>     foreground=fg,
        >>>     foreground_mask=fg_mask,
        >>>     bbox_xyxy=bbox,
        >>>     mode="conservative",   # or "aggressive"
        >>>     verbose=False,
        >>>     seed=123,
        >>>     strength=1.0,
        >>>     split_ratio=0.33,
        >>>     save_path="result_dir/conservative",
        >>> )

    Expected result:
        The foreground object is inserted into the background image
        at the specified bounding box, with realistic blending.

    .. image:: _static/image/os_insert_result.jpg

    """
    def __init__(
        self,
        device: str = "cuda:0",
        model_dir: str | Path | None = None,
        *,
        eager_aggressive_init: bool = False,
        objectstitch_ckpt_path: str | Path | None = None,
        objectstitch_config_path: str | Path | None = None,
        objectstitch_clip_dir: str | Path | None = None,
        sam_checkpoint: str | Path | None = None,
        flux_fill_path: str | Path | None = None,
        flux_redux_path: str | Path | None = None,
        ia_lora_path: str | Path | None = None,
    ) -> None:
        # 设置模型目录
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.environ.get('LIBCOM_MODEL_DIR', cur_dir)
        
        # 设置默认路径
        if model_dir is None:
            model_dir = self.model_dir
        
        # 下载必要的模型权重
        self._download_model_weights()
        
        self.config = OSInsertConfig(
            model_dir=Path(model_dir),
            device=device,
            objectstitch_ckpt_path=Path(objectstitch_ckpt_path) if objectstitch_ckpt_path is not None else None,
            objectstitch_config_path=Path(objectstitch_config_path) if objectstitch_config_path is not None else None,
            objectstitch_clip_dir=Path(objectstitch_clip_dir) if objectstitch_clip_dir is not None else None,
            sam_checkpoint=Path(sam_checkpoint) if sam_checkpoint is not None else None,
            flux_fill_path=Path(flux_fill_path) if flux_fill_path is not None else None,
            flux_redux_path=Path(flux_redux_path) if flux_redux_path is not None else None,
            ia_lora_path=Path(ia_lora_path) if ia_lora_path is not None else None,
        )

        self._ia_net = InsertAnythingModel(
            model_dir=self.config.model_dir,
            flux_fill_path=self.config.flux_fill_path,
            flux_redux_path=self.config.flux_redux_path,
            ia_lora_path=self.config.ia_lora_path,
            device=self.config.device,
        )

        self._os_model = None
        self._os_sampler = None
        self._sam_predictor = None

        if eager_aggressive_init:
            self._get_objectstitch_model_and_sampler()
            self._get_sam_predictor()
    
    def _download_model_weights(self):
        """下载所有必要的模型权重"""
        # ObjectStitch 模型权重
        objectstitch_ckpt_path = os.path.join(self.model_dir, 'pretrained_models', 'ObjectStitch.pth')
        download_pretrained_model(objectstitch_ckpt_path)
        
        # ObjectStitch 配置文件 - 使用 source 目录中的本地文件
        objectstitch_config_path = os.path.join(os.path.dirname(__file__), 'source', 'v1.yaml')
        if not os.path.exists(objectstitch_config_path):
            print(f'v1.yaml not found at {objectstitch_config_path}')
            raise Exception('v1.yaml not found')
        print('Using local v1.yaml configuration file from source directory')
        
        # CLIP 模型
        clip_dir = os.path.join(self.model_dir, 'shared_pretrained_models', 'openai-clip-vit-large-patch14')
        download_entire_folder(clip_dir)
        
        # SAM 模型权重
        sam_ckpt_path = os.path.join(self.model_dir, 'pretrained_models', 'sam_vit_h_4b8939.pth')
        download_pretrained_model(sam_ckpt_path)
        
        # InsertAnything LoRA 权重
        ia_lora_path = os.path.join(self.model_dir, 'pretrained_models', 'insert_anything_lora.safetensors')
        download_pretrained_model(ia_lora_path)

    def _get_objectstitch_model_and_sampler(self):
        if self._os_model is not None and self._os_sampler is not None:
            return self._os_model, self._os_sampler

        objectstitch_ckpt = self.config.objectstitch_ckpt_path
        if objectstitch_ckpt is None:
            objectstitch_ckpt = os.path.join(self.model_dir, 'pretrained_models', 'ObjectStitch.pth')

        objectstitch_cfg = self.config.objectstitch_config_path
        if objectstitch_cfg is None:
            objectstitch_cfg = os.path.join(os.path.dirname(__file__), 'source', 'v1.yaml')

        clip_dir = self.config.objectstitch_clip_dir
        if clip_dir is None:
            clip_dir = os.path.join(self.model_dir, 'shared_pretrained_models', 'openai-clip-vit-large-patch14')

        os_cfg = ObjectStitchConfig(
            ckpt_path=objectstitch_ckpt,
            config_path=objectstitch_cfg,
            clip_dir=clip_dir,
            device=self.config.device,
        )
        self._os_model, self._os_sampler = load_objectstitch_model_and_sampler(config=os_cfg)
        return self._os_model, self._os_sampler

    def _get_sam_predictor(self):
        if self._sam_predictor is not None:
            return self._sam_predictor

        sam_ckpt = self.config.sam_checkpoint
        if sam_ckpt is None:
            sam_ckpt = os.path.join(self.model_dir, 'pretrained_models', 'sam_vit_h_4b8939.pth')

        sam_cfg = SamOnObjectStitchConfig(
            sam_checkpoint=sam_ckpt,
            device=self.config.device,
        )
        self._sam_predictor = build_sam_predictor(config=sam_cfg)
        return self._sam_predictor

    def __call__(
        self,
        background_path: str | Path,
        foreground_path: str | Path,
        foreground_mask_path: str | Path,
        bbox: list[int],
        result_dir: str | Path,
        mode: Literal["aggressive", "conservative"] = "conservative",
        cleanup_intermediate: bool = True,
        verbose: bool = False,
        seed: int = 123,
        strength: float = 1.0,
        split_ratio: float = 0.5,
    ) -> np.ndarray | None:
        """Run a single OSInsert inference.

        Parameters
        ----------
        background_path:
            Path to the background image.
        foreground_path:
            Path to the foreground image used as the InsertAnything reference
            image.
        foreground_mask_path:
            Binary mask for the foreground image.
        bbox:
            List containing ``[x1, y1, x2, y2]``, specifying the
            insertion region on the background image.
        result_dir:
            Directory where the final composed image will be written.
        mode:
            - ``"conservative"``: background + bbox -> mask -> InsertAnything.
            - ``"aggressive"``: ObjectStitch + SAM -> combined source/mask ->
              InsertAnything.
        cleanup_intermediate:
            Deprecated. Present for backward compatibility.
        verbose:
            If True, save intermediate artifacts into ``result_dir/intermediates``.
            Default False (do not save intermediates).
        seed:
            Random seed for InsertAnything.
        strength:
            InsertAnything strength parameter.

        Returns:
            Generated composited image (np.array): The inserted result.
        """

        if mode not in {"aggressive", "conservative"}:
            raise ValueError(f"Unsupported mode: {mode}")

        # ------------------------------------------------------------------
        # Path normalization and output directory.
        # ------------------------------------------------------------------
        background_path = Path(background_path)
        foreground_path = Path(foreground_path)
        foreground_mask_path = Path(foreground_mask_path)
        result_dir = Path(result_dir)

        os.makedirs(result_dir, exist_ok=True)

        intermediates_dir = result_dir / "intermediates"
        if verbose:
            os.makedirs(intermediates_dir, exist_ok=True)

        # InsertAnything expects a list of seeds.
        seeds = [seed]

        # Load background once; used by both modes.
        bg = cv2.imread(str(background_path))
        if bg is None:
            raise FileNotFoundError(background_path)
        h, w = bg.shape[:2]

        fg = cv2.imread(str(foreground_path))
        if fg is None:
            raise FileNotFoundError(foreground_path)
        fg_mask = cv2.imread(str(foreground_mask_path))
        if fg_mask is None:
            raise FileNotFoundError(foreground_mask_path)
        if fg_mask.ndim == 3:
            fg_mask = fg_mask[:, :, 0]

        # 直接使用传入的 bbox 列表
        assert len(bbox) == 4, f"bbox should be a list of 4 integers: [x1, y1, x2, y2], got {bbox}"

        # ------------------------------------------------------------------
        # Aggressive mode: ObjectStitch + SAM + InsertAnything.
        # ------------------------------------------------------------------
        if mode == "aggressive":
            # 1) ObjectStitch coarse composite (cached weights).
            os_model, os_sampler = self._get_objectstitch_model_and_sampler()
            os_rgb = run_objectstitch_single_image_from_images_cached(
                background=cv2.cvtColor(bg, cv2.COLOR_BGR2RGB),
                foreground=cv2.cvtColor(fg, cv2.COLOR_BGR2RGB),
                foreground_mask=fg_mask,
                bbox_xyxy=tuple(bbox),
                model=os_model,
                sampler=os_sampler,
                device=self.config.device,
                seed=seed,
            )

            # 2) SAM mask on top of ObjectStitch composite (in-memory).
            predictor = self._get_sam_predictor()
            sam_mask = run_sam_on_objectstitch_with_predictor(
                predictor=predictor,
                os_image=cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR),
                bg_shape_hw=(h, w),
                bbox_xyxy_bg=tuple(bbox),
            )

            # 3) Construct InsertAnything source & mask following
            #    exp/run_insertanything_strength_sweep_dispatch.py::make_source_and_mask
            bg_bgr = bg  # already read above
            os_bgr = cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR)

            hh, ww = bg_bgr.shape[:2]
            os_bgr = cv2.resize(os_bgr, (ww, hh), interpolation=cv2.INTER_AREA)
            if sam_mask.shape[:2] != (hh, ww):
                sam_mask = cv2.resize(sam_mask, (ww, hh), interpolation=cv2.INTER_NEAREST)

            m = (sam_mask > 127).astype(np.float32)
            m3 = np.stack([m, m, m], axis=-1)

            src_bgr = bg_bgr.astype(np.float32) * (1.0 - m3) + os_bgr.astype(np.float32) * m3
            src_bgr = np.clip(src_bgr, 0, 255).astype(np.uint8)

            # BBox-based mask for InsertAnything second-stage (bbox mask)
            bbox_mask = make_rect_mask_from_bbox(h, w, bbox)

            if verbose:
                cv2.imwrite(str(intermediates_dir / "objectstitch_coarse_rgb.png"), cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(intermediates_dir / "sam_mask.png"), sam_mask)
                cv2.imwrite(str(intermediates_dir / "blended_source.png"), src_bgr)
                cv2.imwrite(str(intermediates_dir / "bbox_mask.png"), bbox_mask)

            # 4) InsertAnything refinement using blended source, bbox mask
            #    (for the second half of denoising) and SAM mask
            #    (for the first half, wired via sam_mask_path).
            result = insertanything_infer(
                source_image=cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB),
                mask_image=bbox_mask,
                ref_image=cv2.cvtColor(fg, cv2.COLOR_BGR2RGB),
                ref_mask=fg_mask,
                sam_mask=sam_mask,
                seeds=seeds,
                strength=strength,
                split_ratio=split_ratio,
                save_path=str(result_dir),
                filename_suffix="",
                net=self._ia_net,
                return_image=True,
            )

            return result

        # ------------------------------------------------------------------
        # Conservative mode: background + bbox -> mask -> InsertAnything.
        # ------------------------------------------------------------------
        mask = make_rect_mask_from_bbox(h, w, bbox)

        if verbose:
            cv2.imwrite(str(intermediates_dir / "bbox_mask.png"), mask)

        result = insertanything_infer(
            source_image=cv2.cvtColor(bg, cv2.COLOR_BGR2RGB),
            mask_image=mask,
            ref_image=cv2.cvtColor(fg, cv2.COLOR_BGR2RGB),
            ref_mask=fg_mask,
            seeds=seeds,
            strength=strength,
            split_ratio=split_ratio,
            save_path=str(result_dir),
            filename_suffix="",
            net=self._ia_net,
            return_image=True,
        )

        return result

    def infer_images(
        self,
        *,
        background: np.ndarray,
        foreground: np.ndarray,
        foreground_mask: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        mode: Literal["aggressive", "conservative"] = "conservative",
        verbose: bool = False,
        seed: int = 123,
        strength: float = 1.0,
        split_ratio: float = 0.5,
        save_path: str | Path | None = None,
        filename_suffix: str = "",
    ) -> np.ndarray | None:
        if background.ndim != 3:
            raise ValueError("background must be HxWx3")
        h, w = background.shape[:2]

        out_dir = str(save_path) if save_path is not None else "./result"

        if verbose and save_path is not None:
            inter_dir = Path(save_path) / "intermediates"
            os.makedirs(inter_dir, exist_ok=True)

        # InsertAnything expects a list of seeds.
        seeds = [seed]

        if mode == "aggressive":
            os_model, os_sampler = self._get_objectstitch_model_and_sampler()
            os_rgb = run_objectstitch_single_image_from_images_cached(
                background=background[:, :, ::-1],
                foreground=foreground[:, :, ::-1],
                foreground_mask=foreground_mask,
                bbox_xyxy=tuple(bbox_xyxy),
                model=os_model,
                sampler=os_sampler,
                device=self.config.device,
                seed=seed,
            )

            predictor = self._get_sam_predictor()
            sam_mask = run_sam_on_objectstitch_with_predictor(
                predictor=predictor,
                os_image=cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR),
                bg_shape_hw=(h, w),
                bbox_xyxy_bg=tuple(bbox_xyxy),
            )

            bg_bgr = background
            os_bgr = cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR)
            hh, ww = bg_bgr.shape[:2]
            os_bgr = cv2.resize(os_bgr, (ww, hh), interpolation=cv2.INTER_AREA)
            if sam_mask.shape[:2] != (hh, ww):
                sam_mask = cv2.resize(sam_mask, (ww, hh), interpolation=cv2.INTER_NEAREST)

            m = (sam_mask > 127).astype(np.float32)
            m3 = np.stack([m, m, m], axis=-1)
            src_bgr = bg_bgr.astype(np.float32) * (1.0 - m3) + os_bgr.astype(np.float32) * m3
            src_bgr = np.clip(src_bgr, 0, 255).astype(np.uint8)

            bbox_mask = make_rect_mask_from_bbox(h, w, list(bbox_xyxy))

            if verbose and save_path is not None:
                inter_dir = Path(save_path) / "intermediates"
                prefix = f"{filename_suffix}_" if filename_suffix else ""
                cv2.imwrite(str(inter_dir / f"{prefix}objectstitch_coarse_rgb.png"), cv2.cvtColor(os_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(inter_dir / f"{prefix}sam_mask.png"), sam_mask)
                cv2.imwrite(str(inter_dir / f"{prefix}blended_source.png"), src_bgr)
                cv2.imwrite(str(inter_dir / f"{prefix}bbox_mask.png"), bbox_mask)

            return insertanything_infer(
                source_image=cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB),
                mask_image=bbox_mask,
                ref_image=cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB),
                ref_mask=foreground_mask,
                sam_mask=sam_mask,
                seeds=seeds,
                strength=strength,
                split_ratio=split_ratio,
                save_path=out_dir,
                filename_suffix=filename_suffix,
                net=self._ia_net,
                return_image=True,
            )

        if mode != "conservative":
            raise ValueError(f"Unsupported mode: {mode}")

        mask = make_rect_mask_from_bbox(h, w, list(bbox_xyxy))

        if verbose and save_path is not None:
            inter_dir = Path(save_path) / "intermediates"
            prefix = f"{filename_suffix}_" if filename_suffix else ""
            cv2.imwrite(str(inter_dir / f"{prefix}bbox_mask.png"), mask)

        return insertanything_infer(
            source_image=cv2.cvtColor(background, cv2.COLOR_BGR2RGB),
            mask_image=mask,
            ref_image=cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB),
            ref_mask=foreground_mask,
            seeds=seeds,
            strength=strength,
            split_ratio=split_ratio,
            save_path=out_dir,
            filename_suffix=filename_suffix,
            net=self._ia_net,
            return_image=True,
        )
