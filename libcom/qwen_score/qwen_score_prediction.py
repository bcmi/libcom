import os

import numpy as np
import torch
from PIL import Image

from libcom.utils.environment import check_gpu_device
from libcom.utils.process_image import opencv_to_pil, read_image_pil

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_dir = os.environ.get('LIBCOM_MODEL_DIR', cur_dir)

# --------------------------------------------------------------------------- #
# 权重位置
#   - 基座：优先 kwargs['base_model'] > LIBCOM_QWEN_BASE_MODEL > <model_dir>/pretrained_models/ >
#             <LIBCOM_QWEN_CKPT_DIR>/Qwen3-VL-8B-Instruct > HuggingFace id
#   - LoRA：优先 kwargs['lora_path'] > LIBCOM_QWEN_LORA_DIR/<name> >
#             <model_dir>/pretrained_models/<name> > <LIBCOM_QWEN_CKPT_DIR>/<name>
# --------------------------------------------------------------------------- #
QWEN_BASE_DIRNAME = 'Qwen3-VL-8B-Instruct'
QWEN_BASE_HF_ID   = 'Qwen/Qwen3-VL-8B-Instruct'
DEFAULT_CKPT_ROOT = os.environ.get('LIBCOM_QWEN_CKPT_DIR',
                                   '/data/chenhaoyu/vlm-eval-project/checkpoints')

PROMPTS = {
    'OPA': (
        "Task: OPA (object placement assessment).\n"
        "The first image is a composite image and the second image is the binary mask "
        "that highlights the foreground object in the first image. "
        "Judge whether the placement of the foreground object is reasonable, i.e. "
        "whether its scale and location are plausible in the scene. "
        "Answer with exactly one digit: 1 if the placement is reasonable, "
        "0 if it is unreasonable."
    ),
    'FOS': (
        "Task: FOS (foreground object search).\n"
        "The first image is a composite image and the second image is the binary mask "
        "that highlights the foreground object in the first image. "
        "Judge whether the perspective angle of the foreground object matches that of "
        "the background, and whether the semantic context of the foreground object is "
        "reasonable in the scene. "
        "Answer with exactly one digit: 1 if it is reasonable, 0 if it is not."
    ),
    'Harmony': (
        "Task: Harmony (illumination consistency).\n"
        "The first image is a composite image and the second image is the binary mask "
        "that highlights the foreground object in the first image. "
        "Judge whether the foreground object is harmonious with the background, i.e. "
        "whether its illumination, color tone and shadows are consistent with the scene. "
        "Answer with exactly one digit: 1 if it is harmonious, 0 if it is not."
    ),
}

# model_type -> (任务名, 默认 LoRA 目录名, 默认 max_pixels)
#   三个子任务统一使用 802816（与 highres adapter 的训练口径一致），
#   避免不同任务之间因为输入分辨率不同而出现分数不可比的问题。
MODEL_TYPES = {
    # OPA 用 highres adapter：它就是在 802816 下训练的（val AUC 0.9314 / Acc 0.8475），
    # 与其余两个任务的分辨率口径一致；旧 adapter `lora_opa_qwen3_4k_new` 是 401408 训的。
    'QwenVL-OPA':     ('OPA',     'lora_opa_qwen3_4k_highres', 802816),
    'QwenVL-FOS':     ('FOS',     'lora_fos_qwen3_4k_highres', 802816),
    'QwenVL-Harmony': ('Harmony', 'lora_harmony_highres',      802816),
}
model_set = list(MODEL_TYPES)

# 各任务的默认 TTA 配置（来自验证集实跑结论）：
#   OPA / FOS：原图 + 水平翻转 + 1.1x 放大，算术平均（3 视图）
#   Harmony  ：原图 + 1.1x + 1.2x 放大（不翻转），取中位数（3 视图）
#     光照任务里翻转会改变光照方向，属于有符号扰动；而原图是最弱的一路，
#     中位数能把单一视图的离群判断否决掉，实测 Acc 0.7309（vs 单视图 0.6951）。
TTA_CONFIG = {
    'QwenVL-OPA':     {'views': ('orig', 'hflip', 'zoom'), 'zooms': (1.1,), 'agg': 'mean'},
    'QwenVL-FOS':     {'views': ('orig', 'hflip', 'zoom'), 'zooms': (1.1,), 'agg': 'mean'},
    'QwenVL-Harmony': {'views': ('orig', 'zoom'), 'zooms': (1.1, 1.2), 'agg': 'median'},
}


def _zoom_view(img, zoom):
    '''中心裁剪 1/zoom 区域后缩放回原尺寸（zoom>1 即放大主体）。'''
    w, h = img.size
    cw, ch = max(1, int(round(w / zoom))), max(1, int(round(h / zoom)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)


def build_tta_views(image, mask, views, zooms):
    '''
    生成 TTA 视图列表 [(image, mask), ...]。
    image 与 mask 必须做完全相同的几何变换，否则两图错位、分数无意义。
    '''
    out = []
    for view in views:
        if view == 'orig':
            out.append((image, mask))
        elif view == 'hflip':
            out.append((image.transpose(Image.FLIP_LEFT_RIGHT),
                        mask.transpose(Image.FLIP_LEFT_RIGHT)))
        elif view == 'zoom':
            for z in zooms:
                out.append((_zoom_view(image, z), _zoom_view(mask, z)))
        else:
            raise ValueError(f'unknown tta view: {view}')
    return out


def _read_mask_rgb(mask_input):
    '''
    训练时 mask 是按 RGB 三通道读入的（Image.open(...).convert('RGB')），
    这里必须完全一致：先转灰度再转 RGB 会改变软边 mask 的像素值，导致分数偏移。
    '''
    if isinstance(mask_input, str):
        assert os.path.exists(mask_input), mask_input
        return Image.open(mask_input).convert('RGB')
    if isinstance(mask_input, np.ndarray):
        return opencv_to_pil(mask_input).convert('RGB')
    return mask_input.convert('RGB')


def _pick_existing(candidates):
    '''返回第一个已存在的路径，都不存在则返回 None。'''
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def resolve_base_model(base_model=None):
    if base_model:
        return base_model
    found = _pick_existing([
        os.environ.get('LIBCOM_QWEN_BASE_MODEL'),
        os.path.join(model_dir, 'pretrained_models', QWEN_BASE_DIRNAME),
        os.path.join(DEFAULT_CKPT_ROOT, QWEN_BASE_DIRNAME),
    ])
    # 本地找不到时回退到 HuggingFace id（首次会自动下载，约 16GB）
    return found or QWEN_BASE_HF_ID


def resolve_lora_path(lora_name, lora_path=None, lora_root=None):
    if lora_path:
        assert os.path.exists(lora_path), f'LoRA path not found: {lora_path}'
        return lora_path
    roots = [lora_root, os.environ.get('LIBCOM_QWEN_LORA_DIR'),
             os.path.join(model_dir, 'pretrained_models'), DEFAULT_CKPT_ROOT]
    found = _pick_existing([os.path.join(r, lora_name) for r in roots if r])
    if found is None:
        # Fall back to the libcom pretrained-model downloader
        # (expects <lora_name>.zip in the BCMIZB/Libcom_pretrained_models repo).
        try:
            from libcom.utils.model_download import download_entire_folder
            candidate = os.path.join(model_dir, 'pretrained_models', lora_name)
            found = download_entire_folder(candidate)
        except Exception:
            found = None
    if found is None:
        raise FileNotFoundError(
            f'Cannot find LoRA adapter "{lora_name}". Tried roots: '
            f'{[r for r in roots if r]}. '
            'Set LIBCOM_QWEN_LORA_DIR / LIBCOM_QWEN_CKPT_DIR, or pass lora_path=...'
        )
    return found


class QwenScoreModel:
    """
    Qwen3-VL based score prediction model for composition quality assessment.

    Three task options are supported, each backed by its own LoRA adapter:

    ==================  =========  ===============================  ==================
    model_type          task       default LoRA directory           default max_pixels
    ==================  =========  ===============================  ==================
    ``QwenVL-OPA``      OPA        ``lora_opa_qwen3_4k_highres``    802816
    ``QwenVL-FOS``      FOS        ``lora_fos_qwen3_4k_highres``    802816
    ``QwenVL-Harmony``  Harmony    ``lora_harmony_highres``         802816
    ==================  =========  ===============================  ==================

    The score is P('1') of the answer token: the last position of the prompt is fed
    through the LM head, and the logits of the digits '0'/'1' are softmax-normalized.
    This is the same scoring protocol used during training / evaluation, so the score
    ranges from 0 to 1 and is directly comparable to ``OPAScoreModel`` etc.

    Test-time augmentation is OFF by default (``enable_tta=False``), so scoring uses
    one forward pass. Enable it with ``enable_tta=True`` to use the multi-view TTA
    used for the reported benchmark numbers. Each task uses the view set that won
    on the validation set:

    - ``QwenVL-OPA`` / ``QwenVL-FOS``: orig + horizontal flip + 1.1x zoom, **mean**
    - ``QwenVL-Harmony``: orig + 1.1x zoom + 1.2x zoom, **median**
      (flipping rotates the light direction, so it is dropped for the illumination task;
      the median also vetoes the weakest single view, which is the original image)

    TTA costs 3 forward passes. Single-view inference is faster, but validation
    accuracy can be lower: e.g. Harmony 0.6951 vs 0.7309 Acc.

    Args:
        device (str | torch.device): gpu id
        model_type (str): one of ``QwenVL-OPA`` / ``QwenVL-FOS`` / ``QwenVL-Harmony``
        kwargs (dict): other parameters for building model

            - ``base_model`` (str): base VLM path or HuggingFace id.
              Default: ``$LIBCOM_QWEN_BASE_MODEL`` or ``<model_dir>/pretrained_models/Qwen3-VL-8B-Instruct``.
            - ``lora_path`` (str): explicit LoRA adapter directory (overrides model_type default).
            - ``lora_root`` (str): directory that contains the LoRA sub-directories.
            - ``max_pixels`` (int): max pixels per input image. Default follows the
              adapter: 802816 for all three tasks (OPA / FOS / Harmony), so that the
              three scores share the same input-resolution setting. Override only if you
              know the matching training setting.
            - ``enable_tta`` (bool): whether to apply multi-view TTA, default False.
              Set True to enable multi-view inference.
            - ``tta_agg`` (str): override the aggregation of TTA views, ``'mean'`` or
              ``'median'``. Default follows the task (mean for OPA/FOS, median for Harmony).
            - ``device_map`` (str): passed to ``from_pretrained`` when set (e.g. ``'auto'``).
            - ``use_cache`` (bool): default True (inference only).

    Examples:
        >>> from libcom import QwenScoreModel
        >>> from libcom import get_composite_image
        >>> import cv2
        >>> net = QwenScoreModel(device=0, model_type='QwenVL-OPA')
        >>> bg_img, fg_img, fg_mask = 'source/background/17.jpg', 'source/foreground/17.jpg', 'source/foreground_mask/17.png'
        >>> bbox = [475, 697, 1275, 1401]
        >>> comp, comp_mask = get_composite_image(fg_img, fg_mask, bg_img, bbox)
        >>> score = net(comp, comp_mask)
        >>> print(f'qwen opa score: {score:.4f}')

        >>> # Harmony adapter with the high-resolution setting used in training
        >>> net_h = QwenScoreModel(device=0, model_type='QwenVL-Harmony', max_pixels=802816)
        >>> harmony_score = net_h(comp, comp_mask)

    """
    def __init__(self, device=0, model_type='QwenVL-OPA', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        task, lora_name, default_pixels = MODEL_TYPES[model_type]
        self.task_name = task
        self.prompt = PROMPTS[task]
        self.max_pixels = int(kwargs.get('max_pixels', default_pixels))
        self.device = check_gpu_device(device)
        self.enable_tta = bool(kwargs.get('enable_tta', False))
        self.tta_config = dict(TTA_CONFIG[model_type])
        tta_agg = kwargs.get('tta_agg')
        if tta_agg is not None:
            assert tta_agg in ('mean', 'median'), "tta_agg must be 'mean' or 'median'"
            self.tta_config['agg'] = tta_agg

        base_model = resolve_base_model(kwargs.get('base_model'))
        lora_path = resolve_lora_path(lora_name, kwargs.get('lora_path'), kwargs.get('lora_root'))
        self.base_model_path = base_model
        self.lora_path = lora_path
        print(f'[QwenScoreModel] task={task}, base={base_model}, lora={lora_path}')
        self.build_processor(base_model)
        self.build_pretrained_model(base_model, lora_path, kwargs)

    def build_processor(self, base_model):
        from transformers import AutoProcessor
        kwargs = {'use_fast': True}
        if self.max_pixels > 0:
            # use_fast=True 时 image_processor 只读 size（longest_edge/shortest_edge），
            # max_pixels 仅对 slow 版本生效，因此两个都传。
            kwargs['max_pixels'] = self.max_pixels
            kwargs['size'] = {'longest_edge': self.max_pixels, 'shortest_edge': 65536}
        self.processor = AutoProcessor.from_pretrained(base_model, **kwargs)

    def build_pretrained_model(self, base_model, lora_path, kwargs):
        from peft import PeftModel
        from transformers import Qwen3VLForConditionalGeneration

        load_kwargs = {'torch_dtype': torch.bfloat16}
        device_map = kwargs.get('device_map')
        if device_map:
            load_kwargs['device_map'] = device_map
        model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **load_kwargs)
        model.config.use_cache = bool(kwargs.get('use_cache', True))
        model = PeftModel.from_pretrained(model, lora_path)
        if device_map:
            self.model = model.eval()
        else:
            self.model = model.to(self.device).eval()

        tokenizer = self.processor.tokenizer
        self.id_0 = tokenizer.encode('0', add_special_tokens=False)[-1]
        self.id_1 = tokenizer.encode('1', add_special_tokens=False)[-1]

    def inputs_preprocess(self, composite_image, composite_mask):
        image = read_image_pil(composite_image)
        mask = _read_mask_rgb(composite_mask)  # 与训练时一致：mask 按 RGB 喂入
        return self.encode_images(image, mask)

    def encode_images(self, image, mask):
        '''Encode PIL images and return the processor outputs and prompt length.'''
        user_msg = {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'image', 'image': mask},
                {'type': 'text', 'text': self.prompt},
            ],
        }
        text = self.processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image, mask], return_tensors='pt')
        prompt_len = int(inputs['input_ids'].shape[1])
        return inputs, prompt_len

    def outputs_postprocess(self, logits):
        pair = torch.stack([logits[self.id_0], logits[self.id_1]])
        score = torch.softmax(pair, dim=-1)[1].cpu().item()
        return score

    @torch.no_grad()
    def score_single(self, image, mask):
        '''Score one image and mask with a single forward pass, without TTA.'''
        inputs, prompt_len = self.encode_images(image, mask)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0, prompt_len - 1, :].float()
        return self.outputs_postprocess(logits)

    @torch.no_grad()
    def __call__(self, composite_image, composite_mask):
        """
        Predicting the composition quality score with a Qwen3-VL + LoRA model.

        Args:
            composite_image (str | numpy.ndarray | PIL.Image): The path to composite image,
                the composite image in ndarray form, or a PIL image.
            composite_mask (str | numpy.ndarray | PIL.Image): Mask of composite image which
                indicates the foreground object region in the composite image.

        Returns:
            score (float): Predicted score ranges from 0 to 1, where a larger score means
            the composition is more likely to be judged as reasonable by the corresponding task.

        Note:
            TTA is disabled by default, so scoring uses one forward pass. Set
            ``enable_tta=True`` to run three views and aggregate their scores.
            OPA and FOS average the original, horizontally flipped, and 1.1x
            zoomed views. Harmony takes the median of the original, 1.1x,
            and 1.2x zoomed views.

        """
        image = read_image_pil(composite_image)
        mask = _read_mask_rgb(composite_mask)
        if not self.enable_tta:
            return self.score_single(image, mask)

        cfg = self.tta_config
        views = build_tta_views(image, mask, cfg['views'], cfg['zooms'])
        scores = [self.score_single(v_image, v_mask) for v_image, v_mask in views]
        if cfg['agg'] == 'median':
            return float(np.median(scores))
        return float(np.mean(scores))
