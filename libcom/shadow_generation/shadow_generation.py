import argparse
import logging
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import os, cv2, torch, random
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from pytorch_lightning import seed_everything
from diffusers.utils import check_min_version, is_wandb_available
from torch.utils.data import Dataset
from .source.cldm.base_network import MaskCls, RegNetwork
from .source.PostProcessModel import PostProcess
from .source.cldm.model import load_state_dict
from .source.attention_processor import IPAttnProcessor, AttnProcessor

seed_everything(42)

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def print_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Memory] {prefix} Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

class TestDataset(Dataset):
    def __init__(self, shadowfree_img, object_mask, text_encoders, tokenizers, caption_column='text'):
        """
        shadowfree_img: str(path) 或 np.ndarray (H, W, 3)
        object_mask: str(path) 或 np.ndarray (H, W)
        """
        self.shadowfree_img = self._load_image(shadowfree_img)
        self.object_mask = self._load_mask(object_mask)
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.caption_column = caption_column

        # 预处理图片，生成 control input, target, bounding box, embedding 占位
        self.preprocess_data()
        self.precompute_embeddings()

    def _load_image(self, img_input):
        """自动处理路径或 np.ndarray 的彩色图片"""
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Image not found: {img_input}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, np.ndarray):
            img = img_input
        else:
            raise TypeError("shadowfree_img must be str or np.ndarray")
        img = cv2.resize(img, (512, 512))
        return img

    def _load_mask(self, mask_input):
        """自动处理路径或 np.ndarray 的灰度 mask"""
        if isinstance(mask_input, str):
            mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask not found: {mask_input}")
        elif isinstance(mask_input, np.ndarray):
            mask = mask_input
        else:
            raise TypeError("object_mask must be str or np.ndarray")
        mask = cv2.resize(mask, (512, 512))
        return mask

    def preprocess_data(self):
        # Bounding box 前景
        _, fg_instance_thresh = cv2.threshold(self.object_mask, 128, 255, cv2.THRESH_BINARY)
        contours_instance, _ = cv2.findContours(fg_instance_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        merged_contour_points_instance = np.concatenate(contours_instance)
        rect_instance = cv2.minAreaRect(merged_contour_points_instance)
        (x, y), (w, h), theta = rect_instance
        if w < h:
            w, h = h, w
            theta += 90
        self.fg_instance = np.array([x, y, w+1, h+1, theta]).astype(int)
        self.bbx_region = torch.zeros((512, 512), dtype=torch.float16)

        # Control input 和 target
        control_input = np.concatenate([self.shadowfree_img, self.object_mask[..., None]], axis=-1)
        self.control_input = control_input.astype(np.float16) / 255.0
        self.target = (self.shadowfree_img.astype(np.float16) / 127.5) - 1.0

        # Embedding 占位
        self.embeddings_placeholder = torch.zeros((64, 2048), dtype=torch.float16)
        self.prompt = 'foreground object with shadow'
        self.name = 'single_image'

    def precompute_embeddings(self):
        dummy_batch = {
            self.caption_column: [self.prompt],
            "__fake_image__": [None]
        }

        with torch.no_grad():
            embeddings = compute_embeddings(
                batch=dummy_batch,
                proportion_empty_prompts=0,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers
            )

        # 取第0个 batch 的结果
        self.prompt_embeds = embeddings["prompt_embeds"].cpu()[0]
        self.text_embeds = embeddings["text_embeds"].cpu()[0]
        self.time_ids = embeddings["time_ids"].cpu()[0]

    def __len__(self):
        return 1  # 单张图片

    def __getitem__(self, idx):
        return {
            "pixel_values": torch.tensor(self.target).permute(2, 0, 1).float(),
            "conditioning_pixel_values": torch.tensor(self.control_input).permute(2, 0, 1).float(),
            "bbx": self.bbx_region,
            "fg": self.fg_instance,
            "embeddings": self.embeddings_placeholder,
            "prompt_ids": self.prompt_embeds,
            "unet_added_conditions": {
                "text_embeds": self.text_embeds,
                "time_ids": self.time_ids
            },
            "name": self.name
        }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs, down_block_additional_residuals, mid_block_additional_residual, mask_embeddings):
        ip_tokens = self.image_proj_model(mask_embeddings)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, down_block_additional_residuals=down_block_additional_residuals, mid_block_additional_residual=mid_block_additional_residual, return_dict=False)[0]
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=768, extra_embeddings_dim=2048, clip_extra_context_tokens=64):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(extra_embeddings_dim, cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        extra_embeds = self.proj(image_embeds)
        extra_embeds = self.norm(extra_embeds)
        return extra_embeds

def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=False):
        original_size = (512, 512)
        target_size = (512, 512)
        crops_coords_top_left = (0, 0)
        prompt_batch = batch['text']
 
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds
        add_text_embeds = add_text_embeds
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")


    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=False):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def collate_fn(examples):
    pixel_values = torch.stack([torch.from_numpy(example["pixel_values"].transpose(2, 0, 1)) if isinstance(example["pixel_values"], np.ndarray) else example["pixel_values"].permute(0, 3, 1, 2) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([torch.from_numpy(example["conditioning_pixel_values"].transpose(2, 0, 1)) if isinstance(example["conditioning_pixel_values"], np.ndarray) else example["conditioning_pixel_values"].permute(0, 3, 1, 2) for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
    add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gc
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import logging
from libcom.utils.model_download import download_pretrained_model, download_entire_folder

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_dir = os.environ.get('LIBCOM_MODEL_DIR',cur_dir)

logger = logging.getLogger(__name__)

class ShadowGenerationModel:
    """
    Foreground Shadow generation model based on diffusion model.

    Args:
        device (str | torch.device): gpu id

    Examples:
        >>> from libcom import ShadowGenerationModel
        >>> from libcom.utils.process_image import make_image_grid
        >>> import cv2
        >>> net = ShadowGenerationModel()
        >>> comp_image1 = "../tests/shadow_generation/composite/1.png"
        >>> comp_mask1 = "../tests/shadow_generation/composite_mask/1.png"
        >>> preds = net(comp_image1, comp_mask1, number=5)
        >>> grid_img  = make_image_grid([comp_image1, comp_mask1] + preds)
        >>> cv2.imwrite('../docs/_static/image/shadow_generation_result1.jpg', grid_img)
        >>> comp_image2 = "../tests/shadow_generation/composite/2.png"
        >>> comp_mask2 = "../tests/shadow_generation/composite_mask/2.png"
        >>> preds = net(comp_image2, comp_mask2, number=5)
        >>> grid_img  = make_image_grid([comp_image2, comp_mask2] + preds)
        >>> cv2.imwrite('../docs/_static/image/shadow_generation_result2.jpg', grid_img)

    Expected result:

    .. image:: _static/image/shadow_generation_result1.jpg
        :scale: 21 %
    .. image:: _static/image/shadow_generation_result2.jpg
        :scale: 21 %

    """
    def __init__(self, device=0):
        self.args = parse_args()
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.float32 
        

        self.controlnet_model_path = os.path.join(model_dir, 'shadow_controlnet')
        self.ip_adapter_path = os.path.join(model_dir, 'pretrained_models', 'ip_adapter.ckpt')
        self.ppp_weight_path = os.path.join(model_dir, 'pretrained_models', 'Shadow_ppp.ckpt')
        self.cls_weight_path = os.path.join(model_dir, 'pretrained_models', 'Shadow_cls.pth')
        self.reg_weight_path = os.path.join(model_dir, 'pretrained_models', 'Shadow_reg.pth')
        self.cls_label_path = os.path.join(model_dir, 'pretrained_models', 'Shadow_cls_label.pkl')

        download_entire_folder(self.controlnet_model_path)
        download_pretrained_model(self.ip_adapter_path)
        download_pretrained_model(self.ppp_weight_path)
        download_pretrained_model(self.cls_weight_path)
        download_pretrained_model(self.reg_weight_path)
        download_pretrained_model(self.cls_label_path)
      
        # Load tokenizers
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.args.revision, use_fast=False
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=self.args.revision, use_fast=False
        )

        # Load text encoders
        text_encoder_cls_one = import_model_class_from_model_name_or_path(self.args.pretrained_model_name_or_path, self.args.revision)
        text_encoder_cls_two = import_model_class_from_model_name_or_path(self.args.pretrained_model_name_or_path, self.args.revision, subfolder="text_encoder_2")

        self.text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.args.revision, variant=self.args.variant
        ).to(self.device, dtype=self.weight_dtype)
        self.text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=self.args.revision, variant=self.args.variant
        ).to(self.device, dtype=self.weight_dtype)

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="scheduler")
        vae_path = self.args.pretrained_model_name_or_path if self.args.pretrained_vae_model_name_or_path is None else self.args.pretrained_vae_model_name_or_path
        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if self.args.pretrained_vae_model_name_or_path is None else None,
            revision=self.args.revision,
            variant=self.args.variant
        ).to(self.device, dtype=self.weight_dtype)

        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="unet", revision=self.args.revision, variant=self.args.variant
        ).to(self.device, dtype=self.weight_dtype)

        # MaskCls & RegNetwork
        self.mask_cls = MaskCls(num_classes=256).to(self.device, dtype=self.weight_dtype)
        if os.path.exists(self.cls_weight_path):
            self.mask_cls.load_state_dict(torch.load(self.cls_weight_path, weights_only=False)['net'])
            print('Loaded mask classification model.')

        self.bbx_reg = RegNetwork().to(self.device, dtype=self.weight_dtype)
        if os.path.exists(self.reg_weight_path):
            self.bbx_reg.load_state_dict(torch.load(self.reg_weight_path, weights_only=False)['net'])
            print('Loaded rotated bounding box regression model.')

        # ControlNet
        if self.controlnet_model_path:
            self.controlnet = ControlNetModel.from_pretrained(self.controlnet_model_path).to(self.device, dtype=self.weight_dtype)
        else:
            self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_channels=5).to(self.device, dtype=self.weight_dtype)

        # IP-Adapter
        num_tokens = 64
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            extra_embeddings_dim=2048,
            clip_extra_context_tokens=num_tokens,
        ).to(self.device, dtype=self.weight_dtype)

        # init adapter modules
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)

        adapter_modules = nn.ModuleList(self.unet.attn_processors.values())
        self.ip_adapter = IPAdapter(self.unet, self.image_proj_model, adapter_modules, self.ip_adapter_path)
        self.ip_adapter.to(self.device, dtype=self.weight_dtype)
        self.ip_adapter.requires_grad_(False)

        # Freeze all
        for model in [self.vae, self.unet, self.text_encoder_one, self.text_encoder_two, self.mask_cls, self.bbx_reg, self.controlnet]:
            model.requires_grad_(False)

        # Ensure all attention processors are on same device
        for proc in self.unet.attn_processors.values():
            for k, v in proc.state_dict().items():
                proc._parameters[k] = v.to(self.device, dtype=self.weight_dtype)
        
        self.post_model = PostProcess(infe_steps=50).cpu()
        self.post_model.load_state_dict(load_state_dict(self.ppp_weight_path, location='cuda'), strict=False)
        self.post_model = self.post_model.to(self.device)
        self.post_model.eval()

    def post_process(self, decoded, shadowfree_img, object_mask):
        """
        decoded: np.uint8 HWC [0-255] RGB
        shadowfree_img: np.uint8 HWC [0-255] RGB
        object_mask: np.uint8 HW [0-255]
        """
        width, height = 512, 512
        decoded_512 = cv2.resize(decoded, (width, height))
        shadowfree_img_512 = cv2.resize(shadowfree_img, (width, height))
        object_mask_512 = cv2.resize(object_mask, (width, height))

        comp_img_scaled = shadowfree_img_512.astype(np.float32)/127.5 - 1.0
        obj_mask = object_mask_512[:,:,np.newaxis].astype(np.float32)/255.0
        image_scaled = decoded_512.astype(np.float32)/127.5 - 1.0

        comp_img_scaled = torch.from_numpy(comp_img_scaled).float().unsqueeze(0).to(self.device)
        obj_mask = torch.from_numpy(obj_mask).float().unsqueeze(0).to(self.device)
        image_scaled = torch.from_numpy(image_scaled).float().unsqueeze(0).to(self.device)

        input_tensor = torch.concat([image_scaled, comp_img_scaled, obj_mask], dim=-1).permute(0,3,1,2)
        null_timesteps = torch.zeros(1, device=self.device)

        with torch.no_grad():
            output = self.post_model.post_process_net(input_tensor, timesteps=null_timesteps)
            output = output.permute(0,2,3,1)  # [B,H,W,C]

            pred_mask = torch.greater_equal(output[:,:,:,3],0).float()
            pred_mask = pred_mask.unsqueeze(-1)  # [B,H,W,1]

            adjusted_img = torch.clamp(image_scaled, -1.,1)
            adjusted_img = (adjusted_img + 1)/2.0 * 255

            comp_img_tensor = (comp_img_scaled +1)/2.0 * 255
            new_composite = adjusted_img * pred_mask + (1 - pred_mask) * comp_img_tensor
            new_composite = new_composite.squeeze(0).cpu().numpy().astype(np.uint8)

        new_composite_512 = cv2.resize(new_composite, (512,512))
        return new_composite_512

    def __call__(self, shadowfree_img, object_mask, number=5):
        """
        Generate shadow for foreground object.
        
        Args:
            shadowfree_img (str | numpy.ndarray): The path to composite image or composite image in ndarray form.
            object_mask (str | numpy.ndarray): The path to foreground object mask or foreground object mask in ndarray form.
            number (int): Number of images to be inferenced. default: 5.
        
        Returns: 
            generated_images (list): A list of images with generated foreground shadows. Each image is in ndarray form with a shape of 512x512x3

        """
        
        test_dataset = TestDataset(
            shadowfree_img=shadowfree_img,
            object_mask=object_mask,
            text_encoders=[self.text_encoder_one, self.text_encoder_two],
            tokenizers=[self.tokenizer_one, self.tokenizer_two],
            caption_column='text'
        )

        test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=self.args.dataloader_num_workers)

        preds = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
                batch_name = batch["name"]
                for repeat in range(number):
                    # Move all tensors to device & float32
                    pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
                    geometry_input = batch["conditioning_pixel_values"].to(self.device, dtype=self.weight_dtype)
                    bbx_mask = batch['bbx'].to(self.device, dtype=self.weight_dtype)
                    fg_instance_bbx = batch['fg'].to(self.device, dtype=self.weight_dtype)
                    added_conditions = {k: v.to(self.device, dtype=self.weight_dtype) for k,v in batch["unet_added_conditions"].items()}
                    prompt_ids = batch["prompt_ids"].to(self.device, dtype=self.weight_dtype)
                    mask_embeddings = batch['embeddings'].to(self.device, dtype=self.weight_dtype)

                    # Encode image
                    latents = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
                    latents = latents.to(self.device, dtype=self.weight_dtype)

                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    latents = noisy_latents

                    # Generate bounding boxes
                    pred_t = self.bbx_reg(geometry_input)
                    img_height, img_width = 512, 512
                    pred_bbx = pred_t.new(pred_t.shape)
                    pred_bbx[:,0] = pred_t[:,0] * fg_instance_bbx[:,2] + fg_instance_bbx[:,0]
                    pred_bbx[:,1] = pred_t[:,1] * fg_instance_bbx[:,3] + fg_instance_bbx[:,1]
                    pred_bbx[:,2] = fg_instance_bbx[:,2] * torch.exp(pred_t[:,2])
                    pred_bbx[:,3] = fg_instance_bbx[:,3] * torch.exp(pred_t[:,3])
                    pred_bbx[:,4] = pred_t[:,4] * 180 / np.pi + fg_instance_bbx[:,4]
                    for i in range(pred_bbx.shape[0]):
                        if pred_bbx[i,4] > 0:
                            temp = pred_bbx[i,2]
                            pred_bbx[i,2] = pred_bbx[i,3]
                            pred_bbx[i,3] = temp
                            pred_bbx[i,4] -= 90
                        x,y,w,h,theta = pred_bbx[i]
                        box = ((x.item(),y.item()),(w.item(),h.item()),theta.item())
                        box_points = cv2.boxPoints(box)
                        box_points = np.clip(box_points, 0, [img_width-1,img_height-1])
                        cv2.fillPoly(bbx_mask[i].cpu().numpy().astype(np.uint8), [np.int32(box_points)], 1)
                    bbx_region = bbx_mask.unsqueeze(1).to(self.device, dtype=self.weight_dtype)

                    controlnet_image = torch.cat((geometry_input, bbx_region), 1)

                    # Set scheduler timesteps
                    num_inference_steps = 50
                    self.noise_scheduler.set_timesteps(num_inference_steps)
                    self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(self.device)

                    # Denoising loop
                    for t in tqdm(self.noise_scheduler.timesteps, desc=f"Denoising {batch_name[0]}_{repeat}", leave=False):
                        # ControlNet
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            latents, t, encoder_hidden_states=prompt_ids, added_cond_kwargs=added_conditions,
                            controlnet_cond=controlnet_image, return_dict=False
                        )

                        # Mask embeddings adjustment
                        mask_label = self.mask_cls(geometry_input)
                        with open(self.cls_label_path, 'rb') as f:
                            centroid_dict = pickle.load(f)
                        _, top64_label = torch.topk(mask_label, 64, largest=True, sorted=False)
                        for i in range(mask_embeddings.shape[0]):
                            for j in range(top64_label.shape[1]):
                                label = top64_label[i,j].item()
                                if label in centroid_dict:
                                    mask_embeddings[i,j,:] += torch.tensor(centroid_dict[label], dtype=torch.float32, device=self.device)

                        # IP Adapter
                        model_pred = self.ip_adapter(
                            latents,
                            t,
                            encoder_hidden_states=prompt_ids,
                            added_cond_kwargs=added_conditions,
                            down_block_additional_residuals=[sample.to(self.device, dtype=self.weight_dtype) for sample in down_block_res_samples],
                            mid_block_additional_residual=mid_block_res_sample.to(self.device, dtype=self.weight_dtype),
                            mask_embeddings=mask_embeddings
                        )

                        # Update latents
                        latents = self.noise_scheduler.step(model_output=model_pred, timestep=t, sample=latents).prev_sample

                    # Decode
                    latents = latents / self.vae.config.scaling_factor
                    decoded = self.vae.decode(latents).sample
                    decoded = (decoded.clamp(-1,1)+1)/2
                    decoded = decoded.cpu().permute(0,2,3,1).numpy()
                    decoded = (decoded*255).astype(np.uint8)[0]

                    shadowfree_img_cv = cv2.imread(shadowfree_img)
                    shadowfree_img_cv = cv2.cvtColor(shadowfree_img_cv, cv2.COLOR_BGR2RGB)
                    object_mask_cv = cv2.imread(object_mask, cv2.IMREAD_GRAYSCALE)

                    decoded_post = self.post_process(decoded, shadowfree_img_cv, object_mask_cv)
                    decoded_post = cv2.cvtColor(decoded_post, cv2.COLOR_RGB2BGR)

                    preds.append(decoded_post)

        return preds
