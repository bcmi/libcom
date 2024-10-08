import torch
import torchvision
from libcom.utils.model_download import download_pretrained_model, download_entire_folder
from libcom.utils.process_image import *
from libcom.utils.environment import *
import torch 
import os
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
try:
    from lightning_fabric.utilities.seed import log
    log.propagate = False
except:
    pass
from torch import device
import torchvision
from .source.ObjectStitch.ldm.util import instantiate_from_config
from .source.ObjectStitch.ldm.models.diffusion.ddim import DDIMSampler
from .source.ObjectStitch.ldm.models.diffusion.plms import PLMSSampler
from .source.ObjectStitch.ldm.data.open_images import get_tensor, get_tensor_clip, get_bbox_tensor, bbox2mask, mask2bbox, tensor2numpy

import copy

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_dir = os.environ.get('LIBCOM_MODEL_DIR',cur_dir)
model_set = ['ObjectStitch'] 

class Mure_ObjectStitchModel:
    """
    Unofficial implementation of the paper "ObjectStitch: Object Compositing with Diffusion Model", CVPR 2023.
    Building upon ObjectStitch, we have made improvements to support input of multiple foreground images.
    
    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type.
        kwargs (dict): sampler='ddim' (default) or 'plms', other parameters for building model
    
    Examples:
        >>> from libcom import MureObjectStitchModel
        >>> from libcom.utils.process_image import make_image_grid, draw_bbox_on_image
        >>> import cv2
        >>> import os
        >>> net    = MureObjectStitchModel(device=0, sampler='plms')
        >>> sample_list = ['000000000003', '000000000004']
        >>> sample_dir  = './tests/mure_objectstitch/'
        >>> bbox_list   = [[623, 1297, 1159, 1564], [363, 205, 476, 276]]
        >>> for i, sample in enumerate(sample_list):
        >>>     bg_img = sample_dir + f'background/{sample}.jpg'
        >>>     fg_img_path = sample_dir + f'foreground/{sample}/'
        >>>     fg_mask_path = sample_dir + f'foreground_mask/{sample}/'
        >>>     fg_img_list = [os.path.join(fg_img_path, f) for f in os.listdir(fg_img_path)]
        >>>     fg_mask_list = [os.path.join(fg_mask_path, f) for f in os.listdir(fg_mask_path)]
        >>>     bbox   = bbox_list[i]
        >>>     comp, show_fg_img = net(bg_img, fg_img_list, fg_mask_list, bbox, sample_steps=25, num_samples=3)
        >>>     bg_img   = draw_bbox_on_image(bg_img, bbox)
        >>>     grid_img = make_image_grid([bg_img, show_fg_img] + [comp[i] for i in range(len(comp))])
        >>>     cv2.imwrite(f'../docs/_static/image/mureobjectstitch_result{i+1}.jpg', grid_img)

    Expected result:

    .. image:: _static/image/mureobjectstitch_result1.jpg
        :scale: 21 %
    .. image:: _static/image/mureobjectstitch_result2.jpg
        :scale: 21 %
            
        
    """
    def __init__(self, device=0, model_type='ObjectStitch', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        
        weight_path = os.path.join(cur_dir, 'pretrained_models', f'{self.model_type}.pth')
        download_pretrained_model(weight_path)
        
        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, weight_path):
        pl_sd  = torch.load(weight_path, map_location="cpu")
        sd     = pl_sd["state_dict"]
        config = OmegaConf.load(os.path.join(cur_dir, 'source/ObjectStitch/configs/objectstitch.yaml'))
        clip_path = os.path.join(model_dir, '../shared_pretrained_models', 'openai-clip-vit-large-patch14')
        download_entire_folder(clip_path)
        config.model.params.cond_stage_config.params.version = clip_path
        model  = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        self.model   = model.to(self.device).eval()
        if self.option.get('sampler', 'ddim') == 'plms':
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)
        
    def build_data_transformer(self):
        self.image_size       = (512, 512)
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.sd_transform   = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.latent_shape   = [4, self.image_size[0] // 8, self.image_size[1] // 8]

    def constant_pad_bbox(self, bbox, width, height, value=10):
        # Get reference image
        bbox_pad = copy.deepcopy(bbox)
        left_space = bbox[0]
        up_space = bbox[1]
        right_space = width - bbox[2]
        down_space = height - bbox[3]

        bbox_pad[0] = bbox[0]-min(value, left_space)
        bbox_pad[1] = bbox[1]-min(value, up_space)
        bbox_pad[2] = bbox[2]+min(value, right_space)
        bbox_pad[3] = bbox[3]+min(value, down_space)
        return bbox_pad

    def crop_foreground_by_bbox(self, img, mask, bbox, pad_bbox=10):
        width, height = img.shape[1], img.shape[0]
        bbox_pad = self.constant_pad_bbox(
            bbox, width, height, pad_bbox) if pad_bbox > 0 else bbox
        img = img[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2]]
        if mask is not None:
            mask = mask[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2]]
        return img, mask, bbox_pad

    def draw_compose_fg_img(self, fg_img_compose):
        final_img = Image.new('RGB', (512, 512), (255, 255, 255))
        fg_img_nums = len(fg_img_compose)

        if fg_img_nums == 1:
            size = (512, 512)
            positions = [(0, 0)]
        elif fg_img_nums == 2:
            size = (256, 512)
            positions = [(0, 0), (256, 0)]
        elif fg_img_nums == 3:
            size = (256, 512)
            positions = [(0, 0), (256, 0), (0, 256)]
        elif fg_img_nums == 4:
            positions = [(0, 0), (256, 0), (0, 256), (256, 256)]
            size = (256, 256)
        else :
            positions = [(0, 0), (256, 0), (0, 256), (256, 256), (128, 128)]
            size = (256, 256)

        if fg_img_nums>5:
            fg_img_compose = fg_img_compose[:5]
        
        for idx, img in enumerate(fg_img_compose):
            fg_img = img.resize(size)
            final_img.paste(fg_img, positions[idx])
        
        return final_img

    def rescale_image_with_bbox(self, image, bbox=None, long_size=1024):
        src_width, src_height = image.size
        if max(src_width, src_height) <= long_size:
            dst_img = image
            dst_width, dst_height = dst_img.size
        else:
            scale = float(long_size) / max(src_width, src_height)
            dst_width, dst_height = int(scale * src_width), int(scale * src_height)
            dst_img = image.resize((dst_width, dst_height))
        if bbox == None:
            return dst_img
        bbox[0] = int(float(bbox[0]) / src_width * dst_width)
        bbox[1] = int(float(bbox[1]) / src_height * dst_height)
        bbox[2] = int(float(bbox[2]) / src_width * dst_width)
        bbox[3] = int(float(bbox[3]) / src_height * dst_height)
        return dst_img, bbox

    def mask_bboxregion_coordinate(self, mask):
        valid_index = np.argwhere(mask == 255)  # [length,2]
        if np.shape(valid_index)[0] < 1:
            x_left = 0
            x_right = 0
            y_bottom = 0
            y_top = 0
        else:
            x_left = np.min(valid_index[:, 1])
            x_right = np.max(valid_index[:, 1])
            y_bottom = np.max(valid_index[:, 0])
            y_top = np.min(valid_index[:, 0])
        return x_left, x_right, y_bottom, y_top

    def generate_multifg(self, fg_list_path, fgmask_list_path):
        fg_list, fg_mask_list, fg_img_list, fg_img_compose = [], [], [], []

        assert len(fg_list_path) < 11, "too many foreground images"
        for fg_img_path in fg_list_path:
            fg_img = Image.open(fg_img_path).convert('RGB')
            fg_list.append(fg_img)
        for fg_mask_name in fgmask_list_path:
            fg_mask = Image.open(fg_mask_name).convert('RGB')
            fg_mask_list.append(fg_mask)
        
        for idx, fg_mask in enumerate(fg_mask_list):
            fg_mask = fg_mask.convert('L')
            mask = np.asarray(fg_mask)
            m = np.array(mask > 0).astype(np.uint8)
            fg_mask = Image.fromarray(m * 255)
            x_left, x_right, y_bottom, y_top = self.mask_bboxregion_coordinate(np.array(fg_mask))
            H, W = (np.array(fg_mask)).shape[:2]
            x_right=min(x_right, W-1)
            y_bottom=min(y_bottom, H-1)
            fg_bbox = [x_left, y_top, x_right, y_bottom]
            fg_img, fg_bbox = self.rescale_image_with_bbox(fg_list[idx], fg_bbox)
            fg_img = np.array(fg_img)
            fg_mask = fg_mask.resize((fg_img.shape[1], fg_img.shape[0]))
            fg_mask = np.array(fg_mask)
            fg_img, fg_mask, fg_bbox = self.crop_foreground_by_bbox(fg_img, fg_mask, fg_bbox)
            fg_mask = np.array(Image.fromarray(fg_mask).convert('RGB'))
            black = np.zeros_like(fg_mask)
            fg_img = np.where(fg_mask > 127, fg_img, black)
            fg_img = Image.fromarray(fg_img)
            fg_img_compose.append(fg_img)
            fg_t = self.clip_transform(fg_img)
            fg_img_list.append(fg_t.unsqueeze(0))
        fg_img = self.draw_compose_fg_img(fg_img_compose)

        return fg_img_list, fg_img

    def generate_image_batch(self, bg_path, fg_list_path, fgmask_list_path, bbox):

        bg_img     = Image.open(bg_path).convert('RGB')
        bg_w, bg_h = bg_img.size
        bg_t       = self.sd_transform(bg_img)

        ## jiaxuan
        fg_img_list, fg_img = self.generate_multifg(fg_list_path, fgmask_list_path)

        mask       = Image.fromarray(bbox2mask(bbox, bg_w, bg_h))
        mask_t     = self.mask_transform(mask)
        mask_t     = torch.where(mask_t > 0.5, 1, 0).float()
        inpaint_t  = bg_t * (1 - mask_t)
        bbox_t     = get_bbox_tensor(bbox, bg_w, bg_h)

        return {"bg_img":  inpaint_t.unsqueeze(0),
                "bg_mask": mask_t.unsqueeze(0),
                "fg_img":  fg_img,
                "fg_img_list": fg_img_list,
                "bbox":    bbox_t.unsqueeze(0)}
    
    def prepare_input(self, batch, shape, num_samples):
        if num_samples > 1:
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = torch.cat([batch[k]] * num_samples, dim=0)

        test_model_kwargs={}
        bg_img    = batch['bg_img'].to(self.device)
        bg_latent = self.model.encode_first_stage(bg_img)
        bg_latent = self.model.get_first_stage_encoding(bg_latent).detach()
        test_model_kwargs['bg_latent'] = bg_latent
        rs_mask = F.interpolate(batch['bg_mask'].to(self.device), shape[-2:])
        rs_mask = torch.where(rs_mask > 0.5, 1.0, 0.0)
        test_model_kwargs['bg_mask']  = rs_mask
        test_model_kwargs['bbox']  = batch['bbox'].to(self.device)

        condition_list = []
        for fg_img in batch['fg_img_list']:
            fg_img = fg_img.to(self.device)
            condition = self.model.get_learned_conditioning(fg_img)
            condition = self.model.proj_out(condition)
            condition_list.append(condition)
        c = torch.cat(condition_list, dim=1)
        c = torch.cat([c] * num_samples, dim=0)
        uc = self.model.learnable_vector.repeat(c.shape[0], c.shape[1], 1)  # 1,1,768
        return test_model_kwargs, c, uc
    
    def inputs_preprocess(self, background_image, fg_list_path, fgmask_list_path, bbox, num_samples):

        batch = self.generate_image_batch(background_image, fg_list_path, fgmask_list_path, bbox)
        test_kwargs, c, uc = self.prepare_input(batch, self.latent_shape, num_samples)
        show_fg_img = batch["fg_img"]

        return test_kwargs, c, uc, show_fg_img
    
    
    def outputs_postprocess(self, outputs):
        x_samples_ddim = self.model.decode_first_stage(outputs[:,:4]).cpu().float()
        comp_img = tensor2numpy(x_samples_ddim, image_size=self.image_size)
        if len(comp_img) == 1:
            return comp_img[0]
        return comp_img
                
    @torch.no_grad()
    def __call__(self, background_image, foreground_image, foreground_mask, bbox, 
                 num_samples=1, sample_steps=50, guidance_scale=5, seed=321):
        """
        Controllable image composition based on diffusion model.

        Args:
            background_image (str | numpy.ndarray): The path to background image or the background image in ndarray form.
            foreground_image (str | numpy.ndarray): The path to the list of foreground images or the foreground images in ndarray form.
            foreground_mask (None | str | numpy.ndarray): Mask of foreground image which indicates the foreground object region in the foreground image.
            bbox (list): The bounding box which indicates the foreground's location in the background. [x1, y1, x2, y2].
            num_samples (int): Number of images to be generated. default: 1.
            sample_steps (int): Number of denoising steps. The recommended setting is 25 for PLMS sampler and 50 for DDIM sampler. default: 50.
            guidance_scale (int): Scale in classifier-free guidance (minimum: 1; maximum: 20). default: 5.
            seed (int): Random Seed is used to reproduce results and same seed will lead to same results. 

        Returns:
            composite_images (numpy.ndarray): Generated images with a shape of 512x512x3 or Nx512x512x3, where N indicates the number of generated images. 


        """
        seed_everything(seed)
        test_kwargs, c, uc, show_fg_img = self.inputs_preprocess(background_image, foreground_image, 
                                                    foreground_mask, bbox, num_samples)

        start_code = torch.randn([num_samples]+self.latent_shape, device=self.device)
        outputs, _ = self.sampler.sample(S=sample_steps,
                                    conditioning=c,
                                    batch_size=num_samples,
                                    shape=self.latent_shape,
                                    verbose=False,
                                    eta=0.0,
                                    x_T=start_code,
                                    unconditional_guidance_scale=guidance_scale,
                                    unconditional_conditioning=uc,
                                    test_model_kwargs=test_kwargs)
        comp_img   = self.outputs_postprocess(outputs)
        return comp_img, show_fg_img