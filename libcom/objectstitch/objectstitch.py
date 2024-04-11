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

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_dir = os.environ.get('LIBCOM_MODEL_DIR',cur_dir)
model_set = ['ObjectStitch'] 

class ObjectStitchModel:
    """
    Unofficial implementation of the paper "ObjectStitch: Object Compositing with Diffusion Model", CVPR 2023.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type.
        kwargs (dict): sampler='ddim' (default) or 'plms', other parameters for building model
    
    Examples:
        >>> from libcom import ObjectStitchModel
        >>> from libcom.utils.process_image import make_image_grid, draw_bbox_on_image
        >>> import cv2
        >>> net    = ObjectStitchModel(device=0, sampler='plms')
        >>> sample_list = ['000000003658', '000000165136']
        >>> sample_dir  = '../tests/objectstitch/'
        >>> bbox_list   = [[0, 253, 186, 509], [217, 177, 499, 410]]
        
        >>> for i, sample in enumerate(sample_list):
        >>>     bg_img = sample_dir + f'background/{sample}.png'
        >>>     fg_img = sample_dir + f'foreground/{sample}.png'
        >>>     fg_mask= sample_dir + f'foreground_mask/{sample}.png'
        >>>     bbox   = bbox_list[i]
        >>>     comp   = net(bg_img, fg_img, fg_mask, bbox, sample_steps=25, num_samples=3)
        >>>     bg_img   = draw_bbox_on_image(bg_img, bbox)
        >>>     grid_img = make_image_grid([bg_img, fg_img] + [comp[i] for i in range(len(comp))])
        >>>     cv2.imwrite(f'../docs/_static/image/objectstitch_result{i+1}.jpg', grid_img)

    Expected result:

    .. image:: _static/image/objectstitch_result1.jpg
        :scale: 38 %
    .. image:: _static/image/objectstitch_result2.jpg
        :scale: 38 %
            
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
        
    def generate_image_batch(self, bg_path, fg_path, fg_mask_path, bbox):
        bg_img     = Image.open(bg_path).convert('RGB')
        bg_w, bg_h = bg_img.size
        bg_t       = self.sd_transform(bg_img)
        fg_img     = Image.open(fg_path).convert('RGB')
        # fill the non-object region of the foreground image with black pixel
        fg_mask= Image.open(fg_mask_path).convert('RGB')
        fg_mask= fg_mask.resize((fg_img.width, fg_img.height))
        black  = np.zeros_like(fg_mask)
        fg_mask= np.asarray(fg_mask)
        fg_img = np.asarray(fg_img)
        fg_img = np.where(fg_mask > 127, fg_img, black)
        fg_img = Image.fromarray(fg_img)
        fg_t       = self.clip_transform(fg_img)
        mask       = Image.fromarray(bbox2mask(bbox, bg_w, bg_h))
        mask_t     = self.mask_transform(mask)
        mask_t     = torch.where(mask_t > 0.5, 1, 0).float()
        inpaint_t  = bg_t * (1 - mask_t)
        bbox_t     = get_bbox_tensor(bbox, bg_w, bg_h)
        return {"bg_img":  inpaint_t.unsqueeze(0),
                "bg_mask": mask_t.unsqueeze(0),
                "fg_img":  fg_t.unsqueeze(0),
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
        fg_tensor = batch['fg_img'].to(self.device)
        
        c = self.model.get_learned_conditioning(fg_tensor)
        c = self.model.proj_out(c)
        uc = self.model.learnable_vector.repeat(c.shape[0], c.shape[1], 1) # 1,1,768
        return test_model_kwargs, c, uc
    
    def inputs_preprocess(self, background_image, foreground_image, foreground_mask, bbox, num_samples):
        batch = self.generate_image_batch(background_image, foreground_image, foreground_mask, bbox)
        test_kwargs, c, uc = self.prepare_input(batch, self.latent_shape, num_samples)
        return test_kwargs, c, uc 
    
    
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
            foreground_image (str | numpy.ndarray): The path to foreground image or the background image in ndarray form.
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
        test_kwargs, c, uc = self.inputs_preprocess(background_image, foreground_image, 
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
        return comp_img