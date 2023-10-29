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
from .source.ControlCom.ldm.util import instantiate_from_config
from .source.ControlCom.ldm.models.diffusion.ddim import DDIMSampler
from .source.ControlCom.ldm.models.diffusion.plms import PLMSSampler
from .source.ControlCom.ldm.data.open_images_control import get_tensor, get_tensor_clip, get_bbox_tensor, bbox2mask, tensor2numpy

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_set = ['ControlCom'] 
task_set  = ['blending', 'harmonization'] # 'viewsynthesis', 'composition'

class ControlComModel:
    def __init__(self, device=0, model_type='ControlCom', **kwargs):
        '''
        device: gpu id, type=str/torch.device
        model_type: predefined model type, type=str
        kwargs: other parameters for building model, type=dict
        '''
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        
        weight_path = os.path.join(cur_dir, 'pretrained_models', 'ControlCom.pth')
        download_pretrained_model(weight_path)
        
        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, weight_path):
        pl_sd  = torch.load(weight_path, map_location="cpu")
        sd     = pl_sd["state_dict"]
        config = OmegaConf.load(os.path.join(cur_dir, 'source/ControlCom/configs/controlcom.yaml'))
        clip_path = os.path.join(cur_dir, '../shared_pretrained_models', 'openai-clip-vit-large-patch14')
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
        if fg_mask_path != None:
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

    def prepare_input(self, batch, shape, num_samples, indicator):
        if num_samples > 1:
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = torch.cat([batch[k]] * num_samples, dim=0)
        device    = self.device
        test_model_kwargs={}
        bg_img    = batch['bg_img'].to(device)
        bg_latent = self.model.encode_first_stage(bg_img)
        bg_latent = self.model.get_first_stage_encoding(bg_latent).detach()
        test_model_kwargs['bg_latent'] = bg_latent
        rs_mask = F.interpolate(batch['bg_mask'].to(device), shape[-2:])
        rs_mask = torch.where(rs_mask > 0.5, 1.0, 0.0)
        test_model_kwargs['bg_mask']  = rs_mask
        test_model_kwargs['bbox'] = batch['bbox'].to(device)
        fg_tensor = batch['fg_img'].to(device)
        c = self.model.get_learned_conditioning((fg_tensor, None))
        indicator = torch.tensor(indicator).int().to(device)
        if num_samples // len(indicator) > 1:
            indicator = indicator.repeat_interleave(num_samples // len(indicator), dim=0)
        c.append(indicator)
        c.append(torch.tensor([True] * indicator.shape[0])) 
        uc_global = self.model.learnable_vector.repeat(c[0].shape[0], 1, 1) # 1,1,768
        if hasattr(self.model, 'get_unconditional_local_embedding'):
            uc_local = self.model.get_unconditional_local_embedding(c[1])
        else:
            uc_local = c[1]
        uc = [uc_global, uc_local, indicator, torch.tensor([False] * indicator.shape[0])]
        return test_model_kwargs, c, uc
    
    def inputs_preprocess(self, background_image, foreground_image, bbox, 
                          foreground_mask, indicator, num_samples):
        batch = self.generate_image_batch(background_image, foreground_image, foreground_mask, bbox)
        test_kwargs, c, uc = self.prepare_input(batch, self.latent_shape, num_samples*len(indicator), indicator)
        return test_kwargs, c, uc 
    
    
    def outputs_postprocess(self, outputs):
        x_samples_ddim = self.model.decode_first_stage(outputs[:,:4]).cpu().float()
        comp_img = tensor2numpy(x_samples_ddim, image_size=self.image_size)
        if len(comp_img) == 1:
            return comp_img[0]
        return comp_img
    
    def task_to_indicator(self, task):
        if not isinstance(task, list):
            task  = [task]
        indicator = []
        for t in task:
            assert t in task_set, f'Unsupported task {t}'
            if t == 'blending':
                indicator.append([0,0])
            elif t == 'harmonization':
                indicator.append([1,0])
        return indicator
                
    @torch.no_grad()
    def __call__(self, background_image, foreground_image, bbox, 
                 foreground_mask=None, task='blending', 
                 num_samples=1, sample_steps=50, guidance_scale=5, seed=321):
        '''
        background_image, foreground_image: type=str or numpy array or PIL.Image
        bbox: [x1, y1, x2, y2]
        task: str or list of str
        num_samples: int
        '''
        seed_everything(seed)
        indicator = self.task_to_indicator(task)
        test_kwargs, c, uc = self.inputs_preprocess(background_image, foreground_image, bbox, 
                                                    foreground_mask, indicator, num_samples)
        start_code = torch.randn([num_samples]+self.latent_shape, device=self.device)
        start_code = start_code.repeat(len(indicator), 1, 1, 1)
        outputs, _ = self.sampler.sample(S=sample_steps,
                                    conditioning=c,
                                    batch_size=num_samples*len(indicator),
                                    shape=self.latent_shape,
                                    verbose=False,
                                    eta=0.0,
                                    x_T=start_code,
                                    unconditional_guidance_scale=guidance_scale,
                                    unconditional_conditioning=uc,
                                    test_model_kwargs=test_kwargs)
        comp_img   = self.outputs_postprocess(outputs)
        return comp_img