import torch
import torchvision
from libcom.utils.model_download import download_pretrained_model, download_entire_folder
from libcom.utils.process_image import *
from libcom.utils.environment import *
import torch 
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms.functional as tf
from .source.PHDNet.phdnet import PHDNet
from .source.PHDiffusion.ldm.modules.encoders.adapter import Adapter,NoRes_Adapter
from .source.PHDiffusion.ldm.models.diffusion.scheduling_pndm import PNDMScheduler
from .source.PHDiffusion.ldm.util import instantiate_from_config
import logging
logging.getLogger('transformers').setLevel(logging.ERROR) # disable transformer lib warning 

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_set = ['PHDNet', 'PHDiffusion'] 

class PainterlyHarmonizationModel:
    def __init__(self, device=0, model_type='PHDNet', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        
        if model_type == 'PHDNet':
            weight_path = os.path.join(cur_dir, 'pretrained_models', model_type + '.pth')
            self.device = check_gpu_device(device)
            download_pretrained_model(weight_path)
            self.build_pretrained_model(weight_path)
        elif model_type == 'PHDiffusion':
            self.use_residual = self.option.get('use_residual', True)
            if self.use_residual:
                phdiff_weight_path = os.path.join(cur_dir, 'pretrained_models', model_type+'WithRes.pth')
            else:
                phdiff_weight_path = os.path.join(cur_dir, 'pretrained_models', model_type+'.pth')
            sd_weight_path = os.path.join(cur_dir, '../shared_pretrained_models', 'sd-v1-4.ckpt')
            download_pretrained_model(sd_weight_path)
            download_pretrained_model(phdiff_weight_path)
            self.device = check_gpu_device(device)
            self.build_pretrained_model(sd_weight_path, phdiff_weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, *weight_path):
        if len(weight_path) == 1:
            weight_path = weight_path[0]
            # build PHDNet
            assert self.model_type == 'PHDNet', self.model_type
            model = PHDNet(self.device)
            model.load_networks(weight_path)
            self.model = model.to(self.device).eval()
        elif len(weight_path) == 2:
            # build PHDiffusion model
            sd_weight_path, phdiff_weight_path = weight_path
            assert self.model_type == 'PHDiffusion', self.model_type
            self.config= OmegaConf.load(cur_dir+'/source/PHDiffusion/stable_diffusion.yaml')
            clip_path = os.path.join(cur_dir, '../shared_pretrained_models', 'openai-clip-vit-large-patch14')
            download_entire_folder(clip_path)
            self.config.model.params.cond_stage_config.params.model_path = clip_path
            pl_sd = torch.load(sd_weight_path, map_location="cpu")
            sd = pl_sd["state_dict"]
            model = instantiate_from_config(self.config.model)
            model.load_state_dict(sd, strict=False)

            if self.use_residual:
                adapter=Adapter(cin=int(64 * 4), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
            else:
                adapter=NoRes_Adapter(cin=int(64 * 4), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
            model_resume_state = torch.load(phdiff_weight_path, map_location='cpu')
            adapter.load_state_dict(model_resume_state['ad'])
            model.model.diffusion_model.interact_blocks.load_state_dict(model_resume_state['interact'])

            self.model   = model.to(self.device).eval()
            self.adapter = adapter.to(self.device).eval()
            
            self.scheduler = PNDMScheduler(
                                    beta_end=0.012,
                                    beta_schedule='scaled_linear',
                                    beta_start=0.00085,
                                    num_train_timesteps=1000,
                                    set_alpha_to_one=False,
                                    skip_prk_steps=True,
                                    steps_offset=1,
                                    )

    def build_data_transformer(self):
        self.image_size = 512
        self.transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def inputs_preprocess(self, composite_image, composite_mask):
        img  = read_image_pil(composite_image)
        img  = self.transformer(img)
        mask = read_mask_pil(composite_mask).convert('L')
        mask = self.mask_transform(mask)
        if self.model_type == 'PHDiffusion':
            img  = img.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device) 
        return img, mask
    
    def outputs_postprocess(self, outputs):
        if outputs.dim() == 4:
            outputs = outputs.squeeze(0)
        outputs = (torch.clamp((outputs.permute(1, 2, 0) + 1) / 2.0 * 255, 0, 255)).cpu().numpy()
        outputs = outputs.astype(np.uint8)
        outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        return outputs
    
    @torch.no_grad()
    def PHDNet_inference(self, composite_image, composite_mask):
        '''
        composite_image, composite_mask: type=str or numpy array or PIL.Image
        '''
        comp, mask = self.inputs_preprocess(composite_image, composite_mask)
        outputs    = self.model(comp, mask)
        preds      = self.outputs_postprocess(outputs)
        return preds
    
    @torch.no_grad()
    def __call__(self, composite_image, composite_mask, sample_steps=50, strength=0.7, random_seed=None):
        if self.model_type == 'PHDNet':
            return self.PHDNet_inference(composite_image, composite_mask)
        if random_seed != None:
            seed_everything(random_seed)
        comp, mask = self.inputs_preprocess(composite_image, composite_mask)
        self.scheduler.set_timesteps(sample_steps, device=self.device)
        init_timestep = min(int(sample_steps * strength), sample_steps)
        t_start       = max(sample_steps - init_timestep, 0)
        timesteps     = self.scheduler.timesteps[t_start:]
        
        c = self.model.get_learned_conditioning([''])
        batch_size = mask.shape[0]
        latent_timestep = timesteps[:1].repeat(batch_size)

        x_0 = self.model.encode_first_stage(comp)
        x_0 = self.model.get_first_stage_encoding(x_0)
        mask_latents = F.interpolate(mask, size=x_0.shape[-2:])
        noise   = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype)
        latents = self.scheduler.add_noise(x_0, noise, latent_timestep)

        adapter_input = torch.cat((comp, mask), dim=1).to(dtype=comp.dtype)
        features_adapter = self.adapter(adapter_input)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            noise = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype)
            t_latents = self.scheduler.add_noise(x_0, noise, t)
            latents = latents * mask_latents + t_latents * (1 - mask_latents)

            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.model.model.diffusion_model(x=latent_model_input,
                                                          fg_mask=mask_latents, 
                                                          timesteps=t.repeat(batch_size), 
                                                          context=c, 
                                                          features_adapter=features_adapter)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        x_samples_ddim = self.model.decode_first_stage(latents)
        preds = self.outputs_postprocess(x_samples_ddim)
        return preds

