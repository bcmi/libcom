import torch
import torchvision
import torchvision.transforms as transforms
from libcom.utils.model_download import download_pretrained_model, download_entire_folder
from libcom.utils.process_image import *
from libcom.utils.environment import *
import torch
import os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
try:
    from lightning_fabric.utilities.seed import log
    log.propagate = False
except:
    pass
from .source.PostProcessModel import PostProcess
from .source.cldm.model import load_state_dict

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_set = ['ShadowGeneration'] 

class ShadowGenerationModel:
    """
    Foreground shadow generation model based on diffusion model and control net.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type
        kwargs (dict): other parameters for building model

    Examples:
        >>> from libcom import ShadowGenerationModel
        >>> from libcom.utils.process_image import make_image_grid
        >>> import cv2
        >>> net = ShadowGenerationModel(device=2, model_type='ShadowGeneration')
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

    .. image:: _static/image/shadow_generation_result3.jpg
        :scale: 32 %
    .. image:: _static/image/shadow_generation_result4.jpg
        :scale: 32 %
        
    """
    def __init__(self, device=0, model_type='ShadowGeneration', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        cldm_weight_path = os.path.join(cur_dir, 'pretrained_models', 'Shadow_cldm.pth')
        ppp_weight_path  = os.path.join(cur_dir, 'pretrained_models', 'Shadow_ppp.pth')
        download_pretrained_model(ppp_weight_path)
        download_pretrained_model(cldm_weight_path)
        self.device = check_gpu_device(device)
        self.build_pretrained_model(ppp_weight_path, cldm_weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, ppp_weight_path, cldm_weight_path):
        config_path = os.path.join(cur_dir, 'source', 'cldm_v15.yaml')
        config      = OmegaConf.load(config_path)
        clip_path   = os.path.join(cur_dir, '../shared_pretrained_models', 'openai-clip-vit-large-patch14')
        download_entire_folder(clip_path)
        config.model.params.cond_stage_config.params.version = clip_path
        model = PostProcess(
            model_path=config,
            control_net_path=cldm_weight_path,
            infe_steps=50
            )
        model.load_state_dict(load_state_dict(ppp_weight_path, location='cpu'), strict=False)
        self.model = model.to(self.device).eval()
        
    def build_data_transformer(self):
        self.image_size = 512
        self.transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def inputs_preprocess(self, composite_image, composite_mask):
        img  = read_image_pil(composite_image)
        img  = self.transformer(img).permute(1,2,0)
        mask = read_mask_pil(composite_mask)
        mask = self.transformer(mask).permute(1,2,0)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        cat_img = torch.cat([img, mask], dim=-1)
        return img.to(self.device), mask.to(self.device), cat_img.to(self.device)
    
    def outputs_postprocess(self, outputs):
        img, output = outputs
        img = img * 255
        pred_mask = torch.greater_equal(output[:, :, :, 3], 0).int()
        adjusted_img = output[:, :, :, :3]
        adjusted_img = torch.clamp(adjusted_img, -1., 1.)
        adjusted_img = (adjusted_img + 1.0) / 2.0
        adjusted_img = (adjusted_img * 255).int()
        composite_img = adjusted_img * pred_mask.unsqueeze(3) + (1-pred_mask.unsqueeze(3)) * img
        composite_img = np.array(composite_img.cpu().squeeze(0), dtype=np.uint8)
        # change rgb to bgr color
        composite_img = composite_img[:,:,[2,1,0]]
        return composite_img
    
    @torch.no_grad()
    def inf_img(self, inputs):
        img, mask, cat_img = inputs
        batch = dict(jpg=img, txt=[''], hint=cat_img, mask=None, gt_mask=mask)
        # get predicted image   
        images = self.model.model.log_images(batch, mode='pndm', input=(img*2-1).permute(0,3,1,2), 
                                             ddim_steps=50, add_noise_strength=1)
        image_scaled = images['samples_cfg_scale_9.00'].permute(0,2,3,1)
        # go through post process net
        input = torch.concat([image_scaled, img * 2 - 1, mask], dim=-1)
        null_timeteps = torch.zeros(1, device=input.device)
        output = self.model.post_process_net(input.permute(0,3,1,2), timesteps=null_timeteps)
        output = output.permute(0,2,3,1)

        return img, output
    

    @torch.no_grad()
    def __call__(self, composite_image, composite_mask, number=5, seed=0):
        """
        Generate shadow for foreground object.
        
        Args:
            composite_img (str | numpy.ndarray): The path to composite image or composite image in ndarray form.
            composite_mask (str | numpy.ndarray): The path to foreground object mask or foreground object mask in ndarray form.
            number (int): Number of images to be inferenced. default: 5.
            seed: Random Seed is used to reproduce results and same seed will lead to same results. 
        
        Returns: 
            generated_images (list): A list of images with generated foreground shadows. Each image is in ndarray form with a shape of 512x512x3

        """
        seed_everything(seed)
        inputs    = self.inputs_preprocess(composite_image, composite_mask)
        preds = []
        for _ in range(number):
            outputs = self.inf_img(inputs)
            pred = self.outputs_postprocess(outputs)
            preds.append(pred)
        return preds