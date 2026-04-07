import torch
import torchvision
from libcom.utils.model_download import download_pretrained_model, download_entire_folder
from libcom.utils.process_image import *
from libcom.utils.environment import *
import os
import torchvision.transforms as transforms
import sys
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


cur_dir   = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, 'source/src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from libcom.image_harmonization.source.pct_net import *
from libcom.image_harmonization.source.cdt_net import *


from libcom.image_harmonization.source.src.lbm.inference import get_model
from diffusers import FlowMatchEulerDiscreteScheduler

# =======================================================

model_dir = os.environ.get('LIBCOM_MODEL_DIR',cur_dir)

model_set = ['PCTNet', 'LBM'] 

class ImageHarmonizationModel:
    """
    Image harmonization model.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type, 'PCTNet' or 'LBM'
        kwargs (dict): other parameters for building model.
                       For LBM, you can set 'ckpt_path' here.

    Examples:
        >>> from libcom import ImageHarmonizationModel
        >>> import cv2
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image
        >>> #Use PCTNet
        >>> PCTNet = ImageHarmonizationModel(device=0, model_type='PCTNet')
        >>> comp_img1  = '../tests/source/composite/comp1_PCTNet.jpg'
        >>> comp_mask1 = '../tests/source/composite_mask/mask1_PCTNet.png'
        >>> PCT_result1 = PCTNet(comp_img1, comp_mask1)
        >>> cv2.imwrite('../docs/_static/image/image_harmonization_PCT_result1.jpg', np.concatenate([cv2.imread(comp_img1), cv2.imread(comp_mask1), PCT_result1],axis=1))
        
        >>> #Use LBM
        >>> LBM = ImageHarmonizationModel(device=0, model_type='LBM')
        >>> comp_img  = '../tests/source/composite/1.jpg'
        >>> comp_mask = '../tests/source/composite_mask/1.png'
        >>> LBM_result = LBM(comp_img, comp_mask, steps=4)
        >>> cv2.imwrite('../docs/_static/image/image_harmonization_LBM_result.jpg', np.concatenate([cv2.imread(comp_img), cv2.imread(comp_mask), LBM_result],axis=1))

    Expected result:

    .. image:: _static/image/image_harmonization_PCT_result1.jpg
    .. image:: _static/image/image_harmonization_LBM_result.jpg

    """
    def __init__(self, device=0, model_type='PCTNet', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        self.device = check_gpu_device(device)
        
        if self.model_type == 'PCTNet':
            
            weight_path = os.path.join(model_dir, 'pretrained_models', 'PCTNet.pth')
            download_pretrained_model(weight_path)
            lut_path = os.path.join(model_dir, 'pretrained_models', 'IdentityLUT33.txt')
            download_pretrained_model(lut_path)
            self.build_pretrained_model(weight_path)
            self.build_data_transformer()
            
        elif self.model_type == 'LBM':
            lbm_dir = os.path.join(model_dir, 'pretrained_models', 'lbm_ckpt')
            download_entire_folder(lbm_dir)
            self.build_pretrained_model(lbm_dir)

    def build_pretrained_model(self, weight_path):
        if self.model_type == 'LBM':
            self.model = get_model(weight_path, torch_dtype=torch.bfloat16, device=self.device)
            self.model.bridge_noise_sigma = 0.005
            if self.model.sampling_noise_scheduler is None:
                self.model.sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
                )
            self.model.eval()
        else:
            model = PCTNet()
            model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
            self.model = model.to(self.device).eval()
        
    def build_data_transformer(self):
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def inputs_preprocess(self, composite_image, composite_mask):
        img = read_image_opencv(composite_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = read_mask_opencv(composite_mask) / 255.0
        img_lr = cv2.resize(img, (256, 256))
        mask_lr = cv2.resize(mask, (256, 256))

        #to tensor
        img = self.transformer(img).float().to(self.device)
        mask = self.transformer(mask).float().to(self.device)
        img_lr = self.transformer(img_lr).float().to(self.device)
        mask_lr = self.transformer(mask_lr).float().to(self.device)
        return img, mask, img_lr, mask_lr
    
    def outputs_postprocess(self, outputs):
        if len(outputs.shape) == 4:
            outputs = outputs.squeeze(0)
        outputs = (torch.clamp(255.0 * outputs.permute(1, 2, 0), 0, 255)).cpu().numpy()
        outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        return outputs
    
    @torch.no_grad()
    def __call__(self, composite_image, composite_mask, **kwargs):
        """
        Given a composite image and a foreground mask, perform harmonization on the foreground.

        Args:
            composite_image (str | numpy.ndarray): The path to composite image or the compposite image in ndarray form.
            composite_mask (str | numpy.ndarray): Mask of composite image which indicates the foreground object region in the composite image.
            **kwargs: Extra parameters for inference (e.g., steps=4, resolution=1024 for LBM).

        Returns:
            harmonized_image (np.array): The harmonized result.
        
        """
        if self.model_type == 'LBM':
            return self._inference_lbm(composite_image, composite_mask, **kwargs)

        img, mask, img_lr, mask_lr = self.inputs_preprocess(composite_image, composite_mask)

        outputs = self.model(img_lr, img, mask_lr, mask)
        preds = self.outputs_postprocess(outputs)
        return preds

    @torch.no_grad()
    def _inference_lbm(self, composite_image, composite_mask, **kwargs):
        steps = kwargs.get('steps', self.option.get('steps', 4))
        inference_size = kwargs.get('resolution', self.option.get('resolution', 1024))
        latent_size = inference_size // 8

        if isinstance(composite_image, str):
            src_raw = Image.open(composite_image).convert("RGB")
        else:
            src_raw = Image.fromarray(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))

        if isinstance(composite_mask, str):
            mask_raw = Image.open(composite_mask).convert("L")
        else:
            if len(composite_mask.shape) == 3:
                composite_mask = composite_mask[:, :, 0]
            mask_raw = Image.fromarray(composite_mask).convert("L")

        ori_w, ori_h = src_raw.size

        src_resized = src_raw.resize((inference_size, inference_size), Image.BILINEAR)
        src_t = (ToTensor()(src_resized).unsqueeze(0) * 2 - 1).to(self.device, dtype=torch.bfloat16)
        batch = {"source_image_paste": src_t}

        mask_latent_img = mask_raw.resize((latent_size, latent_size), Image.BILINEAR)
        mask_t = ToTensor()(mask_latent_img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)

        z_source = self.model.vae.encode(batch["source_image_paste"])
        
        output_tensor = self.model.sample(
            z=z_source,
            num_steps=int(steps),
            conditioner_inputs=batch,
            max_samples=1,
            mask=mask_t
        ).clamp(-1, 1)

        res_tensor = (output_tensor[0].float().cpu() + 1) / 2
        preds = (torch.clamp(255.0 * res_tensor.permute(1, 2, 0), 0, 255)).numpy().astype(np.uint8)
        preds = cv2.cvtColor(preds, cv2.COLOR_RGB2BGR)
        
        preds = cv2.resize(preds, (ori_w, ori_h))
        
        return preds