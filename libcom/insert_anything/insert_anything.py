import torch
from libcom.utils.model_download import download_pretrained_model, download_entire_folder
from libcom.utils.process_image import *
from libcom.utils.environment import *
import torch
import os
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything
try:
    from lightning_fabric.utilities.seed import log
    log.propagate = False
except:
    pass
import torchvision
from PIL import Image
import torch
import os
import numpy as np
import cv2
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from .source.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask, bbox2mask

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_dir = os.environ.get('LIBCOM_MODEL_DIR',cur_dir)
model_set = ['insertanything']

class InsertAnythingModel:
    """
    Object insertion model 

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type.
        kwargs (dict): other parameters for building model

    Examples:
        >>> from libcom import InsertAnythingModel
        >>> from libcom.utils.process_image import make_image_grid, draw_bbox_on_image
        >>> import cv2
        >>> import os
        >>> net    = InsertAnythingModel(device=0)
        >>> sample_list = ['000000000003', '000000000004']
        >>> sample_dir  = './tests/insertanything/'
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
        >>>     cv2.imwrite(f'../docs/_static/image/insertanything_result{i+1}.jpg', grid_img)

    Expected result:

    .. image:: _static/image/insertanything_result1.jpg
        :scale: 21 %
    .. image:: _static/image/insertanything_result2.jpg
        :scale: 21 %


    """
    def __init__(self, device=0, model_type='insertanything', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs

        weight_path = 'black-forest-labs/FLUX.1-Fill-dev'
        redux_path = 'black-forest-labs/FLUX.1-Redux-dev'
        lora_path = os.path.join(model_dir, 'pretrained_models', 'insert_anything_lora.safetensors')
        download_pretrained_model(lora_path)

        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path, lora_path, redux_path)

    def build_pretrained_model(self, weight_path, lora_path, redux_path):
        self.pipeline = FluxFillPipeline.from_pretrained(weight_path, torch_dtype=torch.bfloat16)
        self.pipeline.load_lora_weights(lora_path)
        self.redux = FluxPriorReduxPipeline.from_pretrained(redux_path, torch_dtype=torch.bfloat16)
        self.pipeline.to(self.device)
        self.redux.to(self.device)
    
    def inputs_preprocess(self, background_image, fg_path, fgmask_path, bbox):
        size = (768, 768)
        bg_img = Image.open(background_image).convert('RGB')
        fg_img = Image.open(fg_path).convert('RGB')
        # 读取 mask，二值化
        mask = (cv2.imread(fgmask_path, cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
        # 调整 mask 尺寸到和 fg_img 一致
        mask = cv2.resize(mask, (fg_img.width, fg_img.height), interpolation=cv2.INTER_NEAREST)
        tar_image = np.array(bg_img)
        tar_mask = bbox2mask(bbox, tar_image.shape[1], tar_image.shape[0])
        ref_image = np.array(fg_img)
        ref_mask = mask
        
        # Remove the background information of the reference picture
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 

        # Extract the box where the reference image is located, and place the reference object at the center of the image
        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:] 
        ref_mask = ref_mask[y1:y2,x1:x2] 
        ratio = 1.3
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio) 
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)

        # Dilate the mask
        kernel = np.ones((7, 7), np.uint8)
        iterations = 2
        tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

        # zome in
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

        tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=2)   
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
        y1,y2,x1,x2 = tar_box_yyxx_crop

        old_tar_image = tar_image.copy()
        tar_image = tar_image[y1:y2,x1:x2,:]
        tar_mask = tar_mask[y1:y2,x1:x2]

        H1, W1 = tar_image.shape[0], tar_image.shape[1]

        tar_mask = pad_to_square(tar_mask, pad_value=0)
        tar_mask = cv2.resize(tar_mask, size)

        # Extract the features of the reference image
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
        pipe_prior_output = self.redux(Image.fromarray(masked_ref_image))

        tar_image = pad_to_square(tar_image, pad_value=255)
        H2, W2 = tar_image.shape[0], tar_image.shape[1]

        tar_image = cv2.resize(tar_image, size)
        diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)

        tar_mask = np.stack([tar_mask,tar_mask,tar_mask],-1)
        mask_black = np.ones_like(tar_image) * 0
        mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)
        
        diptych_ref_tar = Image.fromarray(diptych_ref_tar)
        mask_diptych[mask_diptych == 1] = 255
        mask_diptych = Image.fromarray(mask_diptych)
        extra_sizes = np.array([H1, W1, H2, W2])
        tar_box_yyxx_crop = np.array(tar_box_yyxx_crop)
        
        return {"diptych_ref_tar": diptych_ref_tar,
                "mask_diptych": mask_diptych,
                "pipe_prior_output": pipe_prior_output,
                "old_tar_image": old_tar_image,
                "extra_sizes": extra_sizes,
                "tar_box_yyxx_crop": tar_box_yyxx_crop}


    def outputs_postprocess(self, edited_image, old_tar_image, extra_sizes, tar_box_yyxx_crop):
        width, height = edited_image.size
        left = width // 2
        right = width
        top = 0
        bottom = height
        edited_image = edited_image.crop((left, top, right, bottom))
        edited_image = np.array(edited_image)
        edited_image = crop_back(edited_image, old_tar_image, extra_sizes, tar_box_yyxx_crop) 
        return edited_image


    @torch.no_grad()
    def __call__(self, background_image, foreground_image, foreground_mask, bbox,
                 num_samples=1, sample_steps=50, guidance_scale=30, seed=321):
        """
        Object Insertion model based on FluxFill and FluxRedux.

        Args:
            background_image (str): The path to the background image.
            foreground_image (str): The path to the foreground image.
            foreground_mask (str): Mask of foreground image which indicates the foreground object region in the foreground image.
            bbox (list): The bounding box which indicates the foreground's location in the background. [x1, y1, x2, y2].
            num_samples (int): Number of images to be generated. default: 1.
            sample_steps (int): Number of denoising steps. The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference. default: 50.
            guidance_scale (int): Scale in classifier-free guidance. default: 30.
            seed (int): Random Seed is used to reproduce results and same seed will lead to same results.
        Returns:
            composite_images (numpy.ndarray): Generated images with a shape of 512x512x3 or Nx512x512x3, where N indicates the number of generated images.
        """
        seed_everything(seed)
        batch = self.inputs_preprocess(background_image, foreground_image, foreground_mask, bbox)
        comp_image_list = []
        for i in range(num_samples):
            edited_image = self.pipeline(
                        image=batch['diptych_ref_tar'],
                        mask_image=batch['mask_diptych'],
                        height=batch['mask_diptych'].size[1],
                        width=batch['mask_diptych'].size[0],
                        max_sequence_length=512,
                        num_inference_steps=sample_steps,
                        guidance_scale=guidance_scale,
                        **batch['pipe_prior_output'],
                    ).images[0] 
            comp_img  = self.outputs_postprocess(edited_image, batch['old_tar_image'], batch['extra_sizes'], batch['tar_box_yyxx_crop'])
            comp_image_list.append(cv2.cvtColor(np.array(comp_img), cv2.COLOR_RGB2BGR))
        composite_images = np.stack(comp_image_list, axis=0)
        return composite_images
