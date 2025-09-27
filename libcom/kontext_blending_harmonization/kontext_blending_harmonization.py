import os
import torch
from PIL import Image
import os
import numpy as np
from .source.pipeline import FluxKontextPipeline 
from diffusers import FluxTransformer2DModel
import cv2
from libcom.utils.model_download import download_pretrained_model

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_dir = os.environ.get('LIBCOM_MODEL_DIR', cur_dir)
model_set = ['Kontext_blend', 'Kontext_harm'] 

class KontextBlendingHarmonizationModel:
    """
    Flux Kontext based image blending and harmonization model.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type. "Kontext_blend" refers to the version fintuned on the image blending task. "Kontext_harm" refers to the version finetuned on the image harmonization task. default: "Kontext_blend"
        kwargs (dict): other parameters for building model
    
    Examples:
        >>> from libcom import KontextBlendingHarmonizationModel
        >>> from libcom.utils.process_image import make_image_grid, draw_bbox_on_image
        >>> import cv2

        >>> net = KontextBlendingHarmonizationModel(device=0, model_type="Kontext_blend")
        >>> img_names = ["000000049931.png", "000000460450.png", "6c5601278dcb5e6d_m09728_f5cd2891_17.png"]
        >>> bboxes = [[168, 137, 488, 413], [134, 158, 399, 511], [130, 91, 392, 271]]
        >>> test_dir  = 'tests/controllable_composition/'

        >>> for i in range(len(img_names)):
        >>>     bg_img  = test_dir + 'background/' + img_names[i]
        >>>     fg_img  = test_dir + 'foreground/' + img_names[i]
        >>>     bbox    = bboxes[i]
        >>>     mask    = test_dir + 'foreground_mask/' + img_names[i]
        >>>     comp    = net(bg_img, fg_img, bbox, mask)
        >>>     bg_img  = draw_bbox_on_image(bg_img, bbox)
        >>>     grid_img = make_image_grid([bg_img, fg_img, comp[0]])
        >>>     cv2.imwrite('../docs/_static/image/kontext_result{}.jpg'.format(i+1), grid_img)

    Expected result:

    .. image:: _static/image/kontext_result1.jpg
        :scale: 21 %
    .. image:: _static/image/kontext_result2.jpg
        :scale: 21 %
 
    """

    def __init__(self, device=0, model_type='Kontext_blend', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        
        weight_path = "black-forest-labs/FLUX.1-Kontext-dev"
        blending_lora_path = os.path.join(cur_dir, 'pretrained_models', 'flux_kontext_blending.safetensors')
        harmonization_lora_path = os.path.join(cur_dir, 'pretrained_models', 'flux_kontext_harmonization.safetensors')
        download_pretrained_model(blending_lora_path)
        download_pretrained_model(harmonization_lora_path)
        
        if model_type == 'Kontext_blend':
            lora_path = blending_lora_path
        elif model_type == 'Kontext_harm':
            lora_path = harmonization_lora_path
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.device = f"cuda:{device}"
        self.build_pretrained_model(weight_path, lora_path)

    def build_pretrained_model(self, weight_path, lora_path):
        transformer = FluxTransformer2DModel.from_pretrained(weight_path, subfolder="transformer")
        transformer.requires_grad_(False)
        self.pipeline = FluxKontextPipeline.from_pretrained(
            weight_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.load_lora_weights(lora_path)
        self.pipeline = self.pipeline.to(self.device, dtype=torch.bfloat16)
        
        
    def generate_initial_image(self, bg_path, fg_path, fg_mask_path, bbox):
        bg_img = Image.open(bg_path).convert('RGB')
        fg_img = Image.open(fg_path).convert('RGB')

        if fg_mask_path is not None:
            fg_mask = Image.open(fg_mask_path).convert('L')  # 灰度
            fg_mask = fg_mask.resize((fg_img.width, fg_img.height))
            fg_img_np   = np.array(fg_img)
            fg_mask_np  = np.array(fg_mask)
            # mask>127认为是物体，其他是背景 -> 背景改成白色
            white = np.ones_like(fg_img_np) * 255
            fg_img_np = np.where(fg_mask_np[..., None] > 127, fg_img_np, white)

            fg_img = Image.fromarray(fg_img_np)
        # 按照bbox缩放前景
        x1, y1, x2, y2 = [int(v) for v in bbox]
        fg_resized = fg_img.resize((x2 - x1, y2 - y1))
        # 把前景贴到背景上
        bg_img.paste(fg_resized, (x1, y1, x2, y2))

        return bg_img
        
    
    def inputs_preprocess(self, background_image, foreground_image, bbox, foreground_mask):
        image = self.generate_initial_image(background_image, foreground_image, foreground_mask, bbox)
        return image
    
    @torch.no_grad()
    def __call__(self, background_image, foreground_image, bbox, 
                 foreground_mask=None, prompt='put it here', 
                 num_samples=1, sample_steps=28, guidance_scale=2.5, seed=321):
        """
        Kontext based image blending and harmonization.

        Args:
            background_image (str): The path to background image.
            foreground_image (str): The path to foreground image.
            bbox (list): The bounding box which indicates the foreground's location in the background. [x1, y1, x2, y2].
            foreground_mask (None | str): Mask of foreground image which indicates the foreground object region in the foreground image. default: None.
            prompt (str): The text prompt to guide the image generation. default: 'put it here'.
            num_samples (int): Number of images to be generated for each task. default: 1.
            sample_steps (int): Number of denoising steps. The recommended setting is 28 for FlowMatchEulerDiscreteScheduler. default: 28.
            guidance_scale (int): Scale in classifier-free guidance (minimum: 1; maximum: 20). default: 2.5.
            seed (int): Random Seed is used to reproduce results and same seed will lead to same results. 

        Returns:
            composite_images (numpy.ndarray): Generated images with a shape of 512x512x3 or Nx512x512x3, where N indicates the number of generated images. 
        """
        image = self.inputs_preprocess(background_image, foreground_image, bbox, foreground_mask)
        comp_image_list = []
        for i in range(num_samples):
            comp_img = self.pipeline(
                image=image,
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=guidance_scale,
                num_inference_steps=sample_steps,
                max_sequence_length=512,
            ).images[0]
            comp_image_list.append(cv2.cvtColor(np.array(comp_img), cv2.COLOR_RGB2BGR))
        composite_images = np.stack(comp_image_list, axis=0)
        return composite_images