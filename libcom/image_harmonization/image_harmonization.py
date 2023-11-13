import torch
import torchvision
from libcom.utils.model_download import download_pretrained_model
from libcom.utils.process_image import *
from libcom.utils.environment import *
import os
import torchvision.transforms as transforms
from .source.pct_net import *
from .source.cdt_net import *

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_set = ['PCTNet', 'CDTNet'] 

class ImageHarmonizationModel:
    """
    Image harmonization model.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type, 'PCTNet' or 'CDTNet'
        kwargs (dict): other parameters for building model

    Examples:
        >>> from libcom import ImageHarmonizationModel
        >>> import cv2
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image

        >>> #Use CDTNet
        >>> CDTNet = ImageHarmonizationModel(device=0, model_type='CDTNet')
        >>> comp_img1  = '../tests/source/composite/comp1_CDTNet.jpg'
        >>> comp_mask1 = '../tests/source/composite_mask/mask1_CDTNet.png'
        >>> CDT_result1 = CDTNet(comp_img1, comp_mask1)
        >>> cv2.imwrite('../docs/_static/image/image_harmonization_CDT_result1.jpg', np.concatenate([cv2.imread(comp_img1), cv2.imread(comp_mask1), CDT_result1],axis=1))
        >>> comp_img2  = '../tests/source/composite/comp2_CDTNet.jpg'
        >>> comp_mask2 = '../tests/source/composite_mask/mask2_CDTNet.png'
        >>> CDT_result2 = CDTNet(comp_img2, comp_mask2)
        >>> cv2.imwrite('../docs/_static/image/image_harmonization_CDT_result2.jpg', np.concatenate([cv2.imread(comp_img2), cv2.imread(comp_mask2), CDT_result2],axis=1))
        >>> #Use PCTNet
        >>> PCTNet = ImageHarmonizationModel(device=0, model_type='PCTNet')
        >>> comp_img1  = '../tests/source/composite/comp1_PCTNet.jpg'
        >>> comp_mask1 = '../tests/source/composite_mask/mask1_PCTNet.png'
        >>> PCT_result1 = PCTNet(comp_img1, comp_mask1)
        >>> cv2.imwrite('../docs/_static/image/image_harmonization_PCT_result1.jpg', np.concatenate([cv2.imread(comp_img1), cv2.imread(comp_mask1), PCT_result1],axis=1))
        >>> comp_img2  = '../tests/source/composite/comp2_PCTNet.jpg'
        >>> comp_mask2 = '../tests/source/composite_mask/mask2_PCTNet.png'
        >>> PCT_result2 = PCTNet(comp_img2, comp_mask2)
        >>> cv2.imwrite('../docs/_static/image/image_harmonization_PCT_result2.jpg', np.concatenate([cv2.imread(comp_img2), cv2.imread(comp_mask2), PCT_result2],axis=1))

    Expected result:

    .. image:: _static/image/image_harmonization_CDT_result1.jpg
    .. image:: _static/image/image_harmonization_CDT_result2.jpg
    .. image:: _static/image/image_harmonization_PCT_result1.jpg
    .. image:: _static/image/image_harmonization_PCT_result2.jpg

    """
    def __init__(self, device=0, model_type='PCTNet', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        if self.model_type == 'CDTNet':
            weight_path = os.path.join(cur_dir, 'pretrained_models', 'CDTNet.pth')
        else:
            weight_path = os.path.join(cur_dir, 'pretrained_models', 'PCTNet.pth')
        download_pretrained_model(weight_path)
        lut_path = os.path.join(cur_dir, 'pretrained_models', 'IdentityLUT33.txt')
        download_pretrained_model(lut_path)
        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, weight_path):
        if self.model_type == 'CDTNet':
            model = CDTNet()
        else:
            model = PCTNet()
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
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
    def __call__(self, composite_image, composite_mask):
        """
        Given a composite image and a foreground mask, perform harmonization on the foreground.

        Args:
            composite_image (str | numpy.ndarray): The path to composite image or the compposite image in ndarray form.
            composite_mask (str | numpy.ndarray): Mask of composite image which indicates the foreground object region in the composite image.

        Returns:
            harmonized_image (np.array): The harmonized result.
        
        """
        img, mask, img_lr, mask_lr = self.inputs_preprocess(composite_image, composite_mask)
        if self.model_type == "CDTNet":
            outputs = self.model(img, mask)
        else:
            outputs = self.model(img_lr, img, mask_lr, mask)
        preds = self.outputs_postprocess(outputs)
        return preds