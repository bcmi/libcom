import torch
import torchvision
from libcom.utils.model_download import download_pretrained_model
from libcom.utils.process_image import *
from libcom.utils.environment import *
import os
import torchvision.transforms as transforms
from .source.madis_net import *

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_set = ['IHDRNet'] 

class InharmoniousLocalizationModel:
    """
    Inharmonious region localization model.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type
        kwargs (dict): other parameters for building model

    Examples:
        >>> from libcom import InharmoniousLocalizationModel
        >>> import cv2
        >>> net = InharmoniousLocalizationModel(device=0)
        >>> comp_img1  = '../tests/source/composite/comp1_MadisNet.png'
        >>> inharmonious_localization1 = net(comp_img1)
        >>> comp_img2  = '../tests/source/composite/comp2_MadisNet.png'
        >>> inharmonious_localization2 = net(comp_img2)
        >>> cv2.imwrite('../docs/_static/image/inharmonious_localization_result1.jpg', np.concatenate([cv2.resize(cv2.imread(comp_img1),(256,256)), inharmonious_localization1],axis=1))
        >>> cv2.imwrite('../docs/_static/image/inharmonious_localization_result2.jpg', np.concatenate([cv2.resize(cv2.imread(comp_img2),(256,256)), inharmonious_localization2],axis=1))


    Expected result:

    .. image:: _static/image/inharmonious_localization_result3_4.jpg

    """
    def __init__(self, device=0, model_type='IHDRNet', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        weight_path_g = os.path.join(cur_dir, 'pretrained_models', 'Inharmonious_G.pth')
        download_pretrained_model(weight_path_g)
        weight_path_ihdrnet = os.path.join(cur_dir, 'pretrained_models', 'IHDRNet.pth')
        download_pretrained_model(weight_path_ihdrnet)
        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path_g, weight_path_ihdrnet)
        self.build_data_transformer()

    def build_pretrained_model(self, weight_path_g, weight_path_ihdrnet):
        model = MadisNet()
        model.g.load_state_dict(torch.load(weight_path_g)['state_dict'])
        model.ihdrnet.load_state_dict(torch.load(weight_path_ihdrnet)['state_dict'])
        self.MadisNet_model = model.to(self.device).eval()
        
    def build_data_transformer(self):
        self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
            ])
    
    def inputs_preprocess(self, composite_image):
        img = read_image_opencv(composite_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256,256))
        img = self.transformer(img).float().to(self.device).unsqueeze(0)
        return img
    
    def outputs_postprocess(self, outputs):
        if len(outputs.shape) == 4:
            outputs = outputs.squeeze(0)
        outputs = (torch.clamp(255.0 * outputs.permute(1, 2, 0), 0, 255)).cpu().numpy()
        outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        return outputs
    

    @torch.no_grad()
    def __call__(self, composite_image):
        """
        Given a composite image, predict the mask of the inharmonious region.

        Args:
            composite_image (str | numpy.ndarray): The path to composite image or the compposite image in ndarray form.

        Returns:
            inharmonious_mask (np.array): The inharmonious mask.
        
        """
        img = self.inputs_preprocess(composite_image)
        outputs = self.MadisNet_model(img)[0]
        preds = self.outputs_postprocess(outputs)
        return preds
