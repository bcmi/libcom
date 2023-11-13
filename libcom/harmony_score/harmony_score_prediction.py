import torch
import torchvision
from libcom.utils.model_download import download_pretrained_model
from libcom.utils.process_image import *
from libcom.utils.environment import *
from libcom.harmony_score.source.bargainnet import StyleEncoder
import torch 
import os
import torchvision.transforms as transforms
import math

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_set = ['BargainNet'] 

class HarmonyScoreModel:
    """
    Foreground object search score prediction model.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type.
        kwargs (dict): other parameters for building model
    
    Examples:
        >>> from libcom import HarmonyScoreModel
        >>> from libcom.utils.process_image import make_image_grid
        >>> import cv2
        >>> net = HarmonyScoreModel(device=0, model_type='BargainNet')
        >>> test_dir   = '../tests/harmony_score_prediction/'
        >>> img_names  = ['vaulted-cellar-247391_inharm.jpg', 'ameland-5651866_harm.jpg']
        >>> vis_list,scores = [], []
        >>> for img_name in img_names:
        >>>     comp_img  = test_dir + 'composite/' + img_name
        >>>     comp_mask = test_dir + 'composite_mask/' + img_name
        >>>     score     = net(comp_img, comp_mask)
        >>>     vis_list += [comp_img, comp_mask]
        >>>     scores.append(score)
        >>> grid_img  = make_image_grid(vis_list, text_list=[f'harmony_score:{scores[0]:.2f}', 'composite-mask', f'harmony_score:{scores[1]:.2f}', 'composite-mask'])
        >>> cv2.imwrite('../docs/_static/image/harmonyscore_result1.jpg', grid_img)
 
    Expected result:

    .. image:: _static/image/harmonyscore_result1.jpg
        :scale: 38 %

    """
    def __init__(self, device=0, model_type='BargainNet', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        weight_path = os.path.join(cur_dir, 'pretrained_models', 'BargainNet.pth')
        download_pretrained_model(weight_path)
        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, weight_path):
        model = StyleEncoder(style_dim=16)
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model = model.to(self.device).eval()
        
    def build_data_transformer(self):
        self.image_size = 256
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def inputs_preprocess(self, composite_image, composite_mask):
        image = read_image_opencv(composite_image)
        mask  = read_mask_opencv(composite_mask)
        fg_mask = mask.astype(np.float32) / 255.
        bg_mask = 1 - fg_mask
        fg_mask = self.mask_transform(Image.fromarray(fg_mask))
        fg_mask = fg_mask.unsqueeze(0).to(self.device)
        bg_mask = self.mask_transform(Image.fromarray(bg_mask))
        bg_mask = bg_mask.unsqueeze(0).to(self.device)

        image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image   = self.transform(Image.fromarray(image))
        image   = image.unsqueeze(0).to(self.device)
        return image, bg_mask, fg_mask
    
    def outputs_postprocess(self, bg_style, fg_style):
        eucl_dist = self.Euclidean_distance(bg_style, fg_style)
        # convert distance to harmony level which lies in 0 and 1
        harm_level = math.exp(-0.04212 * eucl_dist)
        return harm_level
    
    def Euclidean_distance(self, vec1, vec2):
        vec1 = vec1.cpu().numpy()
        vec2 = vec2.cpu().numpy()
        dist = np.sqrt(np.sum((vec1 - vec2)**2))
        return dist
    
    @torch.no_grad()
    def __call__(self, composite_image, composite_mask):
        """
        Predicting the compatibility score between background and foreground in the given composite image.

        Args:
            composite_image (str | numpy.ndarray): The path to composite image or the compposite image in ndarray form.
            composite_mask (str | numpy.ndarray): Mask of composite image which indicates the foreground object region in the composite image.
        
        Returns:
            harmony_score (float): Predicted harmony score within [0,1] between background region and foreground region of the given composite image. Larger harmony score implies more harmonious composite image.

        """
        im, bg_mask, fg_mask = self.inputs_preprocess(composite_image, composite_mask)
        bg_style = self.model(im, bg_mask)
        fg_style = self.model(im, fg_mask)
        preds    = self.outputs_postprocess(bg_style, fg_style)
        return preds