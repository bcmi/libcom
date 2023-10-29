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
    def __init__(self, device=0, model_type='BargainNet', **kwargs):
        '''
        device: gpu id, type=str/torch.device
        model_type: predefined model type, type=str
        kwargs: other parameters for building model, type=dict
        '''
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        # insert your code here: modify the model name
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
        '''
        composite_image, composite_mask: type=str or numpy array or PIL.Image
        '''
        im, bg_mask, fg_mask = self.inputs_preprocess(composite_image, composite_mask)
        bg_style = self.model(im, bg_mask)
        fg_style = self.model(im, fg_mask)
        preds    = self.outputs_postprocess(bg_style, fg_style)
        return preds