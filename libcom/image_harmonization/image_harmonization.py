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
    def __init__(self, device=0, model_type='PCTNet', **kwargs):
        '''
        device: gpu id, type=str/torch.device
        model_type: predefined model type, type=str
        kwargs: other parameters for building model, type=dict
        '''
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
        '''
        composite_image, composite_mask: type=str or numpy array or PIL.Image
        '''
        # insert your code here: define inference pipeline
        img, mask, img_lr, mask_lr = self.inputs_preprocess(composite_image, composite_mask)
        if self.model_type == "CDTNet":
            outputs = self.model(img, mask)
        else:
            outputs = self.model(img_lr, img, mask_lr, mask)
        preds = self.outputs_postprocess(outputs)
        return preds