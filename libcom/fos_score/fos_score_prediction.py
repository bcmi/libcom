import torch
import torchvision
from libcom.utils.model_download import download_pretrained_model
from libcom.utils.process_image import *
from libcom.utils.environment import *
from libcom.fos_score.source.config import Config
from libcom.fos_score.source.networks import StudentModel, SingleScaleD
import torch 
import os
import math
import datetime
import torchvision.transforms as transforms
from torchvision.utils import save_image
from einops import rearrange
cur_dir   = os.path.dirname(os.path.abspath(__file__))

model_set = ['FOS_D', 'FOS_E'] 

class FOSScoreModel:
    """
    Foreground object search score prediction model.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type
        kwargs (dict): other parameters for building model
    
    Examples:
        >>> from libcom.utils.process_image import make_image_grid
        >>> from libcom import FOSScoreModel
        >>> import cv2
        >>> MODEL_TYPE = 'FOS_D' # choose from 'FOS_D', 'FOS_E'
        >>> background = '../tests/source/background/f80eda2459853824_m09g1w_b2413ec8_11.png'
        >>> fg_bbox    = [175, 82, 309, 310] # x1,y1,x2,y2
        >>> foreground = '../tests/source/foreground/f80eda2459853824_m09g1w_b2413ec8_11.png'
        >>> foreground_mask = '../tests/source/foreground_mask/f80eda2459853824_m09g1w_b2413ec8_11.png'
        >>> net        = FOSScoreModel(device=0, model_type=MODEL_TYPE)
        >>> score      = net(background_image, foreground_image, bounding_box, foreground_mask=foreground_mask)
        >>> grid_img   = make_image_grid([background_image, foreground_image, composite_image], text_list=[f'fos_score:{score:.2f}'])
        >>> cv2.imshow('fos_score_demo', grid_img)

    Expected result:

    .. image:: _static/image/fos_score_result1.jpg
        :scale: 50 %

            
    """
    def __init__(self, device=0, model_type='FOS_D', **kwargs):
        
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        self.IMAGE_NET_MEAN = [0.5, 0.5, 0.5]
        self.IMAGE_NET_STD  = [0.5, 0.5, 0.5]
        config_file = os.path.join(cur_dir, 'source/config/config_rfosd.yaml')
        self.cfg    = Config(config_file)
        weight_path = os.path.join(cur_dir, 'pretrained_models', '{}.pth'.format(self.model_type))
        download_pretrained_model(weight_path)
        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, weight_path):
        """
        Build pretrained model from path of weight.
        """
        if self.model_type == "FOS_E":
            model = StudentModel(self.cfg)
            model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
            self.model = model.to(self.device).eval()
        else:
            model = SingleScaleD(False)
            model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
            self.model = model.to(self.device).eval()
        
    def build_data_transformer(self):
        self.image_size = self.cfg.image_size
        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGE_NET_MEAN, std=self.IMAGE_NET_STD)
        ])
    
    def inputs_preprocess(self, background_image, foreground_image, foreground_mask, bbox):
        bg  = cv2.cvtColor(read_image_opencv(background_image), cv2.COLOR_BGR2RGB)
        bg_h, bg_w, _ = bg.shape
        fg = cv2.cvtColor(read_image_opencv(foreground_image), cv2.COLOR_BGR2RGB)
        if foreground_mask is not None:
            fg_mask = cv2.cvtColor(read_image_opencv(foreground_mask), cv2.COLOR_BGR2GRAY)
            assert fg.shape[:2] == fg_mask.shape
            fg[np.where(fg_mask != 255)] = 255
        if self.model_type == 'FOS_D':
            x1, y1, x2, y2 = bbox
            rs_fg = cv2.resize(fg.copy(), (x2 - x1, y2 - y1))
            bg[y1:y2, x1:x2] = rs_fg
            x1, y1, x2, y2 = self.get_crop_bbox(bbox, bg_h, bg_w)
            scale_comp = bg.copy()
            scale_comp_r = Image.fromarray(scale_comp, mode="RGB")
            return scale_comp_r

        return bg, fg
    
    def get_crop_bbox(self, bbox, bg_h, bg_w):
        x1, y1, x2, y2 = bbox
        ori_tar_w = x2 - x1
        ori_tar_h = y2 - y1
        add_w = int(ori_tar_w * (math.sqrt(2) - 1) / 2)
        add_h = int(ori_tar_h * (math.sqrt(2) - 1) / 2)
        new_y1 = max(0, y1 - add_h)
        new_y2 = min(bg_h, y2 + add_h)
        new_x1 = max(0, x1 - add_w)
        new_x2 = min(bg_w, x2 + add_w)
        return new_x1, new_y1, new_x2, new_y2
    
    def prepare_input_encoders(self, background_image, foreground_image, bounding_box):
        background_image = fill_box_with_specified_pixel(background_image, bounding_box, self.cfg.fill_pixel)
        bg_t = self.transformer(background_image)

        bg_ori_w, bg_ori_h = background_image.size
        ori_x1, ori_y1 = bounding_box[0:2]
        ori_x2, ori_y2 = bounding_box[2:4]
        query_box = torch.tensor([ori_x1 / bg_ori_w, ori_y1 / bg_ori_h, ori_x2 / bg_ori_w, ori_y2 / bg_ori_h]) * self.cfg.image_size
        query_box = torch.round(query_box)
        query_box = query_box.float()

        ori_tar_w = ori_x2 - ori_x1
        ori_tar_h = ori_y2 - ori_y1

        add_w = int(ori_tar_w * (math.sqrt(2) - 1) / 2)
        add_h = int(ori_tar_h * (math.sqrt(2) - 1) / 2)
        new_y1 = max(0, ori_y1 - add_h)
        new_y2 = min(bg_ori_h, ori_y2 + add_h)
        new_x1 = max(0, ori_x1 - add_w)
        new_x2 = min(bg_ori_w, ori_x2 + add_w)
        new_box = torch.tensor([new_x1 / bg_ori_w, new_y1 / bg_ori_h, new_x2 / bg_ori_w, new_y2 / bg_ori_h]) * self.cfg.image_size
        new_box = torch.round(new_box)
        new_box = new_box.float()

        pad_fg = padding_to_square(foreground_image.copy(), self.cfg.pad_pixel)
        pad_fg = Image.fromarray(pad_fg)
        fg_t = self.transformer(pad_fg)

        sample = {
            'bg': bg_t.unsqueeze(0),
            'fg': fg_t.unsqueeze(0),
            'query_box': query_box.unsqueeze(0),
            'crop_box': new_box.unsqueeze(0),
        }
        return sample

    def prepare_input_disc(self, composite_image):
        comp = self.transformer(composite_image).unsqueeze(0)
        return comp
    
    def preprocess_image(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
        image = rearrange(image, "h w c -> 1 c h w")
        image = image.to(self.device)
        return image
    
    @torch.no_grad()
    def __call__(
            self,
            background_image,
            foreground_image,
            bounding_box,
            foreground_mask=None
            ):
        """
        Predicting the compatibility score between the given background and the given foreground. Called in __call__ function.

        Args:
            background_image (str | numpy.ndarray): The path to background image or the background image in ndarray form.
            foreground_image (str | numpy.ndarray): The path to foreground image or the background image in ndarray form.
            bounding_box (list): The bounding box which indicates the foreground's location in the background. [x1, y1, x2, y2].
            foreground_mask (str | numpy.ndarray): Mask of foreground image which indicates the foreground object region in the foreground image. default: None.
        
        Returns:
            fos_score (float): Predicted compatibility score between the given background image and the given foreground image.

        """
        if self.model_type == 'FOS_E':
            background_image, foreground_image = self.inputs_preprocess(background_image, foreground_image, foreground_mask, bounding_box)
            sample = self.prepare_input_encoders(background_image, foreground_image, bounding_box)
            bg_im = sample['bg'].to(self.device)
            fg_im = sample['fg'].to(self.device)
            q_box  = sample['query_box'].to(self.device)
            c_box  = sample['crop_box'].to(self.device)
            output = self.model(bg_im, fg_im, q_box, c_box)[-1].item()
        else:
            composite_image = self.inputs_preprocess(background_image, foreground_image, foreground_mask, bounding_box)
            composite_image = self.prepare_input_disc(composite_image).to(self.device)
            _, score = self.model(composite_image)
            output = score[-1].item()
        return output
    
def fill_box_with_specified_pixel(bg_im, query_box, fill_value):
    x1, y1 = query_box[0:2]
    x2, y2 = query_box[2:4]
    bg_im = np.array(bg_im)
    bg_im[y1:y2, x1:x2] = fill_value
    bg_im = Image.fromarray(bg_im)
    return bg_im

def padding_to_square(src_img, pad_pixel=255):
    src_h, src_w = src_img.shape[:2]
    if src_h == src_w:
        return src_img
    if src_w > src_h:
        pad_w = 0
        pad_h = src_w - src_h
    else:
        pad_w = src_h - src_w
        pad_h = 0

    pad_y1 = int(pad_h // 2)
    pad_y2 = int(pad_h - pad_y1)
    pad_x1 = int(pad_w // 2)
    pad_x2 = int(pad_w - pad_x1)

    if len(src_img.shape) == 3:
        pad_im = np.pad(src_img, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0,0)),
                        'constant', constant_values=pad_pixel)
    else:
        pad_im = np.pad(src_img, ((pad_y1, pad_y2), (pad_x1, pad_x2)),
                        'constant', constant_values=pad_pixel)
    return pad_im