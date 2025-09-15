import csv
import random
import shutil
import torch
from libcom.utils.model_download import download_pretrained_model
from libcom.utils.process_image import *
from libcom.utils.environment import *
import torch 
import os
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import libcom.fopa_heat_map.source.network.ObPlaNet_simple as network
from libcom.fopa_heat_map.source.data.OBdataset import make_composite_PIL
from libcom.fopa_heat_map.source.prepare_multi_fg_scales import prepare_multi_fg_scales
from libcom.fopa_heat_map.source.data.all_transforms import Compose, JointResize

cur_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.environ.get('LIBCOM_MODEL_DIR',cur_dir)
model_set = ['fopa'] 


class FOPAHeatMapModel:
    """
    Generate a heatmap for a pair of scaled foreground and background.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type

    Examples:
        >>> test_set = get_test_list_fopa_heatmap()
        >>> result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
        >>> if os.path.exists(result_dir):
        >>>     shutil.rmtree(result_dir)
        >>> os.makedirs(result_dir, exist_ok=True)
        >>> os.makedirs(os.path.join(result_dir, 'grid'), exist_ok=True)
        >>> print(f'begin testing {task_name}...')
        >>> net = FOPAHeatMapModel(device=0)
        >>> for pair in test_set[:1]:
        >>>     fg_img, fg_mask, bg_img = pair['foreground'], pair['foreground_mask'], pair['background']
        >>>     bboxes, heatmaps = net(fg_img, fg_mask, bg_img, cache_dir=os.path.join(result_dir, 'cache'), heatmap_dir=os.path.join(result_dir, 'heatmap'))
        >>>     img_name  = os.path.basename(bg_img).replace('.png', '.jpg')
        >>>     grid_img  = make_image_grid([bg_img, fg_img, heatmaps[0]])
        >>>     res_path  = os.path.join(result_dir, 'grid', img_name)
        >>>     cv2.imwrite(res_path, grid_img)
        >>>     print('save result to ', res_path)
        >>> print(f'end testing {task_name}!') 

    Expected result:

    .. image:: _static/image/fopa_heatmap_FOPA_result1.png
        :scale: 72 %

    .. image:: _static/image/fopa_heatmap_FOPA_result3.png
        :scale: 52 %
        
    """
    def __init__(self, device=0, model_type='fopa', **kwargs):
        
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        self.device = check_gpu_device(device)
        fopa_weight = os.path.join(model_dir, 'pretrained_models', 'FOPA.pth')
        download_pretrained_model(fopa_weight)
        sopa_weight = os.path.join(model_dir, 'pretrained_models', 'SOPA.pth')
        download_pretrained_model(sopa_weight)
        self.build_pretrained_model(sopa_weight, fopa_weight)
        self.build_data_transformer()
        
    def build_pretrained_model(self, sopa_weight, fopa_weight):
        model = getattr(network, "ObPlaNet_resnet18")(pretrained=False, weight_path=sopa_weight).to(self.device)
        model.load_state_dict(torch.load(fopa_weight, map_location='cpu'))
        self.model = model.to(self.device).eval()
        
    def build_data_transformer(self):
        self.triple_transform = Compose([JointResize(256)])
        self.image_transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]
        )
        self.mask_transform = transforms.ToTensor()
    
    def generate_heatmap(self, background_image, csv_path, scaled_fg_dir, scaled_mask_dir, heatmap_dir):
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_list = []
        with open(csv_path, mode='r', newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                fg_name   = '{}_{}_{}_{}.jpg'.format(row["fg_name"].split(".")[0],row["bg_name"].split(".")[0],int(row["newWidth"]),int(row["newHeight"]))
                mask_name = '{}_{}_{}_{}.jpg'.format(row["fg_name"].split(".")[0],row["bg_name"].split(".")[0],int(row["newWidth"]),int(row["newHeight"]))
                scale     = row['scale']
                save_name = fg_name.split(".")[0] + '_' + str(scale) + '.jpg'
                
                bg_img    = read_image_pil(background_image)
                fg_img    = read_image_pil(os.path.join(scaled_fg_dir, fg_name))
                mask      = read_mask_pil(os.path.join(scaled_mask_dir, mask_name))
                bg_t, fg_t, mask_t = self.triple_transform(bg_img, fg_img, mask)
                mask_t    = self.mask_transform(mask_t).to(self.device)
                fg_t      = self.image_transform(fg_t).to(self.device)
                bg_t      = self.image_transform(bg_t).to(self.device)
                
                outs, _   = self.model(bg_t.unsqueeze(0), fg_t.unsqueeze(0), mask_t.unsqueeze(0), 'test') 
                outs      = torch.softmax(outs, dim=1)[:,1,:,:]
                outs      = transforms.ToPILImage()(outs)
                out_file  = os.path.join(heatmap_dir, save_name)
                outs.save(out_file) 
                heatmap_list.append(out_file)
        return heatmap_list
    
    def generate_bounding_box(self, foreground_image, foreground_mask, background_image, csv_path,
                              cache_dir, heatmap_dir, fg_scale_num, composite_num_choose, composite_num):
        icount = 0
        with open(csv_path, mode='r', newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                scale = row['scale']
                w = int(row['newWidth'])
                h = int(row['newHeight'])
                save_name = row['fg_name'].split(".")[0] + '_' + str(scale) + '.jpg'
                fg_name = '{}_{}_{}_{}.jpg'.format(row["fg_name"].split(".")[0],row["bg_name"].split(".")[0],int(row["newWidth"]),int(row["newHeight"]))
                save_name = fg_name.split(".")[0] + '_' + str(scale) + '.jpg'
            
                if icount == 0:
                    bg_img     = read_image_pil(background_image)  
                    fg_tocp    = read_image_pil(foreground_image)
                    mask_tocp  = read_image_pil(foreground_mask)
                    
                    composite_dir = os.path.join(cache_dir, f'{fg_scale_num}scales_composite', f'{row["fg_name"].split(".")[0]}_{row["bg_name"].split(".")[0]}')
                    os.makedirs(composite_dir, exist_ok=True)
                    heatmap_center_list = []
                    fg_size_list = []
                
                icount += 1
                heatmap = Image.open(os.path.join(heatmap_dir, save_name))
                heatmap = np.array(heatmap)
                heatmap_center = np.zeros_like(heatmap, dtype=np.float64)
                hb = int(h / bg_img.height * heatmap.shape[0] / 2)
                wb = int(w / bg_img.width  * heatmap.shape[1] / 2)
                heatmap_center[hb:-hb, wb:-wb] = heatmap[hb:-hb, wb:-wb]
                heatmap_center_list.append(heatmap_center)
                fg_size_list.append((h,w))
                
                if icount == fg_scale_num:
                    icount = 0
                    heatmap_center_stack = np.stack(heatmap_center_list)
                    sorted_indices = np.argsort(-heatmap_center_stack, axis=None)
                    sorted_indices = np.unravel_index(sorted_indices, heatmap_center_stack.shape)
                    
                    for i in range(composite_num):
                        iscale, y_, x_ = sorted_indices[0][i], sorted_indices[1][i], sorted_indices[2][i]
                        h, w = fg_size_list[iscale]
                        x_   = x_/heatmap.shape[1]*bg_img.width
                        y_   = y_/heatmap.shape[0]*bg_img.height
                        x    = int(x_ - w / 2)
                        y    = int(y_ - h / 2)

                        composite_img, composite_msk = make_composite_PIL(fg_tocp, mask_tocp, bg_img, [x, y, w, h], return_mask=True)
                        save_img_path = os.path.join(composite_dir, f'{row["fg_name"].split(".")[0]}_{row["bg_name"].split(".")[0]}_{x}_{y}_{w}_{h}.jpg')
                        save_msk_path = os.path.join(composite_dir, f'{row["fg_name"].split(".")[0]}_{row["bg_name"].split(".")[0]}_{x}_{y}_{w}_{h}.png')
                        composite_img.save(save_img_path)
                        composite_msk.save(save_msk_path)

        source_folder = composite_dir
        composite_dir_choose = os.path.join(cache_dir, f'{fg_scale_num}scales_composite_RandomSelect_{composite_num_choose}', f'{row["fg_name"].split(".")[0]}_{row["bg_name"].split(".")[0]}')
        os.makedirs(composite_dir_choose, exist_ok=True)
        
        bbox_list = []
        image_files = [f for f in os.listdir(composite_dir) if f.endswith((".jpg"))]
        selected_images = random.sample(image_files, composite_num_choose)
        for image in selected_images:
            source_path = os.path.join(source_folder, image)
            destination_path = os.path.join(composite_dir_choose, image)
            png_source_path = source_path.replace(".jpg", ".png")
            destination_path_png = os.path.join(composite_dir_choose, image.replace(".jpg", ".png"))
            shutil.copyfile(source_path, destination_path)
            shutil.copyfile(png_source_path, destination_path_png)
            bbox_list_parts = os.path.splitext(os.path.split(source_path)[1])[0].split('_')
            x_box, y_box, w_box, h_box = int(bbox_list_parts[2]), int(bbox_list_parts[3]), int(bbox_list_parts[4]), int(bbox_list_parts[5])
            last_four_numbers = [x_box, y_box, w_box, h_box]
            bbox_list.append(last_four_numbers)
        return bbox_list


    @torch.no_grad()
    def __call__(self, foreground_image, foreground_mask, background_image, cache_dir, heatmap_dir,
                 fg_scale_num=16, composite_num_choose=3, composite_num=50):
        """
        Generate a heatmap for a pair of scaled foreground and background.
        
        Args:
            foreground_image: foreground image path
            foreground_mask: foreground mask path
            background_image: background image path
            cache_dir: folder path where scaled foreground images, scaled mask images and composite images are stored
            heatmap_dir: folder path where heatmaps are stored
            fg_scale_num: number of scales of scaled foreground images and mask images
            composite_num_choose: the number of chosen composite images
            composite_num: the number of composite images with the highest score

        Returns:
            box_list: the path of concatenated background image, foreground image and corresponding heatmap
            heatmap_list: the path of heatmaps

        """
        scaled_fg_dir, scaled_mask_dir, csv_file = prepare_multi_fg_scales(cache_dir, foreground_image, foreground_mask, background_image, fg_scale_num)
        heatmap_list = self.generate_heatmap(background_image, csv_file, 
                                             scaled_fg_dir, scaled_mask_dir, heatmap_dir)
        box_list     = self.generate_bounding_box(foreground_image, foreground_mask, background_image, csv_file,
                                                  cache_dir, heatmap_dir, fg_scale_num, composite_num_choose, composite_num)
        return box_list, heatmap_list
