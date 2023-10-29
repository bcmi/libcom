from common import *
# change to your model
from libcom import PainterlyHarmonizationModel
import os
import cv2
import shutil
import torch

# change to your task name
task_name = 'painterly_image_harmonization'

if __name__ == '__main__':
    # collect pairwise test samples
    test_set = get_test_list_painterly_harmonization()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name}...')
    phd_net = PainterlyHarmonizationModel(device=0, model_type='PHDNet')
    phdiff  = PainterlyHarmonizationModel(device=0, model_type='PHDiffusion', use_residual=False)
    phdiffres = PainterlyHarmonizationModel(device=0, model_type='PHDiffusion')
    for pair in test_set:
        comp_img, comp_mask = pair['composite'], pair['composite_mask']
        phdnet_img = phd_net(comp_img, comp_mask)
        phdiff_img = phdiff(comp_img, comp_mask, sample_steps=25)
        phdiffres_img = phdiffres(comp_img, comp_mask, sample_steps=25)
        
        img_name  = os.path.basename(comp_img).replace('.png', '.jpg')
        grid_img  = make_image_grid([comp_img, comp_mask, phdnet_img, phdiff_img, phdiffres_img])
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')