from common import *
from libcom import Mure_ObjectStitchModel
import os
import cv2
import shutil
import torch

task_name = 'Mure_ObjectStitch'


if __name__ == '__main__':

    # collect pairwise test samples
    test_set = get_objectstitch_test_list()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name}...')

    # build your model
    net = Mure_ObjectStitchModel(device=0, sampler='plms')

    # infer
    for pair in test_set:
        bg_img, bbox = pair['background'], pair['bbox']
        fg_img, fg_mask = pair['foreground'], pair['foreground_mask']

        comp_img, show_fg_img = net(bg_img, fg_img, fg_mask, bbox, sample_steps=25, num_samples=5)
        img_name  = os.path.basename(bg_img).replace('.png', '.jpg')
        bbox_bg   = draw_bbox_on_image(bg_img, bbox)
        grid_img  = make_image_grid([bbox_bg, show_fg_img] + [comp_img[j] for j in range(len(comp_img))])
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')