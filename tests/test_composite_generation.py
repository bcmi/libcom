from common import *
from libcom import get_composite_image
import os
import cv2
import shutil

task_name = 'generate_composite'

if __name__ == '__main__':
    test_set = get_test_list()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name}...')
    for pair in test_set:
        fg_img, fg_mask = pair['foreground'], pair['foreground_mask']
        bg_img, bbox = pair['background'], pair['bbox']
        img_list = [bg_img, fg_img] 
        for option in ['none', 'gaussian', 'poisson']:
            comp_img, comp_mask = get_composite_image(fg_img, fg_mask, bg_img, bbox, option)
            img_list.append(comp_img)
            img_list.append(comp_mask)
        # visulization results
        img_name  = os.path.basename(fg_img).replace('.png', '.jpg')
        grid_img  = make_image_grid(img_list, cols=4)
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')
