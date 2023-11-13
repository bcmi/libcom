from common import *
from libcom import FOSScoreModel
import os
import cv2
import shutil
import torch

task_name = 'fos_score_prediction'
MODEL_TYPE = 'FOS_D' # choose from 'FOS_D', 'FOS_E'

if __name__ == '__main__':
    test_set = get_test_list()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name, MODEL_TYPE)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name}...')
    net = FOSScoreModel(device=0, model_type=MODEL_TYPE)
    for pair in test_set:
        background_image, foreground_image, foreground_mask, composite_image, bounding_box = pair['background'], pair['foreground'], pair['foreground_mask'], pair['composite'], pair['bbox']
        score = net(background_image, foreground_image, bounding_box, foreground_mask=foreground_mask)
        img_name  = os.path.basename(background_image).replace('.png', '.jpg')
        grid_img  = make_image_grid([background_image, foreground_image, composite_image], text_list=[f'fos_score:{score:.2f}'])
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')