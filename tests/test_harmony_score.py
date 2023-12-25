from common import *
from libcom import HarmonyScoreModel
import os
import cv2
import shutil
import torch

task_name = 'harmony_score_prediction'

if __name__ == '__main__':
    test_set = get_test_list_harmony_prediction()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name}...')
    net = HarmonyScoreModel(device=0, model_type='BargainNet')
    for pair in test_set:
        comp_img, comp_mask = pair['composite'], pair['composite_mask']
        score     = net(comp_img, comp_mask)
        if '_harm' in comp_img and score < 0.7:
            continue
        elif '_inharm' in comp_img and score > 0.3:
            continue
        img_name  = os.path.basename(comp_img).replace('.png', '.jpg')
        grid_img  = make_image_grid([comp_img, comp_mask], 
                                    text_list=[f'harmony_score:{score:.2f}', 'composite-mask'])
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')
    