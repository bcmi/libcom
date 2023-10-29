from common import *
from libcom import HarmonyScoreModel
import os
import cv2
import shutil
import torch

# change to your task name
task_name = 'harmony_score_prediction'

if __name__ == '__main__':
    # collect pairwise test samples
    test_set = get_test_list()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name}...')
    net = HarmonyScoreModel(device=0, model_type='BargainNet')
    for pair in test_set:
        # change to your inputs
        comp_img, comp_mask = pair['composite'], pair['composite_mask']
        score = net(comp_img, comp_mask)
        img_name  = os.path.basename(comp_img).replace('.png', '.jpg')
        # visualize source image and resultant images or predicted score.
        # see examples in tests/results/opa_score_prediction and tests/results/generate_composite
        grid_img  = make_image_grid([comp_img, comp_mask], 
                                    text_list=[f'harmony_score:{score:.2f}', 'composite-mask'])
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')