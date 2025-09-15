from common import *
from libcom import ShadowGenerationModel
import os
import cv2
import shutil

task_name = 'shadow_generation'


def get_test_list_shadow_generation():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shadow_generation')
    samples  = []
    for img_name in os.listdir(os.path.join(data_dir, 'composite')):
        pair = {}
        for k in ['composite', 'composite_mask']:
            img_path = os.path.join(data_dir, k, img_name)
            assert os.path.exists(img_path), img_path
            pair[k]  = img_path
        samples.append(pair)
    return samples


if __name__ == '__main__':
    test_set = get_test_list_shadow_generation()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    print(f'begin testing {task_name}...')
    net = ShadowGenerationModel(device=1)

    for pair in test_set:
        comp_img, comp_mask = pair['composite'], pair['composite_mask']

        preds = net(comp_img, comp_mask, number=1)
        grid_img  = make_image_grid([comp_img, comp_mask] + preds)
    
        img_name  = os.path.basename(comp_img).replace('.png', '.jpg')
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')