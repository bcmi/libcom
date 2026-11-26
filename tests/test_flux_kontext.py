from common import *
from libcom import KontextBlendingHarmonizationModel
import argparse
import os
import cv2
import shutil
import subprocess
import sys

task_name = 'flux_kontext'

model_set = ['Kontext_blend', 'Kontext_harm']

def test_model(model_type):
    test_set = get_controlcom_test_list(task_name)
    result_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    result_dir = os.path.join(result_root, model_type)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name} | {model_type}...')

    net = KontextBlendingHarmonizationModel(device=0, model_type=model_type)

    for pair in test_set:
        fg_img, fg_mask = pair['foreground'], pair['foreground_mask']
        bg_img, bbox = pair['background'], pair['bbox']

        comp = net(bg_img, fg_img, bbox, fg_mask)
        bg_img = draw_bbox_on_image(bg_img, bbox)

        img_name = os.path.basename(fg_img).replace('.png', '.jpg')
        grid_img = make_image_grid([bg_img, fg_img, comp[0]])
        res_path = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)

    print(f'end testing {task_name} | {model_type}!\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=model_set)
    args = parser.parse_args()

    if args.model_type:
        test_model(args.model_type)
    else:
        for model_type in model_set:
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--model-type', model_type],
                check=True,
            )
