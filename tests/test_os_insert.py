from common import *
from libcom import OSInsertModel
import argparse
import os
import cv2
import shutil
import subprocess
import sys

task_name = 'os_insert'
MODES = ('conservative', 'aggressive')


def test_model(mode):
    test_set = get_test_list(task_name)
    result_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    result_dir = os.path.join(result_root, mode)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name} | {mode}...')

    model = OSInsertModel(device="cuda:0")

    for pair in test_set:
        bg_path = pair['background']
        fg_path = pair['foreground']
        fg_mask_path = pair['foreground_mask']
        bbox = pair['bbox']

        result = model(
            background_path=bg_path,
            foreground_path=fg_path,
            foreground_mask_path=fg_mask_path,
            bbox=bbox,
            result_dir=result_dir,
            mode=mode,
            verbose=True
        )

        grid_img = make_image_grid([cv2.imread(fg_path), cv2.imread(bg_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR)])
        img_name = os.path.basename(bg_path).replace('.png', '.jpg')
        res_path = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)

    print(f'end testing {task_name} | {mode}!\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    if args.mode:
        test_model(args.mode)
    else:
        for mode in MODES:
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--mode', mode],
                check=True,
            )