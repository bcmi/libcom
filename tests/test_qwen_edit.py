from common import draw_bbox_on_image, load_bbox_from_txt, make_image_grid
from libcom import QwenEditModel
import argparse
import cv2
import os
import subprocess
import sys

task_name = 'qwen_edit'
model_set = ['photorealistic', 'artistic']


def test_model(model_type):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(test_dir, f'{task_name}_source', model_type)
    sample_ids = sorted(name for name in os.listdir(sample_dir)
                        if os.path.isdir(os.path.join(sample_dir, name)))
    if not sample_ids:
        raise ValueError(f'No test samples found in {sample_dir}')
    result_dir = os.path.join(test_dir, 'results', task_name, model_type)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name} | {model_type}...')

    net = QwenEditModel(device=0, model_type=model_type)
    for sample_id in sample_ids:
        sample = os.path.join(sample_dir, sample_id)
        background = os.path.join(sample, 'background.png')
        foreground = os.path.join(sample, 'foreground.png')
        mask = os.path.join(sample, 'foreground_mask.png')
        bbox = load_bbox_from_txt(os.path.join(sample, 'bbox.txt'))
        result = net(background, foreground, bbox, foreground_mask=mask, seed=2511)
        background_with_box = draw_bbox_on_image(
            background, bbox, color=(0, 0, 255), line_width=5
        )
        res_path = os.path.join(result_dir, f'{sample_id}.png')
        grid_img = make_image_grid(
            [background_with_box, foreground, result[0]]
        )
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
