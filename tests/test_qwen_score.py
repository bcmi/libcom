from common import *
from libcom import QwenScoreModel
import os
import cv2
import shutil

task_name = 'qwen_score_prediction'

# 三个选项分别对应三个 LoRA：'QwenVL-OPA' / 'QwenVL-FOS' / 'QwenVL-Harmony'
MODEL_TYPES = {
    'QwenVL-OPA':     802816,
    'QwenVL-FOS':     802816,
    'QwenVL-Harmony': 802816,
}


if __name__ == '__main__':
    test_set = get_test_list(task_name)
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    for model_type, max_pixels in MODEL_TYPES.items():
        print(f'begin testing {task_name} | {model_type} ...')
        net = QwenScoreModel(device=0, model_type=model_type, max_pixels=max_pixels)
        for pair in test_set:
            comp_img, comp_mask = pair['composite'], pair['composite_mask']
            score = net(comp_img, comp_mask)
            img_name = os.path.basename(comp_img).replace('.png', '.jpg')
            grid_img = make_image_grid([comp_img, comp_mask],
                                       text_list=[f'{model_type}:{score:.2f}', 'composite-mask'])
            res_path = os.path.join(result_dir, f'{model_type}_{img_name}')
            cv2.imwrite(res_path, grid_img)
            print('save result to ', res_path)
        print(f'end testing {task_name} | {model_type}!\n')
