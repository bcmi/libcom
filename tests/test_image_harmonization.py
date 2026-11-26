from common import *
from libcom.image_harmonization import ImageHarmonizationModel
import os
import cv2
import shutil

task_name = 'image_harmonization'
MODEL_TYPES = ('PCTNet', 'LBM')

if __name__ == '__main__':
    test_set = get_test_list(task_name)
    result_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)

    for model_type in MODEL_TYPES:
        result_dir = os.path.join(result_root, model_type)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir, exist_ok=True)

        model = ImageHarmonizationModel(model_type=model_type)

        print(f'begin testing {task_name} | {model_type}...')
        for pair in test_set:
            comp_img, comp_mask = pair['composite'], pair['composite_mask']

            output = model(comp_img, comp_mask)

            img_name = os.path.basename(comp_img).replace('.png', '.jpg')
            grid_img = make_image_grid([comp_img, comp_mask, output])

            res_path = os.path.join(result_dir, img_name)
            cv2.imwrite(res_path, grid_img)
            print('save result to ', res_path)

        print(f'end testing {task_name} | {model_type}!\n')