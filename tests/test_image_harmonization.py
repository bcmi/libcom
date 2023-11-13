from common import *
from libcom import ImageHarmonizationModel
import os
import cv2
import shutil

task_name = 'image_harmonization'

if __name__ == '__main__':
    test_set = get_test_list()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    cdt_model = ImageHarmonizationModel(model_type='CDTNet')
    pct_model = ImageHarmonizationModel(model_type='PCTNet')
    print(f'begin testing {task_name}...')
    for pair in test_set:
        comp_img, comp_mask = pair['composite'], pair['composite_mask']
        cdt_res   = cdt_model(comp_img, comp_mask)
        pct_res   = pct_model(comp_img, comp_mask)
        # visulization results
        img_name  = os.path.basename(comp_img).replace('.png', '.jpg')
        grid_img  = make_image_grid([comp_img, comp_mask, cdt_res, pct_res])
        res_path  = os.path.join(result_dir, img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')
