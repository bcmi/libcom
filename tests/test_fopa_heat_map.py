from common import *
from libcom import FOPAHeatMapModel
import os
import cv2
import shutil

task_name = 'fopa_heat_map'

if __name__ == '__main__':
    # collect pairwise test samples
    test_set = get_test_list_fopa_heatmap()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'grid'), exist_ok=True)
    print(f'begin testing {task_name}...')
    
    net = FOPAHeatMapModel(device=0)
    for pair in test_set[:1]:
        fg_img, fg_mask, bg_img = pair['foreground'], pair['foreground_mask'], pair['background']
        bboxes, heatmaps = net(fg_img, fg_mask, bg_img, 
                    cache_dir=os.path.join(result_dir, 'cache'), 
                    heatmap_dir=os.path.join(result_dir, 'heatmap'))
        img_name  = os.path.basename(bg_img).replace('.png', '.jpg')
        grid_img  = make_image_grid([bg_img, fg_img, heatmaps[0]])
        res_path  = os.path.join(result_dir, 'grid', img_name)
        cv2.imwrite(res_path, grid_img)
        print('save result to ', res_path)
    print(f'end testing {task_name}!\n')