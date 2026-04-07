from common import *
from libcom import OSInsertModel
import os
import cv2
import shutil
import torch

task_name = 'os_insert'
MODE = 'aggressive' # choose from 'conservative', 'aggressive'

if __name__ == '__main__':
    test_set = get_test_list()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', task_name, MODE)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    print(f'begin testing {task_name}...')
    
    # 创建模型实例，不需要传入 model_dir 参数
    model = OSInsertModel(device="cuda:0")
    
    for pair in test_set:
        bg_path = pair['background']
        fg_path = pair['foreground']
        fg_mask_path = pair['foreground_mask']
        bbox = pair['bbox']  # bbox 是一个列表，如 [1000, 895, 1480, 1355]
        
        result = model(
            background_path=bg_path,
            foreground_path=fg_path,
            foreground_mask_path=fg_mask_path,
            bbox=bbox,
            result_dir=result_dir,
            mode=MODE,
            verbose=True
        )
        
        print(f'Processed {os.path.basename(bg_path)}')
    print(f'end testing {task_name}!\n')