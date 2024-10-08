from libcom import Mure_ObjectStitchModel
from libcom.utils.process_image import make_image_grid, draw_bbox_on_image
import cv2
import os
net    = Mure_ObjectStitchModel(device=0, sampler='plms')
sample_list = ['000000000003', '000000000004']
sample_dir  = './tests/mure_objectstitch/'
bbox_list   = [[623, 1297, 1159, 1564], [363, 205, 476, 276]]
for i, sample in enumerate(sample_list):
    bg_img = sample_dir + f'background/{sample}.jpg'
    fg_img_path = sample_dir + f'foreground/{sample}/'
    fg_mask_path = sample_dir + f'foreground_mask/{sample}/'
    fg_img_list = [os.path.join(fg_img_path, f) for f in os.listdir(fg_img_path)]
    fg_mask_list = [os.path.join(fg_mask_path, f) for f in os.listdir(fg_mask_path)]
    bbox   = bbox_list[i]
    comp, show_fg_img = net(bg_img, fg_img_list, fg_mask_list, bbox, sample_steps=25, num_samples=3)
    bg_img   = draw_bbox_on_image(bg_img, bbox)
    grid_img = make_image_grid([bg_img, show_fg_img] + [comp[i] for i in range(len(comp))])
    cv2.imwrite(f'../docs/_static/image/mureobjectstitch_result{i+1}.jpg', grid_img)