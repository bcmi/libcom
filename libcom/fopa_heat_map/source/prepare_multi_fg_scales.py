import os
import csv
import numpy as np
from PIL import Image
import cv2

fg_scale_num = 16

def fill_image_by_mask(image, mask, fill_pixel=0, thresh=127):
    image = np.asarray(image).copy()
    mask  = np.asarray(mask)
    fill_img = (np.ones_like(image) * fill_pixel).astype(np.uint8)
    image = np.where(mask > thresh, image, fill_img)
    return Image.fromarray(image)

def prepare_multi_fg_scales(cache_dir,fg_path,mask_path,bg_path,fg_scale_num):
    os.makedirs(cache_dir, exist_ok=True)
    fg_name   = os.path.splitext(os.path.basename(fg_path))[0]
    bg_name   = os.path.splitext(os.path.basename(bg_path))[0]
    fg_scales = list(range(1, fg_scale_num+1))
    fg_scales = [i/(1+fg_scale_num+1) for i in fg_scales]
    
    scaled_fg_dir   = os.path.join(cache_dir, f'fg_{fg_scale_num}scales')
    scaled_mask_dir = os.path.join(cache_dir, f'mask_{fg_scale_num}scales')    
    csv_file = os.path.join(cache_dir, f'{fg_scale_num}scales.csv')
    os.makedirs(scaled_fg_dir,   exist_ok=True)
    os.makedirs(scaled_mask_dir, exist_ok=True)

    file = open(csv_file, mode='w', newline='')
    writer = csv.writer(file)
    csv_head = ['fg_name', 'mask_name', 'bg_name', 'scale', 'newWidth', 'newHeight', 'pos_label', 'neg_label']
    writer.writerow(csv_head)

    bg_img = Image.open(bg_path).convert("RGB")  
    bg_img_aspect = bg_img.height / bg_img.width
    fg_tocp   = Image.open(fg_path).convert("RGB")
    mask_tocp = Image.open(mask_path).convert("RGB")
    fg_tocp   = fill_image_by_mask(fg_tocp, mask_tocp)
    fg_tocp_aspect = fg_tocp.height / fg_tocp.width
    
    for fg_scale in fg_scales:
        if fg_tocp_aspect > bg_img_aspect:
            new_height = int(bg_img.height * fg_scale)
            new_width  = int(new_height / fg_tocp.height * fg_tocp.width)
        else:
            new_width  = int(bg_img.width * fg_scale)
            new_height = int(new_width / fg_tocp.width * fg_tocp.height)
        
        top    = int((bg_img.height - new_height) / 2)
        bottom = top + new_height
        left   = int((bg_img.width - new_width) / 2)
        right  = left + new_width
        
        fg_img_ = np.asarray(fg_tocp.resize((new_width, new_height)))
        mask_   = np.asarray(mask_tocp.resize((new_width, new_height)))
        fg_img  = np.zeros((bg_img.height, bg_img.width, 3), dtype=np.uint8) 
        mask    = np.zeros((bg_img.height, bg_img.width, 3), dtype=np.uint8) 
        fg_img[top:bottom, left:right, :] = fg_img_
        mask[top:bottom, left:right, :] = mask_
        fg_img = Image.fromarray(fg_img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        
        basename = f'{fg_name}_{bg_name}_{new_width}_{new_height}.jpg'
        fg_img_path = os.path.join(scaled_fg_dir, basename)
        mask_path = os.path.join(scaled_mask_dir, basename)
        fg_img.save(fg_img_path)
        mask.save(mask_path)
        writer.writerow([os.path.basename(fg_path), 
                         os.path.basename(mask_path), 
                         os.path.basename(bg_path), 
                         fg_scale, new_width, new_height, 
                         None, None])
    file.close()  
    csv_data = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['pos_label']=="":
                row['pos_label'] = [[0,0]]
            if row['neg_label']=="":
                row['neg_label'] = [[0,0]]
            csv_data.append(row)
    return scaled_fg_dir, scaled_mask_dir, csv_file


