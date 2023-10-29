import numpy as np
import os
import cv2
from libcom.utils.process_image import *


def get_composite_image(foreground_image, foreground_mask, background_image, bbox, option="none"):
    '''
    foreground_image, foreground_mask, background_img: image_path/numpy array/PIL.Image
    bbox: [x1,y1,x2,y2]
    option: one of ['none', 'gaussian', 'poisson']
    '''
    choices = ['none', 'gaussian', 'poisson']
    assert option in choices, f'{option} must be one of {choices}'
    fg_img   = read_image_opencv(foreground_image)
    fg_mask  = read_mask_opencv(foreground_mask)
    bg_img   = read_image_opencv(background_image)
    if option == 'none':
        comp_img,comp_mask = simple_composite_image(bg_img, fg_img, fg_mask, bbox)
    elif option == 'gaussian':
        comp_img,comp_mask = gaussian_composite_image(bg_img, fg_img, fg_mask, bbox)
    else: # option == 'poisson'
        comp_img,comp_mask = poisson_composite_image(bg_img, fg_img, fg_mask, bbox)
    return comp_img, comp_mask


def crop_and_resize_foreground(fg_img, fg_mask, bbox):
    if fg_mask.shape[:2] != fg_img.shape[:2]:
        fg_mask = cv2.resize(fg_mask, (fg_img.shape[1], fg_img.shape[0]))
    fg_bbox = convert_mask_to_bbox(fg_mask)
    fg_region = fg_img[fg_bbox[1] : fg_bbox[3], fg_bbox[0] : fg_bbox[2]]
    x1, y1, x2, y2 = bbox
    fg_region = cv2.resize(fg_region, (x2-x1, y2-y1), cv2.INTER_CUBIC)
    fg_mask   = fg_mask[fg_bbox[1] : fg_bbox[3], fg_bbox[0] : fg_bbox[2]]
    fg_mask   = cv2.resize(fg_mask, (x2-x1, y2-y1))
    fg_mask   = np.where(fg_mask > 127, 255, 0).astype(fg_img.dtype)
    return fg_region, fg_mask 


def simple_composite_image(bg_img, fg_img, fg_mask, bbox):
    fg_region, fg_mask = crop_and_resize_foreground(fg_img, fg_mask, bbox)
    x1, y1, x2, y2 = bbox
    comp_mask = np.zeros((bg_img.shape[0], bg_img.shape[1]), dtype=np.uint8) 
    comp_mask[y1:y2, x1:x2] = fg_mask
    comp_img = bg_img.copy()
    comp_img[y1:y2, x1:x2]  = np.where(fg_mask[:,:,np.newaxis]> 127, fg_region, comp_img[y1:y2, x1:x2])
    return comp_img, comp_mask
    
    
def gaussian_composite_image(bg_img, fg_img, fg_mask, bbox, kernel_size=15):
    if fg_mask.shape[:2] != fg_img.shape[:2]:
        fg_mask = cv2.resize(fg_mask, (fg_img.shape[1], fg_img.shape[0]))
    fg_mask = cv2.GaussianBlur(255-fg_mask, (kernel_size, kernel_size), kernel_size/3.)
    fg_mask = 255 - fg_mask
    fg_bbox = convert_mask_to_bbox(fg_mask)
    fg_region = fg_img[fg_bbox[1] : fg_bbox[3], fg_bbox[0] : fg_bbox[2]]
    x1, y1, x2, y2 = bbox
    fg_region = cv2.resize(fg_region, (x2-x1, y2-y1), cv2.INTER_CUBIC)
    fg_mask   = fg_mask[fg_bbox[1] : fg_bbox[3], fg_bbox[0] : fg_bbox[2]]
    fg_mask   = cv2.resize(fg_mask, (x2-x1, y2-y1))
    norm_mask = (fg_mask.astype(np.float32) / 255)[:,:,np.newaxis]
    
    comp_mask = np.zeros((bg_img.shape[0], bg_img.shape[1]), dtype=np.uint8) 
    comp_mask[y1:y2, x1:x2] = fg_mask
    comp_img = bg_img.copy()
    comp_img[y1:y2, x1:x2] = (fg_region * norm_mask + comp_img[y1:y2, x1:x2] * (1-norm_mask)).astype(comp_img.dtype) 
    return comp_img, comp_mask
    

def poisson_composite_image(bg_img, fg_img, fg_mask, bbox, clone_method=cv2.NORMAL_CLONE):
    fg_region, fg_mask = crop_and_resize_foreground(fg_img, fg_mask, bbox)
    x1, y1, x2, y2 = bbox
    center    = (x1+x2)//2, (y1+y2)//2
    comp_img  = cv2.seamlessClone(fg_region, bg_img, fg_mask[:,:,np.newaxis], center, clone_method)
    comp_mask = np.zeros((bg_img.shape[0], bg_img.shape[1]), dtype=np.uint8) 
    comp_mask[y1:y2, x1:x2] = fg_mask
    return comp_img, comp_mask