import numpy as np
import os
import cv2
from libcom.utils.process_image import *

def color_transfer(composite_image, composite_mask):
    comp_img  = read_image_opencv(composite_image)
    comp_mask = read_mask_opencv(composite_mask)
    if comp_mask.shape[:2] != comp_img.shape[:2]:
        comp_mask = cv2.resize(comp_mask, (comp_img.shape[1], comp_img.shape[0]))
    trans_comp = reinhard(comp_img, comp_mask)
    return trans_comp
        
def reinhard(comp_img, comp_mask):
    # reinhard is a classical color transfer algorithm
    # paper: https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    comp_mask = np.where(comp_mask > 127, 255, 0).astype(np.uint8)
    comp_lab  = cv2.cvtColor(comp_img, cv2.COLOR_BGR2Lab)
    bg_mean, bg_std = cv2.meanStdDev(comp_lab, mask=255 - comp_mask)
    fg_mean, fg_std = cv2.meanStdDev(comp_lab, mask=comp_mask)
    ratio = (bg_std / fg_std).reshape(-1)
    offset = (bg_mean - fg_mean * bg_std / fg_std).reshape(-1)
    trans_lab = cv2.convertScaleAbs(comp_lab * ratio + offset)
    trans_img = cv2.cvtColor(trans_lab, cv2.COLOR_Lab2BGR)
    trans_comp = np.where(comp_mask[:,:,np.newaxis] > 127, trans_img, comp_img)
    return trans_comp