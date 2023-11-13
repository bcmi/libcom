import numpy as np
import os
import cv2
from libcom.utils.process_image import *


def get_composite_image(foreground_image, foreground_mask, background_image, bbox, option="none"):
    """
    Generate composite image through copy-and-paste.

    Args:
        foreground_image (str | numpy.ndarray): The path to foreground image or the background image in ndarray form.
        foreground_mask (str | numpy.ndarray): Mask of foreground image which indicates the foreground object region in the foreground image.
        background_image (str | numpy.ndarray): The path to background image or the background image in ndarray form. 
        bbox (list): The bounding box which indicates the foreground's location in the background. [x1, y1, x2, y2]. 
        option (str): 'none', 'gaussian', or 'poisson'. Image blending method. default: None.

    Returns:
        composite_image (numpy.ndarray): Generated composite image with the same resolution as input background image.
        composite_mask (numpy.ndarray): Generated composite mask with the same resolution as composite image.
    
    Examples:
        >>> from libcom import get_composite_image
        >>> from libcom.utils.process_image import make_image_grid, draw_bbox_on_image
        >>> import cv2
        >>> test_dir = 'source/'
        >>> img_list = ['1.jpg', '8.jpg']
        >>> bbox_list = [[1000, 895, 1480, 1355], [1170, 944, 2331, 3069]] 
        >>> for i,img_name in enumerate(img_list):
        >>>     bg_img  = test_dir + 'background/' + img_name
        >>>     bbox    = bbox_list[i] # x1,y1,x2,y2
        >>>     fg_img  = test_dir + 'foreground/' + img_name
        >>>     fg_mask = test_dir + 'foreground_mask/' + img_name.replace('.jpg', '.png')
        >>>     # generate composite images by naive methods
        >>>     comp_img1, comp_mask1 = get_composite_image(fg_img, fg_mask, bg_img, bbox, 'none')
        >>>     comp_img2, comp_mask2 = get_composite_image(fg_img, fg_mask, bg_img, bbox, 'gaussian')
        >>>     comp_img3, comp_mask3 = get_composite_image(fg_img, fg_mask, bg_img, bbox, 'poisson')
        >>>     vis_list = [bg_img, fg_img, comp_img1, comp_mask1, comp_img2, comp_mask2, comp_img3, comp_mask3]
        >>>     # visualization results
        >>>     grid_img  = make_image_grid(vis_list, cols=4)
        >>>     cv2.imwrite(f'../docs/_static/image/generatecomposite_result{i+1}.jpg', grid_img)
    
    Expected result:

    .. image:: _static/image/generatecomposite_result1.jpg
        :scale: 38 %
        
    .. image:: _static/image/generatecomposite_result2.jpg
        :scale: 38 %
            
    """
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