import numpy as np
import os
import cv2
from libcom.utils.process_image import *

def color_transfer(composite_image, composite_mask):
    """
    Generate composite image through copy-and-paste.

    Args:
        composite_image (str | numpy.ndarray): The path to composite image or the compposite image in ndarray form.
        composite_mask (str | numpy.ndarray): Mask of composite image which indicates the foreground object region in the composite image.

    Returns:
        transfered image (numpy.ndarray): Transfered image with the same resolution as input image.
    
    Examples:
        >>> from libcom import color_transfer
        >>> from libcom.utils.process_image import make_image_grid
        >>> import cv2
        >>> comp_img1  = '../tests/source/composite/1.jpg'
        >>> comp_mask1 = '../tests/source/composite_mask/1.png'
        >>> trans_img1 = color_transfer(comp_img1, comp_mask1)
        >>> comp_img2  = '../tests/source/composite/8.jpg'
        >>> comp_mask2 = '../tests/source/composite_mask/8.png'
        >>> trans_img2 = color_transfer(comp_img2, comp_mask2)
        >>> # visualization results
        >>> grid_img  = make_image_grid([comp_img1, comp_mask1, trans_img1, 
        >>>                             comp_img2, comp_mask2, trans_img2], cols=3)
        >>> cv2.imwrite('../docs/_static/image/colortransfer_result1.jpg', grid_img)

    Expected result:

    .. image:: _static/image/colortransfer_result1.jpg
        :scale: 50 %
        
    """
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