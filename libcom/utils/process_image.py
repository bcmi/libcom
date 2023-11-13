import os
import cv2
from PIL import Image
import numpy as np

def opencv_to_pil(input):
    return Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))

def pil_to_opencv(input):
    return cv2.cvtColor(np.asarray(input), cv2.COLOR_RGB2BGR)

def read_image_opencv(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = cv2.imread(input, cv2.IMREAD_COLOR)        
    elif isinstance(input, Image.Image):
        input = pil_to_opencv(input)
    return input

def read_mask_opencv(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = cv2.imread(input, cv2.IMREAD_GRAYSCALE)        
    elif isinstance(input, Image.Image):
        input = np.asarray(input)
    return input

def read_image_pil(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = Image.open(input).convert('RGB')
    elif isinstance(input, np.ndarray):
        input = opencv_to_pil(input)
    return input    

def read_mask_pil(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = Image.open(input).convert('L')
    elif isinstance(input, np.ndarray):
        input = Image.fromarray(input).convert('L')
    return input

def convert_mask_to_bbox(mask):
    '''
    mask: (h,w) or (h,w,1)
    '''
    if mask.ndim == 3:
        mask = mask[...,0]
    binmask = np.where(mask > 127)
    x1 = int(np.min(binmask[1]))
    x2 = int(np.max(binmask[1]))
    y1 = int(np.min(binmask[0]))
    y2 = int(np.max(binmask[0]))
    return [x1, y1, x2+1, y2+1]

def fill_image_pil(image, mask, fill_pixel=(0,0,0), thresh=127):
    image = fill_image_opencv(image, mask, fill_pixel, thresh)
    return Image.fromarray(image)

def fill_image_opencv(image, mask, fill_pixel=(0,0,0), thresh=127):
    image = np.asarray(image)
    mask  = np.asarray(mask)
    mask  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image[mask < thresh] = fill_pixel
    return image

def make_image_grid(img_list, text_list=None, resolution=(512,512), cols=None, border_color=255, border_width=5):
    if cols == None:
        cols = len(img_list)
    assert len(img_list) % cols == 0, f'{len(img_list)} % {cols} != 0'
    if isinstance(text_list, (list, tuple)):
        text_list += [''] * max(0, len(img_list) - len(text_list))
    rows = len(img_list) // cols
    hor_border = (np.ones((resolution[0], border_width, 3), dtype=np.float32) * border_color).astype(np.uint8)
    index = 0
    grid_img = []
    for i in range(rows):
        row_img = []
        for j in range(cols):
            img = read_image_opencv(img_list[index])
            img = cv2.resize(img, resolution)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if text_list and len(text_list[index]) > 0:
                cv2.putText(img, text_list[index], (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            row_img.append(img)
            if j < cols-1:
                row_img.append(hor_border)
            index += 1
        row_img = np.concatenate(row_img, axis=1)
        grid_img.append(row_img)
        if i < rows-1:
            ver_border = (np.ones((border_width, grid_img[-1].shape[1], 3), dtype=np.float32) * border_color).astype(np.uint8)
            grid_img.append(ver_border)
    grid_img = np.concatenate(grid_img, axis=0)
    return grid_img

def draw_bbox_on_image(input_img, bbox, color=(0,255,255), line_width=5):
    img = read_image_opencv(input_img)
    x1, y1, x2, y2 = bbox
    h,w,_ = img.shape
    x1 = max(x1, line_width)
    y1 = max(y1, line_width)
    x2 = min(x2, w-line_width)
    y2 = min(y2, h-line_width)
    img = cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness=line_width)
    return img
