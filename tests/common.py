import os,sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_dir)
import numpy as np
from libcom.utils.process_image import *

def load_bbox_from_txt(txt_path):
    assert os.path.exists(txt_path), txt_path
    bbox = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            x1,y1,x2,y2 = map(int, line[:4])
            bbox.append([x1, y1, x2, y2])
    return bbox[0] if len(bbox) == 1 else bbox

def get_test_list():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source')
    samples  = []
    for img_name in os.listdir(os.path.join(data_dir, 'background')):
        pair = {}
        for k in ['background', 'foreground', 'foreground_mask', 'composite', 'composite_mask', 'bbox']:
            if k == 'bbox':
                txt_path = os.path.join(data_dir, 'bbox', img_name.split('.')[0] + '.txt')
                bbox = load_bbox_from_txt(txt_path)
                pair[k] = bbox
            else:
                img_path = os.path.join(data_dir, k, img_name)
                if not os.path.exists(img_path):
                    img_path = os.path.join(data_dir, k, img_name.replace('.jpg', '.png'))
                assert os.path.exists(img_path), img_path
                pair[k]  = img_path
        samples.append(pair)
    return samples

def get_test_list_painterly_harmonization():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'painterly_harmonization_source')
    samples  = []
    for img_name in os.listdir(os.path.join(data_dir, 'composite')):
        pair = {}
        for k in ['composite', 'composite_mask']:
            img_path = os.path.join(data_dir, k, img_name)
            assert os.path.exists(img_path), img_path
            pair[k]  = img_path
        samples.append(pair)
    return samples

def get_test_list_harmony_prediction():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harmony_score_prediction')
    samples  = []
    for img_name in os.listdir(os.path.join(data_dir, 'composite')):
        pair = {}
        for k in ['composite', 'composite_mask']:
            img_path = os.path.join(data_dir, k, img_name)
            assert os.path.exists(img_path), img_path
            pair[k]  = img_path
        samples.append(pair)
    return samples

def get_test_list_fopa_heatmap():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fopa_heat_map')
    samples  = []
    for img_name in os.listdir(os.path.join(data_dir, 'background')):
        pair = {}
        for k in ['background', 'foreground_mask', 'foreground']:
            img_path = os.path.join(data_dir, k, img_name)
            assert os.path.exists(img_path), img_path
            pair[k]  = img_path
        samples.append(pair)
    return samples

def get_controlcom_test_list():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'controllable_composition')
    samples  = []
    for img_name in os.listdir(os.path.join(data_dir, 'background')):
        pair = {}
        for k in ['background', 'foreground', 'foreground_mask', 'bbox']:
            if k == 'bbox':
                txt_path = os.path.join(data_dir, 'bbox', img_name.split('.')[0] + '.txt')
                bbox = load_bbox_from_txt(txt_path)
                pair[k] = bbox
            else:
                img_path = os.path.join(data_dir, k, img_name)
                if not os.path.exists(img_path):
                    img_path = os.path.join(data_dir, k, img_name.replace('.jpg', '.png'))
                assert os.path.exists(img_path), img_path
                pair[k]  = img_path
        samples.append(pair)
    return samples

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