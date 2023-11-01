from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import reverse
from cmath import inf
from curses.panel import bottom_panel
from dis import dis
from email.mime import image
import os
from io import BytesIO
import logging
import base64
from sre_parse import State
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from torchvision.transforms import Resize
from PIL import Image,ImageDraw,ImageFile
# from sympy import source
import torch.utils.data as data
import json
import time
import cv2
cv2.setNumThreads(0)
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
from tqdm import tqdm
import sys
import shutil
import transformers
import gc

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)
# import objgraph
# from memory_profiler import profile
# import tracemalloc
# torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 300
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))

def tensor2numpy(image, normalized=False, image_size=(512, 512)):
    image = Resize(image_size)(image)
    if not normalized:
        image = (image + 1.0) / 2.0  # -1,1 -> 0,1; b,c,h,w
    image = torch.clamp(image, 0., 1.)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 2, 3, 1)
    image = image.numpy()
    image = (image * 255).astype(np.uint8)
    image = image[..., [2,1,0]]
    return image

def get_tensor(normalize=True, toTensor=True, resize=True, image_size=(512, 512)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True, resize=True, image_size=(224, 224)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def scan_all_files():
    bbox_dir = os.path.join(proj_dir, '../../dataset/open-images/bbox_mask')
    assert os.path.exists(bbox_dir), bbox_dir
    
    bad_files = []
    for split in os.listdir(bbox_dir):
        total_images, total_pairs, bad_masks, bad_images = 0, 0, 0, 0
        subdir = os.path.join(bbox_dir, split)
        if not os.path.isdir(subdir) or split not in ['train', 'test', 'validation']:
            continue
        for file in tqdm(os.listdir(subdir)):
            try:
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        info = line.split(' ')
                        mask_file = os.path.join(bbox_dir, '../masks', split, info[-2])
                        if os.path.exists(mask_file):
                            total_pairs += 1
                        else:
                            bad_masks += 1
                total_images += 1
            except:
                bad_files.append(file)
                bad_images += 1
        print('{}, {} images({} bad), {} pairs({} bad)'.format(
            split, total_images, bad_images, total_pairs, bad_masks))
        
        if len(bad_files) > 0:
            with open(os.path.join(bbox_dir, 'bad_files.txt'), 'w') as f:
                for file in bad_files:
                    f.write(file + '\n')
        
    print(f'{len(bad_files)} bad_files')

    
def bbox2mask(bbox, mask_w, mask_h):
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
    return mask
    
def mask2bbox(mask):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [x1, y1, x2, y2]

    
def constant_pad_bbox(bbox, width, height, value=10):
    ### Get reference image
    bbox_pad=copy.deepcopy(bbox)
    left_space  = bbox[0]
    up_space    = bbox[1]
    right_space = width  - bbox[2]
    down_space  = height - bbox[3] 

    bbox_pad[0]=bbox[0]-min(value, left_space)
    bbox_pad[1]=bbox[1]-min(value, up_space)
    bbox_pad[2]=bbox[2]+min(value, right_space)
    bbox_pad[3]=bbox[3]+min(value, down_space)
    return bbox_pad
    
def rescale_image_with_bbox(image, bbox=None, long_size=1024):
    src_width, src_height = image.size
    if max(src_width, src_height) <= long_size:
        dst_img = image
        dst_width, dst_height = dst_img.size
    else:
        scale = float(long_size) / max(src_width, src_height)
        dst_width, dst_height = int(scale * src_width), int(scale * src_height)
        dst_img  = image.resize((dst_width, dst_height))
    if bbox == None:
        return dst_img
    bbox[0] = int(float(bbox[0]) / src_width  * dst_width)
    bbox[1] = int(float(bbox[1]) / src_height * dst_height)
    bbox[2] = int(float(bbox[2]) / src_width  * dst_width)
    bbox[3] = int(float(bbox[3]) / src_height * dst_height)
    return dst_img, bbox
    
def crop_foreground_by_bbox(img, mask, bbox, pad_bbox=10):
    width,height = img.shape[1], img.shape[0]
    bbox_pad = constant_pad_bbox(bbox, width, height, pad_bbox) if pad_bbox > 0 else bbox
    img = img[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2]]
    if mask is not None:
        mask = mask[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2]]
    return img, mask, bbox_pad

def image2inpaint(image, mask):
    if len(mask.shape) == 2:
        mask_f = mask[:,:,np.newaxis]
    else:
        mask_f = mask
    mask_f  = mask_f.astype(np.float32) / 255
    inpaint = image.astype(np.float32)
    gray  = np.ones_like(inpaint) * 127
    inpaint = inpaint * (1 - mask_f) + mask_f * gray
    inpaint = np.uint8(inpaint)
    return inpaint

def check_dir(dir):
    assert os.path.exists(dir), dir
    return dir

def get_bbox_tensor(bbox, width, height):
    norm_bbox = bbox
    norm_bbox = torch.tensor(norm_bbox).reshape(-1).float()
    norm_bbox[0::2] /= width
    norm_bbox[1::2] /= height
    return norm_bbox

    
def reverse_image_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = (tensor.float() + 1) / 2
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor, (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def reverse_mask_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def reverse_clip_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073],  dtype=torch.float)
    MEAN = MEAN.reshape(1, 3, 1, 1).to(tensor.device)
    STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float)
    STD  = STD.reshape(1, 3, 1, 1).to(tensor.device)
    tensor = (tensor * STD) + MEAN
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def random_crop_image(image, crop_w, crop_h):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    x_space = image.shape[1] - crop_w
    y_space = image.shape[0] - crop_h
    x1 = np.random.randint(0, x_space) if x_space > 0 else 0
    y1 = np.random.randint(0, y_space) if y_space > 0 else 0
    image = image[y1 : y1+crop_h, x1 : x1+crop_w]
    # assert crop.shape[0] == crop_h and crop.shape[1] == crop_w, (y1, x1, image.shape, crop.shape, crop_w, crop_h)
    return image

def read_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
    return img

def read_mask(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')
    return img

# poisson blending
def poisson_blending(fg, fg_mask, bg, center=None):
    if center is None:
        height, width, _ = bg.shape
        center = (int(width/2), int(height/2))
    return cv2.seamlessClone(fg, bg, fg_mask, center, cv2.MIXED_CLONE)

