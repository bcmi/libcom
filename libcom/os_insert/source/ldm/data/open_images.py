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
from PIL import Image,ImageDraw,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
import albumentations as A
from tqdm import tqdm
import sys
import shutil
import transformers
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


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

class DataAugmentation:
    def __init__(self, border_mode=0):
        self.blur = A.Blur(p=0.3)
        self.appearance_trans = A.Compose([
            A.ColorJitter(brightness=0.3, 
                          contrast=0.3, 
                          saturation=0.3, 
                          hue=0.05, 
                          always_apply=False, 
                          p=1)],
            # additional_targets={'image':'image', 'image1':'image', 'image2':'image'}
            )
        self.geometric_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20,
                     border_mode=border_mode,
                     value=(127,127,127),
                     mask_value=0,
                     p=1),
            A.Perspective(scale=(0.01, 0.1), 
                          pad_mode=border_mode,
                          pad_val =(127,127,127),
                          mask_pad_val=0,
                          fit_output=False, 
                          p=0.5)
        ])
        self.bbox_maxlen = 0.8
        self.crop_bg_p  = 0.5
    
    def __call__(self, bg_img, bbox, bg_mask, fg_img, fg_mask):
        # randomly crop background image
        if self.crop_bg_p > 0 and np.random.rand() < self.crop_bg_p:
            trans_bg, trans_bbox, trans_mask = self.random_crop_background(bg_img, bbox, bg_mask)
        else:
            trans_bg, trans_bbox, trans_mask = bg_img, bbox, bg_mask
        
        bbox_mask = bbox2mask(trans_bbox, trans_bg.shape[1], trans_bg.shape[0])
        # perform illumination and pose transformation on foreground
        trans_fg, trans_fgmask = self.augment_foreground(fg_img.copy(), fg_mask.copy())
        return {"bg_img":   trans_bg,
                "bg_mask":  trans_mask,
                "bbox":     trans_bbox,
                "bbox_mask": bbox_mask,
                "fg_img":   trans_fg,
                "fg_mask":  trans_fgmask,
                "gt_fg_mask": fg_mask}
    
    def augment_foreground(self, img, mask):
        # appearance transformed image
        transformed = self.appearance_trans(image=img)
        img = transformed['image']
        transformed = self.geometric_trans(image=img, mask=mask)
        trans_img  = transformed['image']
        trans_mask = transformed['mask']
        return trans_img, trans_mask

    def random_crop_background(self, image, bbox, mask):
        width, height = image.shape[1], image.shape[0]
        bbox_w = float(bbox[2] - bbox[0]) / width
        bbox_h = float(bbox[3] - bbox[1]) / height
        
        left, right, top, down = 0, width, 0, height 
        if bbox_w < self.bbox_maxlen:
            maxcrop = width - bbox_w * width / self.bbox_maxlen
            left  = int(np.random.rand() * min(maxcrop, bbox[0]))
            right = width - int(np.random.rand() * min(maxcrop, width - bbox[2]))

        if bbox_h < self.bbox_maxlen:
            maxcrop = (height - bbox_h * height / self.bbox_maxlen) / 2
            top   = int(np.random.rand() * min(maxcrop, bbox[1]))
            down  = height - int(np.random.rand() * min(maxcrop, height - bbox[3]))
        
        trans_bbox = [bbox[0] - left, bbox[1] - top, bbox[2] - left, bbox[3] - top]
        trans_image = image[top:down, left:right]
        trans_mask  = mask[top:down, left:right]
        return trans_image, trans_bbox, trans_mask

    
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

class OpenImageDataset(data.Dataset):
    def __init__(self,split,**args):
        self.split=split
        dataset_dir = args['dataset_dir']
        assert os.path.exists(dataset_dir), dataset_dir
        self.bbox_dir = check_dir(os.path.join(dataset_dir, 'refine/box', split))
        self.image_dir= check_dir(os.path.join(dataset_dir, 'images', split))
        self.inpaint_dir = check_dir(os.path.join(dataset_dir, 'refine/inpaint', split))
        self.mask_dir = check_dir(os.path.join(dataset_dir, 'refine/mask', split))
        self.bbox_path_list = np.array(self.load_bbox_path_list())
        self.length=len(self.bbox_path_list)
        self.random_trans = DataAugmentation()
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = (args['image_size'], args['image_size'])
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(normalize=False, image_size=(224, 224))
        self.bad_images = []
    
    def load_bbox_path_list(self):
        # cache_dir  = os.path.dirname(os.path.abspath(self.bbox_dir))
        cache_dir = self.bbox_dir
        cache_file = os.path.join(cache_dir, f'{self.split}.json')
        if os.path.exists(cache_file):
            print('load bbox list from ', cache_file)
            with open(cache_file, 'r') as f:
                bbox_path_list = json.load(f)
        else:
            bbox_path_list= os.listdir(self.bbox_dir)
            bbox_path_list.sort()
            print('save bbox list to ', cache_file)
            with open(cache_file, 'w') as f:
                json.dump(bbox_path_list, f)
        return bbox_path_list

    def load_bbox_file(self, bbox_file):
        bbox_list = []
        with open(bbox_file, 'r') as f:
            for line in f.readlines():
                info  = line.strip().split(' ')
                bbox  = [int(float(f)) for f in info[:4]]
                mask  = os.path.join(self.mask_dir, info[-1])
                inpaint = os.path.join(self.inpaint_dir, info[-1].replace('.png', '.jpg'))
                if os.path.exists(mask) and os.path.exists(inpaint):
                    bbox_list.append((bbox, mask, inpaint))
        return bbox_list

    
    def sample_augmented_data(self, source_np, bbox, mask, fg_img, fg_mask):
        transformed = self.random_trans(source_np, bbox, mask, fg_img, fg_mask)
        # get ground-truth composite image and bbox
        gt_mask = Image.fromarray(transformed["bg_mask"])
        img_width, img_height = gt_mask.size
        gt_mask_tensor = self.mask_transform(gt_mask)
        gt_mask_tensor = torch.where(gt_mask_tensor > 0.5, 1, 0).float() 
        
        gt_img_tensor  = Image.fromarray(transformed['bg_img'])
        gt_img_tensor  = self.sd_transform(gt_img_tensor)
        
        mask_tensor = Image.fromarray(transformed['bbox_mask'])
        mask_tensor = self.mask_transform(mask_tensor)
        mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
        bbox_tensor = transformed['bbox']
        bbox_tensor = get_bbox_tensor(bbox_tensor, img_width, img_height)
        # get foreground and foreground mask
        fg_mask_tensor = Image.fromarray(transformed['fg_mask'])
        fg_mask_tensor = self.clip_mask_transform(fg_mask_tensor)
        fg_mask_tensor = torch.where(fg_mask_tensor > 0.5, 1, 0)
        
        fg_img_tensor = transformed['fg_img'] * (transformed['fg_mask'][:,:,None] > 0.5)
        fg_img_tensor = Image.fromarray(fg_img_tensor)
        fg_img_tensor = self.clip_transform(fg_img_tensor)
        inpaint = gt_img_tensor * (mask_tensor < 0.5)
        
        return {"gt_img":  gt_img_tensor,
                "gt_mask": gt_mask_tensor,
                "bg_img": inpaint,
                "bg_mask": mask_tensor,
                "fg_img":  fg_img_tensor,
                "fg_mask": fg_mask_tensor,
                "bbox": bbox_tensor}
    
    def __getitem__(self, index):
        try:
            # get bbox and mask
            bbox_file  = self.bbox_path_list[index] 
            bbox_path  = os.path.join(self.bbox_dir, bbox_file)
            bbox_list  = self.load_bbox_file(bbox_path)
            bbox,mask_path,inpaint_path = random.choice(bbox_list)
            # get source image and mask
            image_path = os.path.join(self.image_dir, os.path.splitext(bbox_file)[0] + '.jpg')
            source_img  = read_image(image_path)
            source_img, bbox = rescale_image_with_bbox(source_img, bbox)
            source_np   = np.array(source_img)
            mask = read_mask(mask_path)
            mask = mask.resize((source_np.shape[1], source_np.shape[0]))
            mask = np.array(mask)
            # bbox = mask2bbox(mask)
            fg_img, fg_mask, bbox  = crop_foreground_by_bbox(source_np, mask, bbox)
            sample = self.sample_augmented_data(source_np, bbox, mask, fg_img, fg_mask)
            sample['image_path'] = image_path
            return sample
        except Exception as e:
            print(os.getpid(), bbox_file, e)
            index = np.random.randint(0, len(self)-1)
            return self[index]
        
    def __len__(self):
        return self.length
    
class COCOEEDataset(data.Dataset):
    def __init__(self, **args):
        dataset_dir = args['dataset_dir']
        self.use_inpaint_background = args['augment_config'].use_inpaint_background if 'augment_config' in args else True
        assert os.path.exists(dataset_dir), dataset_dir
        self.src_dir = check_dir(os.path.join(dataset_dir, "GT_3500"))
        self.ref_dir = check_dir(os.path.join(dataset_dir, 'Ref_3500'))
        self.mask_dir = check_dir(os.path.join(dataset_dir, 'Mask_bbox_3500'))
        self.gt_mask_dir = check_dir(os.path.join(dataset_dir, 'mask'))
        self.inpaint_dir = check_dir(os.path.join(dataset_dir, 'inpaint'))
        self.ref_mask_dir = check_dir(os.path.join(dataset_dir, 'ref_mask'))
        self.image_list = os.listdir(self.src_dir)
        self.image_list.sort()
        
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = args['image_size'], args['image_size']
        self.sd_transform   = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(normalize=False, image_size=(224, 224))
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        try:
            image = self.image_list[index]
            src_path = os.path.join(self.src_dir, image)
            src_img = read_image(src_path)
            src_tensor = self.sd_transform(src_img)
            im_name  = os.path.splitext(image)[0].split('_')[0]
            # reference image and object mask
            ref_name = im_name + '_ref.png'

            ref_mask_path = os.path.join(self.ref_mask_dir, ref_name)
            assert os.path.exists(ref_mask_path), ref_mask_path
            ref_mask = read_mask(ref_mask_path)
            ref_mask_np = np.array(ref_mask)
            ref_mask_tensor = self.clip_mask_transform(ref_mask)
            ref_mask_tensor = torch.where(ref_mask_tensor > 0.5, 1, 0)

            ref_path = os.path.join(self.ref_dir, ref_name)
            assert os.path.exists(ref_path), ref_path
            ref_img = read_image(ref_path)
            ref_img_np = np.array(ref_img) * (ref_mask_np[:,:,None] > 0.5)
            ref_img = Image.fromarray(ref_img_np) 
            ref_tensor = self.clip_transform(ref_img)
            
            mask_path = os.path.join(self.mask_dir, im_name + '_mask.png')
            assert os.path.exists(mask_path), mask_path
            mask_img = read_mask(mask_path)
            mask_img = mask_img.resize((src_img.width, src_img.height))
            bbox = mask2bbox(np.array(mask_img))
            bbox_tensor = get_bbox_tensor(bbox, src_img.width, src_img.height)
            mask_tensor = self.mask_transform(mask_img) 
            mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
            inpaint_tensor = src_tensor * (1 - mask_tensor)
        
            return {"image_path": src_path,
                    "gt_img":  src_tensor,
                    "gt_mask": mask_tensor,
                    "bg_img":  inpaint_tensor,
                    "bg_mask": mask_tensor,
                    "fg_img":  ref_tensor,
                    'fg_mask': ref_mask_tensor,
                    "bbox":    bbox_tensor}
        except:
            idx = np.random.randint(0, len(self)-1)
            return self[idx]

class FOSEDataset(data.Dataset):
    def __init__(self, dataset_dir='/mnt/new/397927/dataset/FOS_Evaluation'):
        data_root = dataset_dir
        self.bg_dir   = os.path.join(data_root, 'background')
        self.mask_dir = os.path.join(data_root, 'bbox_mask')
        self.bbox_dir = os.path.join(data_root, 'bbox')
        self.fg_dir   = os.path.join(data_root, 'foreground') 
        self.fgmask_dir = os.path.join(data_root, 'foreground_mask')
        self.image_list = os.listdir(self.bg_dir)
        self.image_size = (512, 512)
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.sd_transform   = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        
    def __len__(self):
        return len(self.image_list)

    def load_bbox_file(self, bbox_file):
        bbox_list = []
        with open(bbox_file, 'r') as f:
            for line in f.readlines():
                info  = line.strip().split(' ')
                bbox  = [int(float(f)) for f in info[:4]]
                bbox_list.append(bbox)
        return bbox_list[0]
    
    def __getitem__(self, index):
        image = self.image_list[index]
        bg_path = os.path.join(self.bg_dir, image)
        bg_img  = Image.open(bg_path).convert('RGB')
        bg_w, bg_h = bg_img.size
        bg_t    = self.sd_transform(bg_img)
        fg_path = os.path.join(self.fg_dir, image)
        fg_img  = Image.open(fg_path).convert('RGB')
        fgmask_path = os.path.join(self.fgmask_dir, image)
        fg_mask   = Image.open(fgmask_path).convert('L')
        fg_np  = np.array(fg_img) * (np.array(fg_mask)[:,:,None] > 0.5)
        fg_img = Image.fromarray(fg_np) 

        fg_t     = self.clip_transform(fg_img)
        fgmask_t = self.mask_transform(fg_mask) 
        mask_path = os.path.join(self.mask_dir, image)
        mask = Image.open(mask_path).convert('L')
        mask_t = self.mask_transform(mask)
        mask_t = torch.where(mask_t > 0.5, 1, 0).float()
        inpaint_t = bg_t * (1 - mask_t)
        bbox_path = os.path.join(self.bbox_dir, image.replace('.png', '.txt'))
        bbox   = self.load_bbox_file(bbox_path)
        bbox_t = get_bbox_tensor(bbox, bg_w, bg_h)

        return {"image_path": bg_path,
                "bg_img":  bg_t,
                "inpaint_img":  inpaint_t,
                "bg_mask": mask_t,
                "fg_img":  fg_t,
                "fg_mask": fgmask_t,
                'bbox': bbox_t}

def test_fos_dataset():
    dataset = FOSEDataset()
    dataloader = data.DataLoader(dataset=dataset, 
                            batch_size=4, 
                            shuffle=False,
                            num_workers=4)
    for i, batch in enumerate(dataloader):
        print(i, len(dataset), batch['inpaint_img'].shape, batch['fg_img'].shape)

    
def vis_random_augtype(batch):
    file = batch['image_path']
    gt_t = batch['gt_img']
    gtmask_t = batch['gt_mask']
    bg_t = batch['bg_img']
    bgmask_t  = batch['bg_mask']
    fg_t = batch['fg_img']
    fgmask_t = batch['fg_mask']
    
    gt_imgs  = reverse_image_tensor(gt_t)
    gt_masks = reverse_mask_tensor(gtmask_t) 
    bg_imgs  = reverse_image_tensor(bg_t)
    fg_imgs  = reverse_clip_tensor(fg_t)
    fg_masks = reverse_mask_tensor(fgmask_t)

    ver_border = np.ones((gt_imgs[0].shape[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
    img_list = []
    for i in range(len(gt_imgs)):
        im_name = os.path.basename(file[i]) if len(file) > 1 else os.path.basename(file[0])
        cat_img = np.concatenate([bg_imgs[i], ver_border, fg_imgs[i], ver_border, fg_masks[i], ver_border, gt_imgs[i], ver_border, gt_masks[i]], axis=1)
        if i > 0:
            hor_border = np.ones((10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
            img_list.append(hor_border)
        img_list.append(cat_img)
    img_batch = np.concatenate(img_list, axis=0)
    return img_batch

def test_cocoee_dataset():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.validation
    dataset  = instantiate_from_config(configs)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=4, 
                            shuffle=False,
                            num_workers=4)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    vis_dir = os.path.join(proj_dir, 'outputs/test_samples')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        file = batch['image_path']
        gt_t = batch['gt_img']
        bgmask_t  = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        bg_t = batch['bg_img']
        fg_mask = batch['fg_mask']
        im_name = os.path.basename(file[0])
        # test_fill_mask(batch, i)
        print(i, len(dataloader), gt_t.shape, fg_t.shape, gt_t.shape, bbox_t.shape, fg_mask.shape)
        batch_img = vis_random_augtype(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        if i > 10:
            break
    

def test_open_images():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    configs.params.split = 'validation'
    dataset  = instantiate_from_config(configs)
    bs = 4
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=bs, 
                            shuffle=False,
                            num_workers=4)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    vis_dir = os.path.join(proj_dir, 'outputs/train_samples')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, batch in enumerate(dataloader):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor) and batch[k].shape[0] == 1:
                batch[k] = batch[k][0]

        file = batch['image_path']
        gt_t = batch['gt_img']
        gtmask_t = batch['gt_mask']
        bgmask_t  = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        im_name = os.path.basename(file[0])
        # test_fill_mask(batch, i)
        print(i, len(dataloader), gt_t.shape, gtmask_t.shape, fg_t.shape, gt_t.shape, bbox_t.shape)
        batch_img = vis_random_augtype(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        if i > 10:
            break
    
def test_open_images_efficiency():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/v1.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    configs.params.split = 'train'
    dataset  = instantiate_from_config(configs)
    bs = 16
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=bs, 
                            shuffle=False,
                            num_workers=16)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    start = time.time()
    data_len = len(dataloader)
    for i,batch in enumerate(dataloader):
        image = batch['gt_img']
        end = time.time()
        if i % 10 == 0:
            print('{:.2f}, avg time {:.1f}ms'.format(
                float(i) / data_len, (end-start) / (i+1) * 1000
            ))

        
if __name__ == '__main__':
    # test_mask_blur_batch()
    # test_open_images()
    # test_open_images_efficiency()
    # test_cocoee_dataset()
    test_fos_dataset()


