# -*- coding: utf-8 -*-
import json
import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from libcom.fopa_heat_map.source.data.all_transforms import Compose, JointResize


class CPDataset(Dataset):
    def __init__(self, file, bg_dir, fg_dir, mask_dir, in_size, datatype='train'):
        """
        initialize dataset

        Args:
            file(str): file with training/test data information
            bg_dir(str): folder with background images
            fg_dir(str): folder with foreground images
            mask_dir(str): folder with mask images
            in_size(int): input size of network
            datatype(str): "train" or "test"
        """
    
        self.datatype = datatype
        self.data = _collect_info(file, bg_dir, fg_dir, mask_dir, datatype)
        self.insize = in_size

        self.train_triple_transform = Compose([JointResize(in_size)])
        self.train_img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
            ]
        )
        self.train_mask_transform = transforms.ToTensor()

        self.transforms_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        load each item
        return:
            i: the image index,
            bg_t:(1 * 3 * in_size * in_size) background image,
            mask_t:(1 * 1 * in_size * in_size) scaled foreground mask
            fg_t:(1 * 3 * in_size * in_size) scaled foreground image
            target_t: (1 * in_size * in_size) pixel-wise binary labels
            labels_num: (int) the number of annotated pixels
        """
        i, _, bg_path, fg_path, mask_path, scale, pos_label, neg_label, fg_path_2, mask_path_2, w, h = self.data[index]
                
        fg_name =  fg_path.split('/')[-1][:-4]
        ## save_name: fg_bg_w_h_scale.jpg
        save_name = fg_name + '_' + str(scale) + '.jpg'

        bg_img = Image.open(bg_path)
        fg_img = Image.open(fg_path)
        mask = Image.open(mask_path)
        if len(bg_img.split()) != 3: 
            bg_img = bg_img.convert("RGB")
        if len(fg_img.split()) == 3:
            fg_img = fg_img.convert("RGB")
        if len(mask.split()) == 3:
            mask = mask.convert("L")

        is_flip = False
        if self.datatype == 'train' and np.random.uniform() < 0.5:
            is_flip = True

        # make composite images which are used in feature mimicking 
        fg_tocp = Image.open(fg_path_2).convert("RGB")
        mask_tocp = Image.open(mask_path_2).convert("L")
        composite_list = []
        for pos in pos_label:
            x_, y_ = pos
            x = int(x_ - w / 2)
            y = int(y_ - h / 2)
            composite_list.append(make_composite(fg_tocp, mask_tocp, bg_img, [x, y, w, h], is_flip))

        for pos in neg_label:
            x_, y_ = pos
            x = int(x_ - w / 2)
            y = int(y_ - h / 2) 
            composite_list.append(make_composite(fg_tocp, mask_tocp, bg_img, [x, y, w, h], is_flip))

        composite_list_ = torch.stack(composite_list, dim=0)
        composite_cat = torch.zeros(50 - len(composite_list), 4, 256, 256)
        composite_list = torch.cat((composite_list_, composite_cat), dim=0)

        # positive pixels are 1, negative pixels are 0, other pixels are 255
        # feature_pos: record the positions of annotated pixels
        target, feature_pos = _obtain_target(bg_img.size[0], bg_img.size[1], self.insize, pos_label, neg_label, is_flip)
        for i in range(50 - len(feature_pos)):
            feature_pos.append((0, 0)) # pad the length to 50
        feature_pos = torch.Tensor(feature_pos)
        
        # resize the foreground/background to 256, convert them to tensors
        bg_t, fg_t, mask_t = self.train_triple_transform(bg_img, fg_img, mask)
        mask_t = self.train_mask_transform(mask_t)
        fg_t = self.train_img_transform(fg_t)
        bg_t = self.train_img_transform(bg_t)

        if is_flip == True:
            fg_t = self.transforms_flip(fg_t)
            bg_t = self.transforms_flip(bg_t)
            mask_t = self.transforms_flip(mask_t)

        # tensor is normalized to [0,1]，map back to [0, 255] for ease of computation
        target_t = self.train_mask_transform(target) * 255
        labels_num = (target_t != 255).sum()

        return i, bg_t, mask_t, fg_t, target_t.squeeze(), labels_num, composite_list, feature_pos, w, h, save_name
    

def _obtain_target(original_width, original_height, in_size, pos_label, neg_label, isflip=False):
    """
    put 0, 1 labels on a 256x256 score map
    Args:
        original_width(int): width of original background
        original_height(int): height of original background
        in_size(int): input size of network
        pos_label(list): positive pixels in original background
        neg_label(list): negative pixels in original background
    return:
        target_r: score map with ground-truth labels
    """
    target = np.uint8(np.ones((in_size, in_size)) * 255)
    feature_pos = []
    for pos in pos_label:
        x, y = pos
        x_new = int(x * in_size / original_width) 
        y_new = int(y * in_size / original_height)
        target[y_new, x_new] = 1.
        if isflip:
            x_new = 256 - x_new
        feature_pos.append((x_new, y_new))
    for pos in neg_label:
        x, y = pos
        x_new = int(x * in_size / original_width)
        y_new = int(y * in_size / original_height)
        target[y_new, x_new] = 0.
        if isflip:
            x_new = 256 - x_new
        feature_pos.append((x_new, y_new))
    target_r = Image.fromarray(target)
    if isflip:
        target_r = transforms.RandomHorizontalFlip(p=1)(target_r)
    return target_r, feature_pos


def _collect_info(json_file, bg_dir, fg_dir, mask_dir, datatype='train'):
    """
    load json file and return required information
    Args:
        json_file(str): json file with train/test information 
        bg_dir(str): folder with background images
        fg_dir(str): folder with foreground images
        mask_dir(str): folder with foreground masks
        datatype(str): "train" or "test"
    return:
        index(int): the sample index
        background image path, foreground image path, foreground mask image
        foreground scale, the locations of positive/negative pixels
    """
    f_json = json.load(open(json_file, 'r'))
    return [
        (
            index,                                      
            row['scID'].rjust(12,'0'),
            os.path.join(bg_dir, "%012d.jpg" % int(row['scID'])),  # background image path
            os.path.join(fg_dir, "{}/{}_{}_{}_{}.jpg".format(datatype, int(row['annID']), int(row['scID']), # scaled foreground image path
                                                             int(row['newWidth']), int(row['newHeight']))),
          
            os.path.join(mask_dir, "{}/{}_{}_{}_{}.jpg".format(datatype, int(row['annID']), int(row['scID']), # scaled foreground mask path
                                                               int(row['newWidth']), int(row['newHeight']))),
            row['scale'],
            row['pos_label'], row['neg_label'],
            os.path.join(fg_dir, "foreground/{}.jpg".format(int(row['annID']))), # original foreground image path
            os.path.join(fg_dir, "foreground/mask_{}.jpg".format(int(row['annID']))), # original foreground mask path
            int(row['newWidth']), int(row['newHeight']) # scaled foreground width and height
        )
        for index, row in enumerate(f_json)
    ]


def _to_center(bbox):
    """conver bbox to center pixel"""
    x, y, width, height = bbox
    return x + width // 2, y + height // 2


def create_loader(table_path, bg_dir, fg_dir, mask_dir, in_size, datatype, batch_size, num_workers, shuffle):
    dset = CPDataset(table_path, bg_dir, fg_dir, mask_dir, in_size, datatype)
    data_loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return data_loader


def make_composite(fg_img, mask_img, bg_img, pos, isflip=False):
    x, y, w, h = pos
    bg_h = bg_img.height
    bg_w = bg_img.width
    # resize foreground to expected size [h, w]
    fg_transform = transforms.Compose([ 
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])
    top = max(y, 0)
    bottom = min(y + h, bg_h)
    left = max(x, 0)
    right = min(x + w, bg_w)
    fg_img_ = fg_transform(fg_img)
    mask_img_ = fg_transform(mask_img)
    fg_img = torch.zeros(3, bg_h, bg_w)
    mask_img = torch.zeros(3, bg_h, bg_w)
    fg_img[:, top:bottom, left:right] = fg_img_[:, top - y:bottom - y, left - x:right - x]
    mask_img[:, top:bottom, left:right] = mask_img_[:, top - y:bottom - y, left - x:right - x]
    bg_img = transforms.ToTensor()(bg_img)
    blended = fg_img * mask_img + bg_img * (1 - mask_img)
    com_pic = transforms.ToPILImage()(blended).convert('RGB')
    if isflip == False:
        com_pic = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        )(com_pic)
        mask_img = transforms.ToPILImage()(mask_img).convert('L')
        mask_img = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        )(mask_img)
        com_pic = torch.cat((com_pic, mask_img), dim=0)
    else:
        com_pic = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor()
            ]
        )(com_pic)
        mask_img = transforms.ToPILImage()(mask_img).convert('L')
        mask_img = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor()
            ]
        )(mask_img)
        com_pic = torch.cat((com_pic, mask_img), dim=0)
    return com_pic

def make_composite_PIL(fg_img, mask_img, bg_img, pos, return_mask=False):
    x, y, w, h = pos
    bg_h = bg_img.height
    bg_w = bg_img.width
  
    top = max(y, 0)
    bottom = min(y + h, bg_h)
    left = max(x, 0)
    right = min(x + w, bg_w)
    fg_img_ = fg_img.resize((w,h))
    mask_img_ = mask_img.resize((w,h))
    
    fg_img_ = np.array(fg_img_)
    mask_img_ = np.array(mask_img_, dtype=np.float_)/255
    bg_img = np.array(bg_img)
    
    fg_img = np.zeros((bg_h, bg_w, 3), dtype=np.uint8) 
    mask_img  = np.zeros((bg_h, bg_w, 3), dtype=np.float_) 
    
    fg_img[top:bottom, left:right, :] = fg_img_[top - y:bottom - y, left - x:right - x, :]
    mask_img[top:bottom, left:right, :] = mask_img_[top - y:bottom - y, left - x:right - x, :]
    composite_img = fg_img * mask_img + bg_img * (1 - mask_img)
    
    
    composite_img = Image.fromarray(composite_img.astype(np.uint8))
    if return_mask==False:
        return composite_img
    else:
        composite_msk = Image.fromarray((mask_img*255).astype(np.uint8))
        return composite_img, composite_msk
