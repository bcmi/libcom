import torch
# from clip2 import build_model
# import clip
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os
import torch
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__))
from local_clip import FrozenCLIPImageEmbedder

model = FrozenCLIPImageEmbedder()

import torch
import torch.nn.functional as F
from torchvision import transforms

def process_image_with_mask(mask, image):
    
    mask = mask.unsqueeze(-1)  # (b, 512, 512) -> (b, 512, 512, 1)

    masked_image = image * mask  # (b, 512, 512, 3)

    flipped_vertical = torch.flip(masked_image, dims=[1]) 
    flipped_horizontal = torch.flip(masked_image, dims=[2]) 

    flipped_images = torch.cat([flipped_vertical, flipped_horizontal], dim=0)  # (2b, 512, 512, 3)

    resized_images = F.interpolate(flipped_images.permute(0, 3, 1, 2), size=(224, 224), mode='bilinear', align_corners=False)
    resized_images = resized_images.permute(0, 2, 3, 1)  # (2b, 224, 224, 3)

    black_background = torch.zeros_like(resized_images)  # (2b, 224, 224, 3)

    final_images = torch.where(resized_images != 0, resized_images, black_background)

    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    final_images = final_images.permute(0, 3, 1, 2)  
    final_images = normalize(final_images)

    return final_images


def crop_and_resize_shadow(shadow_img, shadow_mask, target_size=224):

    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(contours[0])  

    shadow_img = shadow_img * shadow_mask[:, :, None]  
    shadow_area = shadow_img[y:y+h, x:x+w]

    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    shadow_resized = cv2.resize(shadow_area, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    background = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = shadow_resized

    return Image.fromarray(background)

def load_image_and_mask(image_path, mask_path):

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

def extract_clip_features(image, model, preprocess, device):

    image = preprocess(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        local_f, global_f = model(image)  
    return local_f.cpu(), global_f.cpu()

def extract_clip_features_2(image, model, device):

    model = model.to(device)
    with torch.no_grad():
        local_f, global_f = model(image)  
    return torch.cat([local_f, global_f], dim=1)

def generate_global_dictionary(features, num_samples=10000, num_clusters=512, D=1024):
    features_flat = features.view(-1, D)

    indices = np.random.choice(features_flat.shape[0], num_samples, replace=True) #, replace=False
    sampled_features = features_flat[indices]

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sampled_features.numpy())
    
    return torch.tensor(kmeans.cluster_centers_)


def generate_shadow_clip_features(shadow_img, shadow_mask):
    device = shadow_img.device
    crop_image = process_image_with_mask(shadow_mask, shadow_img[:,:,:,:3])
    clip_features = extract_clip_features_2(crop_image, model, device)
    b = clip_features.size(0) // 2  # 计算 b
    split_tensors = torch.split(clip_features, b, dim=0)  
    
    concatenated_tensor = torch.cat(split_tensors, dim=1)
    return concatenated_tensor