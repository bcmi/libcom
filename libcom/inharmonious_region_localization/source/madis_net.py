import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import argparse
from .functions import HDRPointwiseNN, DIRLNet


class MadisNet(nn.Module):
    def __init__(self):
        super(MadisNet, self).__init__()

        opt = argparse.Namespace(
            dataset_root='/data/share/IHD',
            name='',
            gpu_ids='0',
            checkpoints_dir='./checkpoints/dirl',
            preprocess='resize',
            no_flip=True,
            num_threads=4,
            batch_size=1,
            load_size=256,
            crop_size=256,
            serial_batches=False,
            epoch='latest',
            mean='0.485, 0.456, 0.406',
            std='0.229, 0.224, 0.225',
            display_freq=400,
            print_freq=50,
            input_nc=3,
            output_nc=1,
            save_latest_freq=5000,
            save_epoch_freq=5,
            save_by_iter=False,
            continue_train=False,
            epoch_count=1,
            phase='test',
            resume=-2,
            nepochs=60,
            pretrain_nepochs=50,
            weight_decay=0.0001,
            beta1=0.9,
            lr=0.0001,
            lr_policy='linear',
            lr_decay_iters=50,
            sync_bn=False,
            lambda_attention=1,
            lambda_detection=1,
            lambda_ssim=0,
            lambda_iou=1,
            lambda_tri=0.001,
            lambda_reg=0.001,
            is_train=0,
            is_val=0,
            port='tcp://192.168.1.201:12345',
            local_rank=0,
            backbone='resnet34',
            ggd_ch=32,
            mda_mode='mask',
            loss_mode='',
            model='dirl',
            pretrain_path='',
            luma_bins=16,
            channel_multiplier=1,
            spatial_bin=16,
            batch_norm=True,
            net_input_size=256,
            net_output_size=256,
            m=0.001,
            theta=0.7
        )
        self.ihdrnet = HDRPointwiseNN(opt)
        self.g = DIRLNet(opt,3)

    def forward(self, img):
        retouched_img, _ = self.ihdrnet(img, img)
        delta_img = retouched_img
        mask_main = self.g(delta_img)['mask']
        return mask_main
