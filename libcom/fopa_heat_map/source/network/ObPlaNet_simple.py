import sys
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.append("..")
from ..backbone.ResNet import Backbone_ResNet18_in3, Backbone_ResNet18_in3_1
from .BaseBlocks import BasicConv2d
from .DynamicModules import simpleDFN
from .tensor_ops import cus_sample, upsample_add

class ObPlaNet_resnet18(nn.Module):
    def __init__(self, pretrained=True, ks=3, scale=3, weight_path=None):
        super(ObPlaNet_resnet18, self).__init__()
        self.Eiters = 0
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.to_pil = transforms.ToPILImage()
        self.scale = scale

        self.add_mask = True

        (
            self.bg_encoder1,
            self.bg_encoder2,
            self.bg_encoder4,
            self.bg_encoder8,
            self.bg_encoder16,
        ) = Backbone_ResNet18_in3(pretrained, weight_path)
        
        # freeze background encoder
        for p in self.parameters():
            p.requires_grad = False

        (
            self.fg_encoder1,
            self.fg_encoder2,
            self.fg_encoder4,
            self.fg_encoder8,
            self.fg_encoder16,
            self.fg_encoder32,
        ) = Backbone_ResNet18_in3_1(pretrained=pretrained)

        if self.add_mask:
            self.mask_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # dynamic conv
        self.fg_trans16 = nn.Conv2d(512, 64, 1)
        self.fg_trans8 = nn.Conv2d(256, 64, 1)
        self.selfdc_16 = simpleDFN(64, 64, 512, ks, 4)
        self.selfdc_8 = simpleDFN(64, 64, 512, ks, 4)

        self.upconv16 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(512, 2, 1)

    def forward(self, bg_in_data, fg_in_data, mask_in_data=None, mode='test'):
        """
        Args:
            bg_in_data: (batch_size * 3 * H * W) background image
            fg_in_data: (batch_size * 3 * H * W) scaled foreground image
            mask_in_data: (batch_size * 1 * H * W) scaled foreground mask
            mode: "train" or "test"
        """
        if ('train' == mode):
            self.Eiters += 1
            
        # extract background and foreground features
        black_mask = torch.zeros(mask_in_data.size()).to(mask_in_data.device)
        bg_in_data_ = torch.cat([bg_in_data, black_mask], dim=1)
        bg_in_data_1 = self.bg_encoder1(bg_in_data_)  # torch.Size([2, 64, 128, 128])
        fg_cat_mask = torch.cat([fg_in_data, mask_in_data], dim=1)
        fg_in_data_1 = self.fg_encoder1(fg_cat_mask)  # torch.Size([2, 64, 128, 128])
  

        bg_in_data_2 = self.bg_encoder2(bg_in_data_1)  # torch.Size([2, 64, 64, 64])
        fg_in_data_2 = self.fg_encoder2(fg_in_data_1)  # torch.Size([2, 64, 128, 128])
        bg_in_data_4 = self.bg_encoder4(bg_in_data_2)  # torch.Size([2, 128, 32, 32])
        fg_in_data_4 = self.fg_encoder4(fg_in_data_2)  # torch.Size([2, 64, 64, 64])
        del fg_in_data_1, fg_in_data_2

        bg_in_data_8 = self.bg_encoder8(bg_in_data_4)  # torch.Size([2, 256, 16, 16])
        fg_in_data_8 = self.fg_encoder8(fg_in_data_4)  # torch.Size([2, 128, 32, 32])
        bg_in_data_16 = self.bg_encoder16(bg_in_data_8)  # torch.Size([2, 512, 8, 8])
        fg_in_data_16 = self.fg_encoder16(fg_in_data_8)  # torch.Size([2, 256, 16, 16])
        fg_in_data_32 = self.fg_encoder32(fg_in_data_16)  # torch.Size([2, 512, 8, 8])

        in_data_8_aux = self.fg_trans8(fg_in_data_16)  # torch.Size([2, 64, 16, 16])
        in_data_16_aux = self.fg_trans16(fg_in_data_32)  # torch.Size([2, 64, 8, 8])

        # Unet decoder
        bg_out_data_16 = bg_in_data_16  # torch.Size([2, 512, 8, 8])

        bg_out_data_8 = self.upsample_add(self.upconv16(bg_out_data_16), bg_in_data_8)  # torch.Size([2, 256, 16, 16])
        bg_out_data_4 = self.upsample_add(self.upconv8(bg_out_data_8), bg_in_data_4)  # torch.Size([2, 128, 32, 32])
        bg_out_data_2 = self.upsample_add(self.upconv4(bg_out_data_4), bg_in_data_2)  # torch.Size([2, 64, 64, 64])
        bg_out_data_1 = self.upsample_add(self.upconv2(bg_out_data_2), bg_in_data_1)  # torch.Size([2, 64, 128, 128])
        del bg_out_data_2, bg_out_data_4, bg_out_data_8, bg_out_data_16

        bg_out_data = self.upconv1(self.upsample(bg_out_data_1, scale_factor=2))  # torch.Size([2, 64, 256, 256])

        # fuse foreground and background features using dynamic conv
        fuse_out = self.upsample_add(self.selfdc_16(bg_out_data_1, in_data_8_aux), \
                                     self.selfdc_8(bg_out_data, in_data_16_aux))  # torch.Size([2, 64, 256, 256])

        out_data = self.classifier(fuse_out)  # torch.Size([2, 2, 256, 256])

        return out_data, fuse_out
