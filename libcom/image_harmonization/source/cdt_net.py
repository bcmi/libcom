import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .functions import ConvBlock, GaussianSmoothing, ChannelAttention, LUT, UNetDecoder, Resnet50


class CDTNet(nn.Module):
    def __init__(
        self,depth=4,
        norm_layer=nn.BatchNorm2d, batchnorm_from=2,
        attend_from=2, attention_mid_k=0.5,
        image_fusion=True,
        ch=32, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(CDTNet, self).__init__()
        self.n_lut = 6
        self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
        self.backbone = "Resnet50"
        self.base_resolution = 256
        #1.pix2pix
        self.encoder = Resnet50()    
        
        #2.rgb2rgb
        self.lut = LUT(256, self.n_lut, backbone='issam')

    def normalize(self, tensor):
        self.mean = self.mean.to(tensor.device)
        self.std = self.std.to(tensor.device)
        return (tensor - self.mean) / self.std


    def forward(self, image, mask, backbone_features=None):  

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
        normed_image = self.normalize(image)
        x = torch.cat((normed_image, mask), dim=1)
        basic_input = F.interpolate(x, size=(self.base_resolution,self.base_resolution), mode='bilinear', align_corners=True).detach()
        intermediates = self.encoder(basic_input, backbone_features)
        #for item in intermediates:
            #print(item.shape)
        lut_output = self.lut(intermediates, image, mask)
        return lut_output


class SpatialSeparatedAttention(nn.Module):
    def __init__(self, in_channels, norm_layer, activation, mid_k=2.0):
        super(SpatialSeparatedAttention, self).__init__()
        self.background_gate = ChannelAttention(in_channels)
        self.foreground_gate = ChannelAttention(in_channels)
        self.mix_gate = ChannelAttention(in_channels)

        mid_channels = int(mid_k * in_channels)
        self.learning_block = nn.Sequential(
            ConvBlock(
                in_channels, mid_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
            ConvBlock(
                mid_channels, in_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
        )
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask):
        mask = self.mask_blurring(F.interpolate(
            mask, size=x.size()[-2:],
            mode='bilinear', align_corners=True
        ))
        background = self.background_gate(x)
        foreground = self.learning_block(self.foreground_gate(x))
        mix = self.mix_gate(x)
        output = mask * (foreground + mix) + (1 - mask) * background
        return output
