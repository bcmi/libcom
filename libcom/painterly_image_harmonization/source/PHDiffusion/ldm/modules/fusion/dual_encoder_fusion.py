import torch
from torch import nn, einsum
from libcom.painterly_image_harmonization.source.PHDiffusion.ldm.modules.attention import CrossAttention
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
from libcom.painterly_image_harmonization.source.PHDiffusion.ldm.modules.diffusionmodules.util import checkpoint
import numpy

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# our dual encoder fusion
class CrossAttentionInteraction(nn.Module):
    def __init__(self, in_channels, n_heads=8, d_head=64,
                 dropout=0.):
        super().__init__()

        inner_dim = n_heads * d_head

        self.norm = Normalize(inner_dim)

        self.crossAtt_1 = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                         context_dim=inner_dim)

        self.crossAtt_2 = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                         context_dim=inner_dim)

        self.fc = nn.Conv1d(in_channels=inner_dim * 2, out_channels=inner_dim, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=1)

    def downsample(self, image_tensor, width, height):
        image_upsample_tensor = torch.nn.functional.interpolate(image_tensor, size=[width, height])
        image_upsample_tensor = image_upsample_tensor.clamp(0, 1)
        return image_upsample_tensor

    def forward(self, adapter_feature, unet_feature, fg_mask):
        ori_adapter_feature = adapter_feature

        b, c, h, w = adapter_feature.shape
        fg_mask = self.downsample(fg_mask, h, w)

        original_h = adapter_feature.shape[-1]

        adapter_feature = self.norm(adapter_feature)

        fg = adapter_feature * fg_mask

        fg = rearrange(fg, 'b c h w -> b c (h w)').contiguous()

        fg = rearrange(fg, 'b c h -> b h c').contiguous()

        bg = adapter_feature * (1 - fg_mask)

        bg = rearrange(bg, 'b c h w -> b (h w) c').contiguous()

        adapter_feature = self.crossAtt_1(fg, bg, mask=fg_mask, is_foreground='ada')

        adapter_feature = adapter_feature.permute(0, 2, 1)

        unet_feature = self.norm(unet_feature)

        unet_fg = unet_feature * fg_mask
        unet_fg = rearrange(unet_fg, 'b c h w -> b c (h w)').contiguous()

        unet_fg = rearrange(unet_fg, 'b c h -> b h c').contiguous()

        unet_bg = unet_feature * (1 - fg_mask)
        unet_bg = rearrange(unet_bg, 'b c h w -> b (h w) c').contiguous()
        unet_feature = self.crossAtt_2(unet_fg, unet_bg, mask=fg_mask, is_foreground='unet')

        unet_feature = unet_feature.permute(0, 2, 1)

        interact_feature = self.fc(torch.cat([adapter_feature, unet_feature], dim=1))  # 1 640 mm -> 1 320 mm

        interact_feature = interact_feature.repeat(1, 1, int(original_h * original_h / interact_feature.shape[-1]))

        b, c, h = interact_feature.shape

        interact_feature = interact_feature.reshape(b, c, original_h, original_h)

        interact_feature = self.conv1(interact_feature)

        new_adapter_feature = interact_feature * fg_mask + ori_adapter_feature * (1 - fg_mask)

        return new_adapter_feature

