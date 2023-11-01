from inspect import isfunction
import math
from numpy import inner
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import os, sys
from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.attention import CrossAttention, zero_module, Normalize
from torchvision.ops import roi_align

class FDN(nn.Module):
    # Spatially-Adaptive Normalization, homepage: https://nvlabs.github.io/SPADE/
    # this code borrows from https://github.com/ShihaoZhaoZSH/Uni-ControlNet/blob/591036b78d13fd17b002ecd3be44d7c84473b47c/models/local_adapter.py#L31  
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        self.conv_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta  = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, context):
        normalized = self.param_free_norm(x)
        assert context.size()[2:] == x.size()[2:]
        gamma = self.conv_gamma(context)
        beta = self.conv_beta(context)
        out = normalized * gamma + beta
        return out


class LocalRefineBlock(nn.Module):
    def __init__(self, in_channels, n_heads=1, d_head=320,
                 depth=1, dropout=0., context_dim=1024, roi_size=9, 
                 add_positional_embedding=False, block_spade=False):
        super().__init__()
        n_heads, d_head  = 1, in_channels
        self.in_channels = in_channels
        self.heads = n_heads 
        inner_dim = n_heads * d_head
        self.add_positional_embedding = add_positional_embedding
        if self.add_positional_embedding:
            self.local_positional_embedding = nn.Parameter(
                torch.randn(roi_size ** 2, in_channels) / in_channels ** 0.5)
        self.local_norm   = Normalize(in_channels)
        self.context_norm = Normalize(context_dim)
        self.scale = d_head ** -0.5
        self.roi_size = roi_size
        self.local_proj_in = nn.Conv2d(in_channels+2,
                                    inner_dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.context_conv  = nn.Conv2d(context_dim,
                                inner_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.local_proj_out = nn.Conv2d(
                                context_dim,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.SPADE = FDN(in_channels, context_dim)#zero_module(FDN(in_channels, context_dim))

    def forward(self, global_x, context, **kwargs):
        indicator, bbox, mask, mask_method = kwargs.get('indicator'), kwargs.get('bbox'), kwargs.get('mask'), kwargs.get('mask_method')
        context_map = rearrange(context, 'b (h w) c -> b c h w', h=16)
        b, c, h, w = global_x.shape
        indices = torch.arange(b).reshape((-1,1)).to(bbox.dtype)
        indices = indices.to(bbox.device)
        idx_bbox = torch.cat([indices, bbox], dim=1) # B,5
        x = roi_align(global_x, idx_bbox, output_size=self.roi_size) # B,C,roi_size,roi_size
        # do something on local feature
        if self.add_positional_embedding:
            x = x + self.local_positional_embedding[None,:,:].to(x.dtype)
        # cross-attention
        x = self.local_norm(x)
        ind_map = repeat(indicator, 'b n -> b n h w', h=x.shape[-2], w=x.shape[-1])
        ind_map = ind_map.to(x.dtype)
        x = torch.cat([x, ind_map], dim=1)
        q = self.local_proj_in(x)
        q = rearrange(q, 'b c h w -> b (h w) c')
        # k = self.context_conv(torch.cat([self.context_norm(context_map), ind_map], dim=1))
        k = self.context_conv(self.context_norm(context_map))
        k = rearrange(k, 'b c h w -> b (h w) c')
        # v = self.local_proj_out(torch.cat([context_map, ind_map], dim=1))
        v = self.local_proj_out(context_map)
        v = rearrange(v, 'b c h w -> b (h w) c')        
        sim  = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = sim.softmax(dim=-1) # b,256,256
        x = torch.einsum('b i j, b j d -> b i d', attn, v)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.roi_size, w=self.roi_size)
        # align conditional foreground feature map with roi feature with using their cross-attention map
        align_context = torch.einsum('b i j, b j d -> b i d', attn, context)
        align_context = rearrange(align_context, 'b (h w) c -> b c h w', h=self.roi_size, w=self.roi_size)
        # update local feature with Spatially-Adaptive Normalization
        if mask != None:
            # only performing SPADE in the foreground area 
            flat_mask = rearrange(mask, 'b c h w -> b (h w) c')
            if mask_method == 'argmax':
                thresh = torch.max(attn, dim=-1)[0].unsqueeze(-1)
                attn   = torch.where(attn >= thresh, torch.ones_like(attn), torch.zeros_like(attn))
            align_mask = torch.einsum('b i j, b j d -> b i d', attn, flat_mask)
            align_mask = torch.clamp(align_mask, max=1.0, min=0.0)
            align_mask = rearrange(align_mask, 'b (h w) c -> b c h w', h=self.roi_size, w=self.roi_size)
            x = torch.where(align_mask > 0.5, self.SPADE(x, align_context), x) 
        else:
            align_mask = None
            x = self.SPADE(x, align_context)
        # paste the updated region feature into original global feature
        bbox_int = (bbox * h).int()
        bbox_int[:,2:] = torch.maximum(bbox_int[:,2:], bbox_int[:,:2] + 1)
        for i in range(b):
            x1,y1,x2,y2  = bbox_int[i]
            local_res = F.interpolate(x[i:i+1], (y2-y1,x2-x1))
            local_x0  = global_x[i:i+1,:,y1:y2,x1:x2]                
            # update foreground region feature by residual learning
            global_x[i:i+1,:,y1:y2,x1:x2] = local_res + local_x0 
        
        if align_mask != None:
            return global_x, align_mask
        else:
            return global_x, attn

if __name__ == '__main__':
    local_att = LocalRefineBlock(320, 1, 320, context_dim=1024, roi_size=16)
    H = W = 64
    feature = torch.randn((1, 1, H, W)).float()
    feature = feature.repeat(3, 320, 1, 1).float()
    bbox = torch.tensor([[0.,0.,0.3,0.3],
                         [0.1,0.1,0.5,0.5],
                         [0.2,0.2,0.4,0.4]]).reshape((-1,4)).float()
    indicator = torch.randint(0, 2, (3, 2))
    context = torch.randn(3, 256, 1024)
    out = local_att(feature, context, bbox=bbox, indicator=indicator)
    if isinstance(out, tuple):
        print([o.shape for o in out])
    else:
        print(out.shape)