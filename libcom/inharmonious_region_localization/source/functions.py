import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from .blocks import Conv2d_cd, ResNetBlock, Conv2dBlock, Conv2dBlock, BasicBlock, BasicConv
import numpy as np
from torchvision import models

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, mode='self'):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        
        self.mode = mode
        reduction = 8
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//reduction , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//reduction , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.D = np.sqrt(in_dim//reduction)
        self.softmax  = nn.Softmax(dim=-1) #
        
    def forward(self,x, y=None):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        if self.mode == 'self':
            proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy / self.D) # BX (N) X (N) 
            proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        else:
            proj_key = self.key_conv(y).view(m_batchsize, -1, width*height)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy / self.D) # BX (N) X (N) 
            proj_value = self.value_conv(y).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        
        if self.mode == 'self':
            out = self.gamma*out + x
        else:
            out = torch.cat([out, x],dim=1)
        return out


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap, mode='hdr'): 
        # Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)], indexing='ij') # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([hg, wg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear', align_corners=True) # Nx12xHxW
        return coeff.squeeze(2)

class ApplyCoeffs(nn.Module):
    def __init__(self, use_norm=False):
        super(ApplyCoeffs, self).__init__()
        self.use_norm = use_norm

    def denormalize(self, x, isMask=False):
        if isMask:
            mean = 0
            std=1
        else:
            mean = torch.zeros_like(x)
            mean[:,0,:,:] = .485
            mean[:,1,:,:] = .456
            mean[:,2,:,:] = .406
            std = torch.zeros_like(x)
            std[:,0,:,:] = 0.229
            std[:,1,:,:] = 0.224
            std[:,2,:,:] = 0.225
        x = (x*std + mean) #*255
        return x # change the range into [0,1]
    
    def norm(self, x):
        mean = torch.zeros_like(x)
        mean[:,0,:,:] = .485
        mean[:,1,:,:] = .456
        mean[:,2,:,:] = .406
        std = torch.zeros_like(x)
        std[:,0,:,:] = 0.229
        std[:,1,:,:] = 0.224
        std[:,2,:,:] = 0.225
        x = (x - mean) / std #*255
        return x

    def forward(self, coeff, full_res_input):
        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        full_res_input = self.denormalize(full_res_input)
        # coeff[:,:,:20] = coeff[:,:,50:70]
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        
        # return self.norm(torch.cat([R, G, B], dim=1))
        if self.use_norm:
            return self.norm(torch.cat([R, G, B], dim=1))
        else:
            return torch.cat([R, G, B], dim=1)

class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = Conv2dBlock(3, 16, ks=1, st=1, padding=0, norm='bn')
        self.conv2 = Conv2dBlock(16, 1, ks=1, st=1, padding=0, norm='none', activation='tanh') #nn.Tanh, nn.Sigmoid

    def forward(self, x):
        return self.conv2(self.conv1(x))#.squeeze(1)

class Coeffs(nn.Module):
    def __init__(self, nin=4, nout=3, params=None):
        super(Coeffs, self).__init__()
        self.params = params
        self.nin = nin 
        self.nout = nout
        
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']
        bn = params['batch_norm']
        
        theta = params['theta']
        nsize = params['net_input_size']
        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize/sb))
        self.splat_features = nn.ModuleList()
        self.lp_features = nn.ModuleList()
        prev_ch = 3 #3
        # Downsample
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(Conv2d_cd(prev_ch, cm*(2**i)*lb, 3, 1, 1, use_bn=use_bn, actv='relu', theta=theta))
            self.splat_features.append(nn.MaxPool2d(2,2,0))
            prev_ch = splat_ch = cm*(2**i)*lb
        # ResNet Blocks
        self.res_blks = nn.ModuleList()
        for i in range(3):
            self.res_blks.append(ResNetBlock(prev_ch, prev_ch))
        #Self-attention
        self.sa = SelfAttention(prev_ch)
        
        self.conv_out = nn.Sequential(*[
            Conv2dBlock(prev_ch, 8*cm*lb, ks=3, st=1, padding=1, norm='bn'),
            Conv2dBlock(8*cm*lb, lb*nin*nout, ks=1, st=1, padding=0, norm='none', activation='none')
        ])
        
        # predicton
        self.conv_out = Conv2dBlock(8*cm*lb, lb*nout*nin, ks=1, st=1, padding=0, norm='none', activation='none')

   
    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)   
            
        for layer in self.res_blks:
            x = layer(x)
        
        x = self.sa(x)
        x = self.conv_out(x) # 1,96,16,16
        
        s = x.shape
        y = torch.stack(torch.split(x, self.nin*self.nout, 1),2) # B x Coefs x Luma x Spatial x Spatial -> (B, 12,8,16,16)
        return y


class HDRPointwiseNN(nn.Module):

    def __init__(self, opt):
        super(HDRPointwiseNN, self).__init__()
        params = {'luma_bins':opt.luma_bins, 'channel_multiplier':opt.channel_multiplier, 'spatial_bin':opt.spatial_bin, 
                'batch_norm':opt.batch_norm, 'net_input_size':opt.net_input_size, 'theta':opt.theta}
        self.coeffs = Coeffs(params=params)
        self.guide = GuideNN(params=params)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

        self.mean = [.485, .456, .406]
        self.std = [.229, .224, .225]
        self.max_val = [(1-m)/s for m,s in zip(self.mean, self.std)]
        self.min_val = [(0-m)/s for m,s in zip(self.mean, self.std)]

    def clip(self, x):
        y = x.new(x.size())
        for i in range(3):
            y[:,i,:,:] = torch.clamp(x[:,i,:,:], min=self.min_val[i], max=self.max_val[i])
        return y

    def norm(self, x):
        mean = torch.zeros_like(x)
        mean[:,0,:,:] = .485
        mean[:,1,:,:] = .456
        mean[:,2,:,:] = .406
        std = torch.zeros_like(x)
        std[:,0,:,:] = 0.229
        std[:,1,:,:] = 0.224
        std[:,2,:,:] = 0.225
        x = (x - mean) / std #*255
        return x

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        illu_out = self.apply_coeffs(slice_coeffs, fullres).sigmoid()
        
        out = self.clip(illu_out)
        return out, guide
    

## ---------------------Bi-directional Feature Integration -----------------
class BidirectionFeatureIntegration(nn.Module):
    def __init__(self, in_ch_list, out_ch=64, fusion_mode='h2l'):
        super(BidirectionFeatureIntegration, self).__init__()
        self.n_input = len(in_ch_list)
        assert self.n_input > 0
        self.fusion_mode = fusion_mode
        self.downsample = nn.AvgPool2d(3,2,1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(True)
        if self.fusion_mode == 'h2l' or self.fusion_mode == 'l2h':
            l_in_ch = in_ch_list[0]
            h_in_ch = in_ch_list[1]

            self.top_down = Conv2dBlock(h_in_ch, l_in_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            self.bottom_up = Conv2dBlock(l_in_ch, h_in_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)
            if self.fusion_mode == 'h2l':
                in_ch_ratio = 2
                self.h_concat = Conv2dBlock(h_in_ch * in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
                self.l_concat = Conv2dBlock(l_in_ch * in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            elif self.fusion_mode == 'l2h':
                in_ch_ratio = 2
                self.l_concat = Conv2dBlock(l_in_ch*in_ch_ratio, out_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)
                self.h_concat = Conv2dBlock(h_in_ch*in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
                
        elif self.fusion_mode == 'hl2m' or self.fusion_mode == 'lh2m':
            l_in_ch = in_ch_list[0]
            m_in_ch = in_ch_list[1]
            h_in_ch = in_ch_list[2]
            
            self.top_down_h2m = Conv2dBlock(h_in_ch, m_in_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            self.top_down_m2l = Conv2dBlock(m_in_ch, l_in_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            self.bottom_up_m2h = Conv2dBlock(m_in_ch, h_in_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)
            self.bottom_up_l2m = Conv2dBlock(l_in_ch, m_in_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)

            in_ch_ratio = 2
            self.l_concat = Conv2dBlock(l_in_ch * in_ch_ratio, out_ch, 3,2,1, norm='bn', activation='relu', activation_first=True)
            self.m_concat = Conv2dBlock(m_in_ch * in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
            self.h_concat = Conv2dBlock(h_in_ch * in_ch_ratio, out_ch, 3,1,1, norm='bn', activation='relu', activation_first=True)
        else:
            raise NameError("Unknown mode:\t{}".format(fusion_mode))

    def forward(self, xl, xm=None, xh=None):
        if self.fusion_mode == 'h2l' or self.fusion_mode == 'l2h':
            # Bottom        xl ----> xh            Up
            #               |         \ 
            # Down           \   xl <----  xh      Top
            #                 \  /     \  /
            #                   C -> + <-C
            #                        ↓
            #                       out
            top_down_results = [xh]
            xh2l = self.top_down(F.interpolate(xh, scale_factor=2))
            top_down_results.insert(0, xl + xh2l)

            bottom_up_results = [xl]
            xl2h = self.bottom_up(xl)
            bottom_up_results.append(xh+xl2h)

            xl_cat = torch.cat([top_down_results[0],bottom_up_results[0]], dim=1)
            xh_cat = torch.cat([top_down_results[1],bottom_up_results[1]], dim=1)
            if self.fusion_mode == 'h2l':
                xh_cat = self.h_concat(F.interpolate(xh_cat, scale_factor=2))
                xl_cat = self.l_concat(xl_cat)
                
            elif self.fusion_mode == 'l2h':
                xh_cat = self.h_concat(xh_cat)
                xl_cat = self.l_concat(xl_cat)
            
            xout = xh_cat + xl_cat
            
        elif self.fusion_mode == 'hl2m' or self.fusion_mode== 'lh2m':
            # Bottom       xl ---->  xm ----> xh            Up
            #               \         \        \
            # Down           \   xl <----  xm <---- xh      Top
            #                 \  /      \  /     \  / 
            #                   C  ---->  C  <---- C
            #                             ↓
            #                            out
            top_down_results = [xh] 
            xh2m = self.top_down_h2m(F.interpolate(xh, scale_factor=2)) + xm
            top_down_results.insert(0, xh2m)
            xm2l = self.top_down_m2l(F.interpolate(xh2m, scale_factor=2)) + xl
            top_down_results.insert(0, xm2l)

            bottom_up_results = [xl]
            xl2m = self.bottom_up_l2m(xl) + xm
            bottom_up_results.append(xl2m)
            xm2h = self.bottom_up_m2h(xl2m) + xh
            bottom_up_results.append(xm2h)

            xl_cat = torch.cat([top_down_results[0],bottom_up_results[0]], dim=1)
            xm_cat = torch.cat([top_down_results[1],bottom_up_results[1]], dim=1)
            xh_cat = torch.cat([top_down_results[2],bottom_up_results[2]], dim=1)
            
            xl_cat = self.l_concat(xl_cat)
            xm_cat = self.m_concat(xm_cat)
            xh_cat = self.h_concat(F.interpolate(xh_cat, scale_factor=2))

            xout = xl_cat + xm_cat + xh_cat
        return xout

class Transition(nn.Module):
    def __init__(self, in_ch_list, out_ch_list):
        super(Transition, self).__init__()
        inch0, inch1, inch2, inch3, inch4 = in_ch_list
        outch0, outch1, outch2, outch3, outch4 = out_ch_list
        
        self.im0 = BidirectionFeatureIntegration([inch0,inch1], outch0, fusion_mode='h2l')
        self.im1 = BidirectionFeatureIntegration([inch0,inch1, inch2], outch1, fusion_mode='hl2m')
        self.im2 = BidirectionFeatureIntegration([inch1,inch2, inch3], outch2, fusion_mode='hl2m')
        self.im3 = BidirectionFeatureIntegration([inch2,inch3, inch4], outch3, fusion_mode='hl2m')
        self.im4 = BidirectionFeatureIntegration([inch3,inch4], outch4, fusion_mode='l2h')

    def forward(self, xs, gc=None):
        out_xs = []
        out_xs.append(self.im0(xl=xs[0], xh=xs[1]))
        out_xs.append(self.im1(xl=xs[0], xm=xs[1], xh=xs[2]))
        out_xs.append(self.im2(xl=xs[1], xm=xs[2], xh=xs[3]))
        out_xs.append(self.im3(xl=xs[2], xm=xs[3], xh=xs[4]))
        out_xs.append(self.im4(xl=xs[3], xh=xs[4]))
        return out_xs

class VanillaTransport(nn.Module):
    def __init__(self, in_ch_list, out_ch_list):
        super(VanillaTransport, self).__init__()
        self.model = nn.ModuleDict()
        idx = 0
        for in_ch, out_ch in zip(in_ch_list, out_ch_list):
            self.model[f'conv_{idx}'] = nn.Conv2d(in_ch_list[idx], out_ch_list[idx], 3,1,1)
            idx += 1
        
    def forward(self, xs):
        x0,x1,x2,x3,x4 = xs
        y0 = self.model['conv_0'](x0) # 224, 64
        y1 = self.model['conv_1'](x1) # 112,64+128
        y2 = self.model['conv_2'](x2) # 56, 256+512+512
        y3 = self.model['conv_3'](x3) # 28, 512+512
        y4 = self.model['conv_4'](x4) # 14, 512
        return [y0,y1,y2,y3,y4]

## -----------------Mask-guided Dual Attention ---------------------


class ECABlock(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

## SA
def _get_kernel(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    """
        normalization
    :param in_:
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)

##

class SpatialGate(nn.Module):
    def __init__(self, in_dim=2, mask_mode='mask'):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.mask_mode = mask_mode
        
        self.spatial = nn.Sequential(*[
            BasicConv(in_dim, in_dim, 3, 1, 1),
            BasicConv(in_dim, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2,  relu=False)
        ])
        
        if 'gb' in mask_mode.split('_')[-1]:
            print("Using Gaussian Filter in mda!")
            gaussian_kernel = np.float32(_get_kernel(31, 4))
            gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
            self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, x):
        x_compress = x
        x_out = self.spatial(x_compress)
        attention = torch.sigmoid(x_out) # broadcasting
        x = x * attention
        if 'gb' in self.mask_mode:
            soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
            soft_attention = min_max_norm(soft_attention)       # normalization
            x = torch.mul(x, soft_attention.max(attention))     # x * max(soft, hard)
        return x, attention#x_out#
        
class MaskguidedDualAttention(nn.Module):
    def __init__(self, gate_channels, mask_mode='mask'):
        super(MaskguidedDualAttention, self).__init__()
        self.ChannelGate = ECABlock(gate_channels)
        self.SpatialGate = SpatialGate(gate_channels, mask_mode=mask_mode)
        self.mask_mode = mask_mode
    def forward(self, x):
        x_ca = self.ChannelGate(x)
        x_out, mask = self.SpatialGate(x_ca)
        return x_out + x_ca, mask

## -----------------Global-context Guided Decoder ---------------------
class GGDBlock(nn.Module):
    def __init__(self, channel=32, is_outmost=False):
        super(GGDBlock, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_inup = Conv2dBlock(channel, channel, 3, 1, padding=1, norm='bn', activation='none', use_bias=False)
        self.conv_inbottom = Conv2dBlock(channel, channel, 3, 1, padding=1, norm='bn', activation='none', use_bias=False)
        self.conv_cat = Conv2dBlock(channel*2, channel, 3, 1, padding=1, norm='bn', activation='none', use_bias=False)

        self.outmost = is_outmost
        if self.outmost:
            self.conv4 = Conv2dBlock(channel, channel, 3, 1, padding=1, norm='bn', activation='none', use_bias=False)
            self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x, up,bottom):
        #         x
        #         ↓
        # <-[C]-- * ---- Up
        # <--↑---------- Bottom 
        x_up = self.conv_inup(self.upsample(up)) * x # 28
        x_bottom = self.conv_inbottom(self.upsample(bottom)) # 56

        x_cat = torch.cat((x_up, x_bottom), 1) # 28
        x_out = self.conv_cat(x_cat) # 28

        xup_out = x_out
        xbottom_out = x_bottom
        
        if self.outmost:
            x_out = self.upsample(x_out)
            # x = self.conv4(x_out)
            x = self.conv5(x_out)
            return {'xup':x, 'xbottom':x_out}
        else:
            return {'xup':xup_out, 'xbottom':xbottom_out}

class GGD(nn.Module):
    def __init__(self, channel=32, nstage=4):
        super(GGD, self).__init__()
        self.decoder = nn.ModuleDict()
        self.nstage = nstage - 1
        for i in range(self.nstage):
            if i == 0: self.decoder['d0'] = GGDBlock(channel=channel,  is_outmost=True)
            else:
                self.decoder['d{}'.format(i)] = GGDBlock(channel=channel, is_outmost=False)
    
    def forward(self, xs):
        #x0,x1,x2,x3,x4,x5=xs
        xup = xdown = xs[-1]
        for i, x in enumerate(
            xs[1:-1][::-1]
        ):
            idx = self.nstage - i - 1
            xout = self.decoder['d{}'.format(idx)](x, xup,xdown)
            xup,xdown = xout['xup'], xout['xbottom']
        return xup

## ----------------DIRL --------------------------------

class InharmoniousEncoder(nn.Module):
    def __init__(self, opt, n_channels=3):
        super(InharmoniousEncoder, self).__init__()
        if opt.backbone == 'resnet34':
            resnet = models.resnet34(pretrained=False)
            self.in_dims = [64, 128, 256, 512, 512]
        elif opt.backbone == 'resnet50':
            resnet = models.resnet50(pretrained=False)
            self.in_dims = [64, 256, 512, 1024, 2048]
        ## -------------Encoder--------------
        self.inconv = nn.Conv2d(n_channels,64,3,1,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True) #224,64
        self.maxpool = nn.MaxPool2d(3,2,1)
        #stage 1
        self.encoder1 = resnet.layer1 #112,64*4
        #stage 2
        self.encoder2 = resnet.layer2 #56,128*4
        #stage 3
        self.encoder3 = resnet.layer3 #28,256*4
        #stage 4
        self.encoder4 = resnet.layer4 #14,512*4
        self.encoder5 = nn.Sequential(*[
            BasicBlock(resnet.inplanes, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        ])
        self.inplanes = resnet.inplanes
        
    def forward(self, x, backbone_features=None):
        hx = x
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)
        
        h1 = self.encoder1(hx) # 224
        h2 = self.encoder2(h1) # 112
        h3 = self.encoder3(h2) # 56
        h4 = self.encoder4(h3) # 28
        hx = self.maxpool(h4)
        h5 = self.encoder5(hx) # 14
        return {"skips":[h1,h2,h3,h4,h5]}

class InharmoniousDecoder(nn.Module):
    def __init__(self,opt, n_channels=3):
        super(InharmoniousDecoder,self).__init__()
        ## -------------Dimention--------------
        self.opt = opt
        if opt.backbone == 'resnet34':
            self.dims = [512,512,256,128,64,64]
        elif opt.backbone == 'resnet50':
            self.dims = [2048, 1024, 512, 256, 64,64]
        self.n_layers = len(self.dims)-1
        
        ## ------------Transition Layer------
        self.trans_in_list = self.dims[:-1][::-1]
        self.trans_out_list = [opt.ggd_ch] * 5
        
        self.trans = Transition(
            in_ch_list=self.trans_in_list,
            out_ch_list=self.trans_out_list,
        )
        ## ------------Attention Layer-----------
        self.attention_layers= nn.ModuleDict() 
        for i in range(self.n_layers):
            self.attention_layers['mda_{}'.format(i)] = MaskguidedDualAttention(opt.ggd_ch, mask_mode=self.opt.mda_mode)
        #  ------------ Decoder Layer-----------  
        self.decoder_layers = nn.ModuleDict() 
        self.decoder_layers['deconv'] = GGD(opt.ggd_ch)
        
    def forward(self,z):
        x = z['skips']
        mda_masks = []
        ## -------------Layer Fusion-------
        x = self.trans(x)
        ## -------------Attention ------
        for i in range(self.n_layers-1, -1, -1):
            fused_layer = x[i]
            fused_layer, m = self.attention_layers['mda_{}'.format(i)](fused_layer)
            dst_shape = tuple(x[0].shape[2:])
            m = F.interpolate(m, size=dst_shape, mode='bilinear', align_corners=True)
            mda_masks.append(m)
            x[i] = fused_layer
        ## ------------Decoding --------
        x = self.decoder_layers['deconv'](x).sigmoid()
        if self.opt.mda_mode != 'vanilla':
            return {"mask":[x]+mda_masks}
        else:
            return {"mask":[x]}
                




class DIRLNet(nn.Module):
    def __init__(self, opt, input_nc=3):
        super(DIRLNet, self).__init__()
        self.encoder = InharmoniousEncoder(opt, input_nc)
        self.decoder = InharmoniousDecoder(opt, input_nc) 
        self.opt = opt
        self.inplanes = self.encoder.inplanes
    def forward(self, x):
        z = self.encoder(x)
        out =self.decoder(z)
        # pred = out['mask']
        extra_info = {'lut_z':z['skips'][3]} 
        out.update(extra_info)
        return out

    def load_dict(self, net, load_path, strict=True):
        ckpt_dict = torch.load(load_path, map_location=self.opt.device)
        if 'best_acc' in ckpt_dict.keys():
            new_state_dict = ckpt_dict['state_dict']
            save_epoch = ckpt_dict['epoch']
            self.best_acc  = ckpt_dict['best_acc']
            print("The model from epoch {} reaches acc at {:.4f} !".format(save_epoch, self.best_acc))
        else:
            new_state_dict = ckpt_dict

        current_state_dict = net.state_dict()
        new_keys = tuple(new_state_dict.keys())
        for k in new_keys:
            if k.startswith('module'):
                v = new_state_dict.pop(k)
                nk = k.split('module.')[-1]
                new_state_dict[nk] = v
        if len(self.opt.gpus) > 1:
            net.module.load_state_dict(new_state_dict, strict=strict)
        else:
            net.load_state_dict(new_state_dict, strict=True) # strict

    def load_pretrain_params(self, load_path):
        encoder_path = os.path.join(load_path, 'encoder_epoch60.pth')
        decoder_path = os.path.join(load_path, 'decoder_epoch60.pth')
        self.load_dict(self.encoder, encoder_path, strict=True)
        self.load_dict(self.decoder, decoder_path, strict=True)
        return


