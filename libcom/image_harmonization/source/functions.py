import torch
import numpy as np
import torchvision.transforms as T
from einops.layers.torch import Rearrange
import math
import torch.nn as nn
import numbers
import torch.nn.functional as F
from functools import partial
import os
import trilinear


 
 
class Bottleneck(torch.nn.Module):
    expansion = 1 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
 
        width = int(out_channel * (width_per_group / 64.)) * groups
 
        self.conv1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = torch.nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = torch.nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = torch.nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = torch.nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
 
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class ResNet(torch.nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 32
 
        self.conv1 = torch.nn.Conv2d(4, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
 
        self.layer1 = self._make_layer(block, 32, blocks_num[0])
        self.layer2 = self._make_layer(block, 64, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 128, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 256, blocks_num[3], stride=2)
 
        if self.include_top:
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1)) 
            self.fc = torch.nn.Linear(512*block.expansion, num_classes)
 
 
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None 

        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(channel * block.expansion) )
 
        layers = []
        layers.append(block(self.in_channels, channel, downsample=downsample, stride=stride))
        self.in_channels = channel * block.expansion 
 
        for _ in range(1, block_num): 
            layers.append(block(self.in_channels, channel)) 
 
        return torch.nn.Sequential(*layers) 
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
 
        return x
    

def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = resnet50()
        self.block_channels = [32,64,128,256]

    def forward(self, x, backbone_features):
        outputs = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x2 = self.model.layer1(x)
        outputs.append(x2)
        x3 = self.model.layer2(x2)
        outputs.append(x3)
        x4 = self.model.layer3(x3)
        outputs.append(x4)
        x5 = self.model.layer4(x4)
        outputs.append(x5)

        return outputs[::-1]


class UNetDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer,
                 attention_layer=None, attend_from=3, image_fusion=True):
        super(UNetDecoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        self.image_fusion = image_fusion
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        # Last encoder layer doesn't pool, so there're only (depth - 1) deconvs
        for d in range(depth - 1):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            stage_attention_layer = attention_layer if 0 <= attend_from <= d else None
            self.up_blocks.append(UNetUpBlock(
                in_channels, out_channels, out_channels,
                norm_layer=norm_layer, activation=partial(nn.ReLU, inplace=True),
                padding=1,
                attention_layer=stage_attention_layer,
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, input_image, mask):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.up_blocks, encoder_outputs[1:]):
            output = block(output, skip_output, mask)
        output_map = output
        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * input_image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)

        return output, output_map


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, pool, padding):
        super(UNetDownBlock, self).__init__()
        self.convs = UNetDoubleConv(
            in_channels, out_channels,
            norm_layer=norm_layer, activation=activation, padding=padding,
        )
        self.pooling = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        conv_x = self.convs(x)
        return conv_x, self.pooling(conv_x)


class UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels_decoder, in_channels_encoder, out_channels,
        norm_layer, activation, padding,
        attention_layer,
    ):
        super(UNetUpBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(
                in_channels_decoder, out_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=None, activation=activation,
            )
        )
        self.convs = UNetDoubleConv(
            in_channels_encoder + out_channels, out_channels,
            norm_layer=norm_layer, activation=activation, padding=padding,
        )
        if attention_layer is not None:
            self.attention = attention_layer(in_channels_encoder + out_channels, norm_layer, activation)
        else:
            self.attention = None

    def forward(self, x, encoder_out, mask=None):
        upsample_x = self.upconv(x)
        x_cat_encoder = torch.cat([encoder_out, upsample_x], dim=1)
        if self.attention is not None:
            x_cat_encoder = self.attention(x_cat_encoder, mask)
        return self.convs(x_cat_encoder)


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, padding):
        super(UNetDoubleConv, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                norm_layer=norm_layer, activation=activation,
            ),
            ConvBlock(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                norm_layer=norm_layer, activation=activation,
            ),
        )

    def forward(self, x):
        return self.block(x)
    

class Weight_predictor_issam(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, fb=True):
        super(Weight_predictor_issam, self).__init__()

        # self.dp = nn.Dropout(p=0.5)
        self.mid_ch = 256
        self.fb = fb
        self.conv = nn.Conv2d(in_channels, self.mid_ch, 1, 1, padding=0)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self.fb:
            # print(self.mid_ch*2)
            self.fc = nn.Conv2d(self.mid_ch*3, out_channels, 1, 1, padding=0)
        else:
            # print(self.mid_ch)
            self.fc = nn.Conv2d(self.mid_ch, out_channels, 1, 1, padding=0)

                    
    def forward(self, encoder_outputs, mask):
        fea_input = encoder_outputs[0]
        if len(fea_input.shape) == 3:
            L, bz, d = fea_input.shape
            fea_input = fea_input.permute(1,2,0).view(bz, d, int(math.sqrt(L)), int(math.sqrt(L)))
        x = self.conv(fea_input)
        if self.fb:
            down_mask = F.interpolate(mask, size=fea_input.shape[2:], mode='bilinear', align_corners=True)
            '''
            sum_mask = torch.sum(down_mask,dim=[1,2,3]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            sum_mask[sum_mask<10] = 10
            sum_1mask = torch.sum(1-down_mask,dim=[1,2,3]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            sum_1mask[sum_1mask<10] = 10
            fg_feature = self.avg_pooling(x*down_mask) * 1024.0 / sum_mask
            bg_feature = self.avg_pooling(x*(1-down_mask)) * 1024.0 / sum_1mask
            '''
            fg_mask = down_mask.detach()
            bg_mask = (1-fg_mask).detach()
            full_mask = (fg_mask + bg_mask).detach()
            m_fg = fg_mask.sum(axis=[2,3], keepdim=True)
            m_bg = bg_mask.sum(axis=[2,3], keepdim=True)
            m_full = full_mask.sum(axis=[2,3], keepdim=True)
            fg_feature_map = x*fg_mask/(m_fg+1e-6)
            bg_feature_map = x*bg_mask/(m_bg+1e-6)
            full_feature_map = x*full_mask/(m_full+1e-6)
         
            fg_feature = fg_feature_map.sum(axis=[2,3]).unsqueeze(-1).unsqueeze(-1)
            bg_feature = bg_feature_map.sum(axis=[2,3]).unsqueeze(-1).unsqueeze(-1)
            full_feature = full_feature_map.sum(axis=[2,3]).unsqueeze(-1).unsqueeze(-1)
            fgbg_fea = torch.cat((fg_feature, bg_feature, full_feature),1)
            x = self.fc(fgbg_fea)
        else:
            feature = self.avg_pooling(x)
            x = self.fc(feature)
        return x
    

class LUT(nn.Module):
    def __init__(self, in_channels=1024, n_lut=3, backbone='issam', fb=True, clamp=False):
        super(LUT, self).__init__()
        self.n_lut = n_lut
        self.fb = fb
        self.clamp = clamp
        # print(self.fb)
        
        if self.n_lut==6:
            self.LUT0 = Generator3DLUT_identity()
            self.LUT1 = Generator3DLUT_zero()
            self.LUT2 = Generator3DLUT_zero()
            self.LUT3 = Generator3DLUT_zero()
            self.LUT4 = Generator3DLUT_zero()
            self.LUT5 = Generator3DLUT_zero()
        else:
            raise NotImplementedError

        self.backbone=backbone
        self.classifier = Weight_predictor_issam(in_channels, n_lut, self.fb)

    # def initial_luts(self, lut_num=3):
    #     self.lut_list.append(Generator3DLUT_identity())
    #     for ii in range(lut_num-1):
    #         self.lut_list.append(Generator3DLUT_zero())
    #     return self.lut_list

    def forward(self, encoder_outputs, image, mask):
        pred_weights = self.classifier(encoder_outputs, mask)
        if len(pred_weights.shape) == 1:
            pred_weights = pred_weights.unsqueeze(0)
        pred_weights = F.softmax(pred_weights, dim=1)
        combine_A = image.new(image.size())
        if self.n_lut==6:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            gen_A2 = self.LUT2(image)
            gen_A3 = self.LUT3(image)
            gen_A4 = self.LUT4(image)
            gen_A5 = self.LUT5(image)
            for b in range(image.size(0)):
                combine_A[b,:,:,:] = pred_weights[b,0] * gen_A0[b,:,:,:] + pred_weights[b,1] * gen_A1[b,:,:,:] + pred_weights[b,2] * gen_A2[b,:,:,:] \
                    + pred_weights[b,3] * gen_A3[b,:,:,:] + pred_weights[b,4] * gen_A4[b,:,:,:] + pred_weights[b,5] * gen_A5[b,:,:,:]
        else:
            raise NotImplementedError
        if self.clamp:
            combine_A = torch.clamp(combine_A,0,1)

        combine_A = combine_A*mask+image*(1-mask)
        
        return combine_A#,pred_weights.squeeze().detach().cpu().numpy()


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        filepath = '/'.join(os.path.realpath(__file__).split("/")[0:-2])
        if dim == 33:
            file = open(filepath + "/pretrained_models/IdentityLUT33.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split()
                    buffer[0,i,j,k] = float(x[0])
                    buffer[1,i,j,k] = float(x[1])
                    buffer[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output

class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3,dim,dim,dim, dtype=torch.float)
        self.LUT = nn.Parameter(self.LUT.clone().detach().requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):

        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output

class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        
        assert 1 == trilinear.forward(lut, 
                                      x, 
                                      output,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == trilinear.backward(x, 
                                       x_grad, 
                                       lut_grad,
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TrilinearInterpolationGS(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolationGS, self).__init__()

    def forward(self, lut, img):
        # scale im between -1 and 1 since its used as grid input in grid_sample
        img = (img - .5) * 2.
        # grid_sample expects NxD_outxH_outxW_outx3 (1x1xHxWx3)
        img = img.permute(0, 2, 3, 1)[:, None]
        # add batch dim to LUT
        lut = lut[None] # [B,C,D_in,H_in,W_in] -> [B,3,M,M,M] 
        # grid sample
        result = F.grid_sample(lut, img, mode='bilinear', padding_mode='border', align_corners=True) # [B, C, D_out, H_out, W_out ]
        # drop added dimensions and permute back
        result = result[:, :, 0,:,:]
        # print('after result', result.shape)
        return lut,result
    

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=4, stride=2, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.ELU,
        bias=True,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        # self.act = activation()

    def forward(self, x):
        # x = self.conv(x)
        # x = self.norm(x)
        # x = self.act(x)
        return self.block(x)


class GaussianSmoothing(nn.Module):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    Apply gaussian smoothing on a tensor (1d, 2d, 3d).
    Filtering is performed seperately for each channel in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors.
            Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data. Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, padding=0, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1.
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.
            kernel *= torch.exp(-((grid - mean) / std) ** 2 / 2) / (std * (2 * math.pi) ** 0.5)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight.
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = torch.repeat_interleave(kernel, channels, 0)

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = padding

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, padding=self.padding, groups=self.groups)
    

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
        ])
        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(
            nn.Linear(len(self.global_pools) * in_channels, intermediate_channels_count),
            nn.ReLU(),
            nn.Linear(intermediate_channels_count, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled_x = []
        for global_pool in self.global_pools:
            pooled_x.append(global_pool(x))
        pooled_x = torch.cat(pooled_x, dim=1).flatten(start_dim=1)
        channel_attention_weights = self.attention_transform(pooled_x)[..., None, None]
        return channel_attention_weights * x
    

class ViT_Harmonizer(nn.Module):
    def __init__(self, output_nc, ksize=4, tr_r_enc_head=2, tr_r_enc_layers=9, input_nc=3, dim_forward=2, tr_act='gelu'):
        super(ViT_Harmonizer, self).__init__()
        dim = 256
        self.patch_to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = ksize, p2 = ksize),
                nn.Linear(ksize*ksize*(input_nc+1), dim)
            )
        self.transformer_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,nhead=tr_r_enc_head, dim_feedforward=dim*dim_forward, activation=tr_act), num_layers=tr_r_enc_layers)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, output_nc, kernel_size=ksize, stride=ksize, padding=0),
            nn.Tanh()
        )

    def forward(self, inputs, backbone_features=None):
        patch_embedding = self.patch_to_embedding(inputs)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2))
        bs, L, C  = patch_embedding.size()
        harmonized = self.dec(content.permute(1,2,0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L))))
        return harmonized

class PCT():
    '''
    Pixel-Wise Color Transform
    applies specified PCT function to image given a parameter map

    transform_type : str
        PCT function name
    dim : int
        dimension of input vector (usually 3)
    affine : bool
        tranform has a translational component
    color_space : str
        transforms input to 'HSV' or 'YUV' to apply transform, does nothing for RGB 
    mean : list
        input normalization mean value
    std : list
        input normalization standard deviation value
    unnorm : bool
        before applying the transformation the input normalization is reversed
    clamp: bool
        clamp output to [0, 1] (before applying input normalization again)
    '''

    def __init__(self, transform_type, dim, affine, 
                    color_space='RGB', mean = [.485, .456, .406], std = [.229, .224, .225], 
                    unnorm=False, clamp=True):
        
        # Color Space
        self.color_trf_in = lambda x: x
        self.color_trf_out = lambda x: x
        
        # Normalization
        self.norm = T.Normalize(mean=mean, std=std)
        self.unnorm = unnorm
        if color_space in ['HSV', 'YUV'] or unnorm:
            self.unnorm = torch.transforms.Normalize(mean=-mean/std, std=1/std)

        self.clamp = clamp

        # Transform Functions
        if transform_type == 'identity':
            self.transform = lambda input, param: param
            self.out_dim = 3
        elif transform_type == 'mul':
            self.transform = lambda input, param: input * param
            self.out_dim = 3
        elif transform_type == 'add':
            self.transform = lambda input, param: input + param
            self.out_dim = 3
        elif 'linear' in transform_type:
            type = transform_type.split('_')[-1]
            self.transform = Linear_PCT(dim, affine, type)
            self.out_dim = self.transform.out_dim
        elif transform_type == 'polynomial':
            self.transform = Linear_PCT(dim, affine, 'linear', 'polynomial')
            self.out_dim = 27
        elif transform_type == 'quadratic':
            self.transform == Polynomial_PCT(dim, 2)
            self.out_dim = 6
        elif transform_type == 'cubic':
            self.transform == Polynomial_PCT(dim, 3)
            self.out_dim = 9
        else:
            self.out_dim = 0
            print('Error: Invalid transform type')

    def __call__(self, input, param):
        
        if self.unnorm:
            input = self.unnorm(input)
        input = self.color_trf_in(input)

        output = self.transform(input, param)
        
        output = self.color_trf_out(output)
        if self.clamp:
            output = torch.clamp(output, 0, 1)
        output = self.norm(output)
        
        return output

    def get_out_dim(self):
        return self.out_dim


class Linear_PCT():

    def __init__(self, dim, affine, type='linear', projection=None):
        
        self.dim = dim
        self.affine = affine
        self.type = type
        self.projection = projection

        if type=='linear':
            self.out_dim = 9
        elif type=='sym':
            self.out_dim = 6
        if affine:
            self.out_dim += 3

    def __call__(self, input, param):
        
        N, C_in, H, W = input.shape
        out = torch.zeros_like(input)

        L0 = 3
        L = self.dim
        for n in range(N):
            x = input[n].movedim(0, -1).view(-1, C_in).unsqueeze(2)                                         # (HW, C_in, 1)
            
            if self.projection == 'polynomial':
                xr, xb, xg = x[:,:1], x[:,0:1], x[:,2:]
                x = torch.cat([xr, xb, xg, xr*xg, xr*xb, xg*xb, xr**2, xg**2, xb**2], axis=1)
                L0 = 9
            elif self.projection == 'sine':
                x = torch.cat([torch.sin(2**(i//2) * x * np.pi/2 * (i % 2) ) for i in range(9)], axis=1)    # (H*W, 3*L, 1)
                L0 = 9

            # Linear Matrix Multiplication
            if self.type == 'sym':
                L0 = 2
                M = param[n, 0:L0*L].movedim(0, -1).view(-1, L0*L)  
                M = torch.stack( [  torch.stack([M[:,0], M[:,3], M[:,5]], dim=1), 
                                    torch.stack([M[:,3], M[:,1], M[:,4]], dim=1),
                                    torch.stack([M[:,5], M[:,4], M[:,2]], dim=1)], dim=2)
            else:
                M = param[n, 0:L0*L].movedim(0, -1).view(-1, L, L0)                                         # (HW, L, L0)
            
            y = torch.matmul(M, x)    
            if self.affine:
                b = param[n,L0*L:(L0+1)*L].movedim(0, -1).view(-1, L).unsqueeze(2)                          # (HW, L, L0)
                y = y + b
            out[n] = y.view(H, W, C_in).movedim(-1, 0)                                                      # (H, W, 3)

        return out


class Polynomial_PCT():

    def __init__(self, dim, deg):
        self.dim = dim
        self.deg = deg

    def __call__(self, input, param):

        N, C_in, H, W = input.shape
        out = torch.zeros_like(input)

        param = param.view(N, self.C_in, self.dim, H, W)
        for l in range(self.deg+1):
            out += param[:, l] * torch.pow(input, l)
        
        return out