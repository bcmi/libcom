from abc import abstractmethod
from functools import partial
import math
from multiprocessing import context
from typing import Iterable
from einops import rearrange, repeat
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, proj_dir)
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from libcom.controllable_composition.source.ControlCom.ldm.modules.attention import SpatialTransformer
from libcom.controllable_composition.source.ControlCom.ldm.util import get_obj_from_str
from libcom.controllable_composition.source.ControlCom.ldm.modules.local_module import LocalRefineBlock

all_feature_dic={}

def clear_feature_dic():
    global all_feature_dic
    all_feature_dic={}
    all_feature_dic["low"]=[]
    all_feature_dic["mid"]=[]
    all_feature_dic["high"]=[]
    all_feature_dic["highest"]=[]

def get_feature_dic():
    global all_feature_dic
    return all_feature_dic

# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj  = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj    = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, context, local_context, **kwargs):
        attn = None
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, (LocalRefineBlock,)):
                out = layer(x, local_context, **kwargs)
                if isinstance(out, tuple):
                    x,attn = out
                else:
                    x = out
            else:
                x = layer(x)
        
        if attn != None:
            return x,attn
        else:
            return x
        

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class My_ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, 4, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, 4, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        add_conv_in_front_of_unet=False,
        local_encoder_config=None
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.add_conv_in_front_of_unet=add_conv_in_front_of_unet
        self.local_encoder_config = local_encoder_config
        self.add_local_block = (self.local_encoder_config and self.local_encoder_config.conditioning_key != 'none')
        if self.add_local_block:
            LocalBlock = get_obj_from_str(self.local_encoder_config.conditioning_key)
        else:
            LocalBlock = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)


        if self.add_conv_in_front_of_unet:
            self.add_resbolck = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, 9, model_channels, 3, padding=1)
                    )
                ]
            )
            add_layers = [
                        My_ResBlock(
                            model_channels,
                            time_embed_dim,
                            dropout,
                            out_channels=model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]

            self.add_resbolck.append(TimestepEmbedSequential(*add_layers))


        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    add_local_block_cur =  self.add_local_block and \
                        (ds in self.local_encoder_config.resolutions) and \
                            self.local_encoder_config.add_in_encoder
                    if add_local_block_cur and self.local_encoder_config.add_before_crossattn:
                        layers.append(
                            LocalBlock(ch, num_heads, dim_head, 
                                       depth=transformer_depth, 
                                       context_dim=self.local_encoder_config.context_dim,
                                       roi_size=self.local_encoder_config.roi_size)
                                       )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                    if add_local_block_cur and not self.local_encoder_config.add_before_crossattn:
                        layers.append(
                            LocalBlock(ch, num_heads, dim_head, 
                                       depth=transformer_depth, 
                                       context_dim=self.local_encoder_config.context_dim,
                                       roi_size=self.local_encoder_config.roi_size)
                                       )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        
        middle_block = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ]
        add_local_block_cur = self.add_local_block and (ds in self.local_encoder_config.resolutions)
        if add_local_block_cur:
            mid_index = 1 if self.local_encoder_config.add_before_crossattn else -1
            middle_block.insert(mid_index, 
                                LocalBlock(ch, num_heads, dim_head, 
                                            depth=transformer_depth, 
                                            context_dim=self.local_encoder_config.context_dim,
                                            roi_size=self.local_encoder_config.roi_size)
                                            )
                                    
        self.middle_block = TimestepEmbedSequential(*middle_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    add_local_block_cur = self.add_local_block and \
                        (ds in self.local_encoder_config.resolutions) and \
                            (self.local_encoder_config.add_in_decoder)
                    if add_local_block_cur and self.local_encoder_config.add_before_crossattn:
                        layers.append(
                            LocalBlock(ch, num_heads, dim_head, 
                                       depth=transformer_depth, 
                                       context_dim=self.local_encoder_config.context_dim,
                                       roi_size=self.local_encoder_config.roi_size)
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                    if add_local_block_cur and self.local_encoder_config.add_before_crossattn:
                        layers.append(
                            LocalBlock(ch, num_heads, dim_head, 
                                       depth=transformer_depth, 
                                       context_dim=self.local_encoder_config.context_dim,
                                       roi_size=self.local_encoder_config.roi_size)
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def freeze(self):
        local_blocks  = []
        identifier = '.local_proj_in.weight'
        for name,weights in self.named_parameters():
            if identifier in name:
                local_prefix = name.split(identifier)[0]
                local_blocks.append(local_prefix)
        for name,weights in self.named_parameters():
            skip = False
            for block in local_blocks:
                if block in name:
                    
                    weights.requires_grad = True
                    skip = True
                    break
            if not skip:
                weights.requires_grad = False

        # if hasattr(self, 'local_positional_embedding'):
        #     self.local_positional_embedding.requires_grad = True
        # self.out.requires_grad = True

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x_bbox, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x_bbox: list of [x, bbox]
            x: an [N x C x ...] Tensor of inputs.
            bbox: an [N x 4] Tensor of bounding box
        :param timesteps: a 1-D batch of timesteps.
        :param context_tuple: conditioning plugged in via crossattn
            global_c:  N x 768
            local_c: N x 1024 x 16 x 16
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        mask = kwargs['mask'] if 'mask' in kwargs else None
        mask_method = kwargs['mask_method'] if 'mask_method' in kwargs else None
        if mask != None and mask.shape[-2:] != (16, 16):
            mask = F.interpolate(mask, (16, 16), mode='bilinear', align_corners=True)
        x, bbox = x_bbox
        context, local_context, indicator, cond_flag = context
        kwargs['bbox'] = bbox
        kwargs['indicator'] = indicator
        kwargs['cond_flag'] = cond_flag
        ind_map = repeat(indicator, 'b n -> b n h w', h=x.shape[-2], w=x.shape[-1])
        ind_map = ind_map.to(x.dtype)
        x = th.cat([x, ind_map], dim=1)
        
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        attn_list = []
        if self.add_conv_in_front_of_unet:
            for module in self.add_resbolck:
                h = module(h, emb, context, local_context, **kwargs)
        for module in self.input_blocks:
            out = module(h, emb, context, local_context, **kwargs)
            if isinstance(out, tuple):
                h,attn = out
                attn_list.append(attn) 
            else:
                h = out
            hs.append(h)

        h = self.middle_block(h, emb, context, local_context, **kwargs)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            out = module(h, emb, context, local_context, **kwargs)
            if isinstance(out, tuple):
                h,attn = out
                attn_list.append(attn) 
            else:
                h = out
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            model_out = self.out(h)
            if model_out.shape[1] > 4:
                pred_eps, pred_mask = model_out[:,:4], th.sigmoid(model_out[:,-1:])
                return pred_eps, pred_mask
            else:
                return model_out, attn_list

    def get_intermediate_features(self, x_bbox, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x_bbox: list of [x, bbox]
            x: an [N x C x ...] Tensor of inputs.
            bbox: an [N x 4] Tensor of bounding box
        :param timesteps: a 1-D batch of timesteps.
        :param context_tuple: conditioning plugged in via crossattn
            global_c:  N x 768
            local_c: N x 1024 x 16 x 16
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        clear_feature_dic()
        x, bbox = x_bbox
        context, local_context, indicator = context
        # ind_map = repeat(indicator, 'b n -> b n h w', h=x.shape[-2], w=x.shape[-1])
        # ind_map = ind_map.to(x.dtype)
        # x = th.cat([x, ind_map], dim=1)
        if self.local_encoder_config.conditioning_key == 'crossattn' and self.local_encoder_config.add_position_emb:
            # combine foreground local feature with position embedding
            local_context = local_context + self.local_positional_embedding[None,:,:].to(x.dtype)
        else:
            local_context = rearrange(local_context, 'b (h w) c -> b c h w', h=16)
        
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        if self.add_conv_in_front_of_unet:
            for module in self.add_resbolck:
                h = module(h, emb, context, bbox, local_context, indicator)

        if timesteps[0]==1:
            flag_time=True
        else:
            flag_time=False
        
        if flag_time:
            for module in self.input_blocks:
                h = module(h, emb, context, bbox, local_context, indicator)
                hs.append(h)
                reshape_h=h.reshape(int(h.size()[0]/2),int(h.size()[1]*2),h.size()[2],h.size()[3])
                
                if h.size()[2]==8:
                    all_feature_dic["low"].append(reshape_h)
                elif h.size()[2]==16:
                    all_feature_dic["mid"].append(reshape_h)
                elif h.size()[2]==32:
                    all_feature_dic["high"].append(reshape_h)
                elif h.size()[2]==64:
                    all_feature_dic["highest"].append(reshape_h)
        else: 
            for module in self.input_blocks:
                h = module(h, emb, context, bbox, local_context, indicator)
                hs.append(h)
                        
        h = self.middle_block(h, emb, context, bbox, local_context, indicator)
        if flag_time:
            
            reshape_h=h.reshape(int(h.size()[0]/2),int(h.size()[1]*2),h.size()[2],h.size()[3])
            
            if h.size()[2]==8:
                all_feature_dic["low"].append(reshape_h)
            elif h.size()[2]==16:
                all_feature_dic["mid"].append(reshape_h)
            elif h.size()[2]==32:
                all_feature_dic["high"].append(reshape_h)
            elif h.size()[2]==64:
                all_feature_dic["highest"].append(reshape_h)
        if flag_time:
            for module in self.output_blocks:
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context, bbox, local_context, indicator)
            
                reshape_h=h.reshape(int(h.size()[0]/2),int(h.size()[1]*2),h.size()[2],h.size()[3])
                
                if h.size()[2]==8:
                    all_feature_dic["low"].append(reshape_h)
                elif h.size()[2]==16:
                    all_feature_dic["mid"].append(reshape_h)
                elif h.size()[2]==32:
                    all_feature_dic["high"].append(reshape_h)
                elif h.size()[2]==64:
                    all_feature_dic["highest"].append(reshape_h)
                
        else:
            for module in self.output_blocks:
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context, bbox, local_context, indicator)        
            
        
        h = h.type(x.dtype)

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            model_out = self.out(h)
            if model_out.shape[1] > 4:
                pred_eps, pred_mask = model_out[:,:4], th.sigmoid(model_out[:,-1:])
                return pred_eps, pred_mask
            else:
                return model_out


if __name__ == '__main__':
    import torch
    device = torch.device("cuda:0")
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    cfg_path = os.path.join(proj_dir, 'configs/finetune_paint.yaml')
    configs  = OmegaConf.load(cfg_path).model.params
    model  = instantiate_from_config(configs.unet_config).to(device)
    # x_bbox, timesteps=None, context_tuple=None
    x = torch.randn(2, 9, 64, 64).to(device)
    bbox = torch.tensor([[0.1, 0.1, 0.3, 0.4],
                        [0.2, 0.4, 0.5, 0.6]], dtype=torch.float, device=device)
    # timesteps = torch.randint(0, 1000, (x.shape[0],), device=device).long()
    timesteps = torch.ones(x.shape[0]).to(device)
    global_cond = torch.randn(x.shape[0], 1, 768).to(device)
    local_cond  = torch.randn(x.shape[0], 256, 1024).to(device)
    indicator = torch.randint(0, 2, (x.shape[0], 2)).to(device)
    mask = torch.randint(0, 2, (x.shape[0], 1, 16, 16)).float().to(device)
    out,attn_list = model([x,bbox], timesteps, [global_cond, local_cond, indicator], mask=mask, mask_method='sum')
    print(out.shape)
    print([a.shape for a in attn_list])
    # generate foreground mask with diffusion intermediate features
    # out = model.get_intermediate_features([x,bbox], timesteps, [global_cond, local_cond, indicator])
    # mask_generator = instantiate_from_config(configs.mask_generator_config).to(device)
    # pre_mask = mask_generator(all_feature_dic)
    # print(pre_mask.shape)