# from share import *

from torch import nn
from .cldm.cldm import ControlLDM
import torch
from .cldm.model import create_model, load_state_dict
from torch.utils.data import DataLoader
from .cldm.logger import PostProcessLogger
import pytorch_lightning as pl
import os
import cv2
import numpy as np
from PIL import Image
from libcom.shadow_generation.source.ldm.modules.diffusionmodules.openaimodel import (ResBlock, TimestepEmbedSequential, AttentionBlock, 
                                                      Upsample, SpatialTransformer, Downsample)
from libcom.shadow_generation.source.ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from libcom.shadow_generation.source.ldm.util import exists

class Post_Process_Net(nn.Module):
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
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

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
            for nr in range(self.num_res_blocks[level]):
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
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                if resblock_updown:
                    stage_last_block = ResBlock(
                                            ch,
                                            time_embed_dim,
                                            dropout,
                                            out_channels=out_ch,
                                            dims=dims,
                                            use_checkpoint=use_checkpoint,
                                            use_scale_shift_norm=use_scale_shift_norm,
                                            down=True,
                                        )
                else:
                    stage_last_block = Downsample(
                                            ch, conv_resample, dims=dims, out_channels=out_ch
                                        )
                self.input_blocks.append(
                    TimestepEmbedSequential(stage_last_block)
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
        self.middle_block = TimestepEmbedSequential(
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
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks1 = nn.ModuleList([])
        self.output_blocks2 = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers1 = [
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
                layers2 = [
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
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers1.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                        layers2.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers1.append(ResBlock(ch,
                                            time_embed_dim,
                                            dropout,
                                            out_channels=out_ch,
                                            dims=dims,
                                            use_checkpoint=use_checkpoint,
                                            use_scale_shift_norm=use_scale_shift_norm,
                                            up=True,
                                            ) if resblock_updown else
                                    Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    layers2.append(ResBlock(ch,
                                            time_embed_dim,
                                            dropout,
                                            out_channels=out_ch,
                                            dims=dims,
                                            use_checkpoint=use_checkpoint,
                                            use_scale_shift_norm=use_scale_shift_norm,
                                            up=True,
                                            ) if resblock_updown else
                                    Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks1.append(TimestepEmbedSequential(*layers1))
                self.output_blocks2.append(TimestepEmbedSequential(*layers2))
                self._feature_size += ch

        self.out1 = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, 3, 3, padding=1)),
        )
        self.out2 = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, 1, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        h1 = h2 = h
        for module1, module2 in zip(self.output_blocks1, self.output_blocks2):
            pre_h = hs.pop()
            h1 = torch.cat([h1, pre_h], dim=1)
            h1 = module1(h1, emb, context)   
            h2 = torch.cat([h2, pre_h], dim=1)
            h2 = module2(h2, emb, context)
        h1 = h1.type(x.dtype)
        h2 = h2.type(x.dtype)
        return torch.cat([self.out1(h1), self.out2(h2)], dim=1)
    

class PostProcess(pl.LightningModule):
    def __init__(self, model_path, control_net_path, infe_steps=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_net = Post_Process_Net(image_size=512, 
                                                 in_channels=7, 
                                                 out_channels=4, 
                                                 model_channels=96,
                                                 attention_resolutions=[],
                                                 num_res_blocks=2,
                                                 channel_mult=[ 1, 2, 2, 4 ],
                                                 num_head_channels=64,
                                                 use_spatial_transformer=False,
                                                 use_linear_in_transformer=True,
                                                 transformer_depth=1,
                                                 context_dim=320,
                                                 legacy=False,
                                                 use_checkpoint=True)
        self.model = create_model(model_path).cpu()
        control_net_weight = load_state_dict(control_net_path, location='cpu')
        self.model.load_state_dict(control_net_weight, strict=False)
        self.model.sd_lock = True
        self.model.only_mid_control = False
        self.learning_rate = 1e-5
        self.infe_steps = infe_steps

    def training_step(self, batch, batch_idx):
        self.post_process_net.train()
        # b,h,w,c
        comp_img_scaled = batch['hint'][:, :, :, :3]
        gt_mask_scaled = batch["gt_mask"]
        gt_img_scaled = batch['jpg']
        obj_mask = batch['hint'][:, :, :, 3:]
        comp_img, gt_mask, gt_img = restore_img(comp_img_scaled, gt_mask_scaled, gt_img_scaled)
        batch_size = gt_mask.shape[0]

        with torch.no_grad():
            if os.path.exists(os.path.join(self.generated_image_path, batch['name'][0])):
                images = []
                for name in batch['name']:
                    img_path = os.path.join(self.generated_image_path, name)
                    image = Image.open(img_path).convert('RGB').resize((256, 256),Image.NEAREST)
                    image = np.array(image)
                    images.append(torch.tensor(image[None, :, : ,:]))
                image = torch.clamp(torch.concat(images, dim=0).to(comp_img.device) + 10, 0, 255)
                image_scaled = (image / 127.5) - 1
            else:
                images = self.model.log_images(batch, mode='pndm', input=(comp_img_scaled*2-1).permute(0,3,1,2), 
                                               ddim_steps=self.infe_steps, add_noise_strength=1)
                image_scaled = images['samples_cfg_scale_9.00'].permute(0,2,3,1)
                image = torch.clamp(image_scaled, -1., 1.)
                image = (image + 1.0) / 2.0
                image = (image * 255).int() # bchw -> bhwc
                for i in range(batch_size):
                    image_to_save = np.array(image[i].detach().cpu(), dtype=np.uint8)
                    img_path = os.path.join(self.generated_image_path, batch['name'][i])
                    image_to_save = Image.fromarray(image_to_save)
                    image_to_save.save(img_path)
                image.to(comp_img.device)

        input = torch.concat([image_scaled, comp_img_scaled * 2 - 1, obj_mask], dim=-1)
        null_timeteps = torch.zeros(batch_size, device=input.device)
        output = self.post_process_net(input.permute(0,3,1,2), timesteps=null_timeteps)
        output = output.permute(0,2,3,1)
        
        pred_mask = output[:, :, :, 3]
        adjusted_img = output[:, :, :, :3]
        
        loss1 = nn.functional.l1_loss(pred_mask, gt_mask_scaled * 2 - 1) 
        loss2 = nn.functional.l1_loss(adjusted_img, gt_img_scaled, reduction='none')
        # ratio = torch.sum(torch.greater_equal(gt_mask, 0.6), dim=(1,2)) / (gt_mask.shape[1] * gt_mask.shape[2])
        # scale = torch.clamp((1/ratio).type(torch.int32), 4, 199)
        #scale=4
        # mask = gt_mask * scale[:, None, None] + 1
        # mask = gt_mask * 4 + 1
        #mask = mask[:, :, :, None]
        #loss2 *= mask
        
        loss = loss1 + loss2.mean()

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.post_process_net.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    # get pred_mask, adjusted img and new composite img for logging sake
    def get_log(self, batch, batch_idx, log_num=1):
        comp_img_scaled = batch['hint'][:, :, :, :3][:log_num]
        gt_mask_scaled = batch["gt_mask"][:log_num]
        gt_img_scaled = batch['jpg'][:log_num]
        obj_mask = batch['hint'][:, :, :, 3:][:log_num]
        comp_img, gt_mask, gt_img = restore_img(comp_img_scaled, gt_mask_scaled, gt_img_scaled)
        trimed_batch = {}
        for key, dataset in batch.items():
            trimed_batch[key] = dataset[:log_num]
        self.post_process_net.eval()

        with torch.no_grad():
            if os.path.exists(os.path.join(self.generated_image_path, trimed_batch['name'][0])):
                images = []
                for name in trimed_batch['name']:
                    img_path = os.path.join(self.generated_image_path, name)
                    image = Image.open(img_path).convert('RGB').resize((256, 256),Image.NEAREST)
                    image = np.array(image)
                    images.append(torch.tensor(image[None, :, : ,:]))
                image = torch.clamp(torch.concat(images, dim=0).to(comp_img.device) + 10, 0, 255)
                image_scaled = (image / 127.5) - 1
            else:
                images = self.model.log_images(trimed_batch, mode='pndm', input=(comp_img_scaled*2-1).permute(0,3,1,2), 
                                               ddim_steps=self.infe_steps, add_noise_strength=1)
                image_scaled = images['samples_cfg_scale_9.00'].permute(0,2,3,1)
                image = torch.clamp(image_scaled, -1., 1.)
                image = (image + 1.0) / 2.0
                image = (image * 255).int()

        input = torch.concat([image_scaled, comp_img_scaled * 2 - 1, obj_mask], dim=-1)
        null_timeteps = torch.zeros(log_num, device=input.device)
        output = self.post_process_net(input.permute(0,3,1,2), timesteps=null_timeteps)
        output = output.permute(0,2,3,1)

        pred_mask = torch.greater_equal(output[:, :, :, 3], -0.1).int()
        adjusted_img = output[:, :, :, :3]
        adjusted_img = torch.clamp(adjusted_img, -1., 1.)
        adjusted_img = (adjusted_img + 1.0) / 2.0
        adjusted_img = (adjusted_img * 255).int()
        new_composite_img = adjusted_img * pred_mask.unsqueeze(3) + (1-pred_mask.unsqueeze(3)) * comp_img

        self.post_process_net.train()

        log_info = {"gt_img": gt_img, "gt_mask":gt_mask, "original_pred_img":image, "adjusted_imgs":adjusted_img, 
                    "new_comp_imgs": new_composite_img,  "pred_masks":pred_mask*255}
        for key, k in log_info.items():
            log_info[key] = k.detach().cpu()
        
        return log_info

@torch.no_grad()
def restore_img(comp_img, gt_mask, gt_img):
    assert torch.all(torch.greater_equal(comp_img, 0)) and torch.all(torch.less_equal(comp_img, 1)), "wrong scale for comp_img"
    assert torch.all(torch.greater_equal(gt_img, -1) ) and torch.all(torch.less_equal(gt_img, 1)), "wrong scale for gt_img"
    assert torch.all(torch.greater_equal(gt_mask, 0) ) and torch.all(torch.less_equal(gt_mask, 1)), "wrong scale for gt_mask"
    
    comp_img = comp_img * 255
    gt_mask = gt_mask * 255
    gt_img = (gt_img + 1) * 127.5

    return comp_img, gt_mask, gt_img
