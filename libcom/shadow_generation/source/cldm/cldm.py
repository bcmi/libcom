import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import cv2

from ..ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ..ldm.modules.attention import SpatialTransformer
from ..ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ..ldm.models.diffusion.ddpm import LatentDiffusion
from ..ldm.util import log_txt_as_img, exists, instantiate_from_config
from ..ldm.models.diffusion.ddim import DDIMSampler
import numpy as np
from .base_network import MaskCls, RegNetwork
import torchvision.models as models
    
class ControlledUnetModel(UNetModel):

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, get_stable_features=False, remask=None, **kwargs):
        if remask is not None:
            return self.out(remask)
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
        encoder_features = hs.copy()
        unet_features = h.clone()
        decoder_features = []
        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)
            decoder_features.append(h)
        h = h.type(x.dtype)
        if get_stable_features:
            return self.out(h), [unet_features, encoder_features, decoder_features]
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
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

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
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
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.add_color_channel = 0

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels, self.add_color_channel)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, 5, 16, 3, padding=1), #TODO
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
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
                        # num_heads = 1
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
                self.zero_convs.append(self.make_zero_conv(ch, self.add_color_channel))
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
                self.zero_convs.append(self.make_zero_conv(ch, self.add_color_channel))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
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
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels, color_hist_channels=0):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels + color_hist_channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs): # TODO
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, cls_net_path, reg_net_path, cls_label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.cls_net_path = cls_net_path
        self.reg_net_path = reg_net_path
        self.cls_label = cls_label
        self.control_scales = [1.0] * 13
        self.mask_cls = MaskCls(num_classes=256)
        
        if os.path.exists(self.cls_net_path):
            self.mask_cls.load_state_dict(torch.load(self.cls_net_path)['net'])
        for param in self.mask_cls.parameters():
            param.requires_grad = False
        self.bbx_reg = RegNetwork()
        if os.path.exists(self.reg_net_path):
            self.bbx_reg.load_state_dict(torch.load(self.reg_net_path)['net'])
        for param in self.bbx_reg.parameters():
            param.requires_grad = False
        self.mask_threshold = 0.5

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def gen_bbx(self, pred_t, fg_instance_bbx, mask):
        img_height, img_width = 512, 512
        pred_bbx = pred_t.new(pred_t.shape)
        pred_bbx[:,0] = pred_t[:,0] * fg_instance_bbx[:,2] + fg_instance_bbx[:,0]
        pred_bbx[:,1] = pred_t[:,1] * fg_instance_bbx[:,3] + fg_instance_bbx[:,1]
        pred_bbx[:,2] = fg_instance_bbx[:,2] * torch.exp(pred_t[:,2])
        pred_bbx[:,3] = fg_instance_bbx[:,3] * torch.exp(pred_t[:,3])
        pred_bbx[:,4] = pred_t[:,4] * 180 / np.pi +fg_instance_bbx[:,4]
        mask = mask.cpu().numpy()
        bs = mask.shape[0]
        for i in range(bs):
            if pred_bbx[i,4] > 0:
                temp = pred_bbx[i,2]
                pred_bbx[i,2] = pred_bbx[i,3]
                pred_bbx[i,3] = temp
                pred_bbx[i,4] = pred_bbx[i,4] - 90
            x, y, w, h, theta = pred_bbx[i,0], pred_bbx[i,1], pred_bbx[i,2], pred_bbx[i,3], pred_bbx[i,4]
            box = ((x, y), (w, h), theta)
            box_points = cv2.boxPoints(box)

            box_points = np.clip(box_points, 0, [img_width - 1, img_height - 1])
            box_points_int = np.int32(box_points)
            cv2.fillPoly(mask[i], [box_points_int], 1)
        mask = torch.tensor(mask).unsqueeze(1).to(dtype=torch.float32).to(self.device)
        return mask

    def apply_model(self, x_noisy, t, cond, get_stable_features=True, batch=None, *args, **kwargs): # TODO
        assert isinstance(cond, dict)
        diffusion_model = self.model.ip_adapter
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(sample=x_noisy, timestep=t, encoder_hidden_states=cond_txt)
            return eps
        else:
            input = batch['cls']
            input = input.to(self.device)
            input = einops.rearrange(input, 'b h w c -> b c h w')
            pred_t = self.bbx_reg(input)
            bbx_mask = batch['bbx']
            fg_instance_bbx = batch['fg']
            fg_instance_bbx = fg_instance_bbx.to(self.device)
            bbx_region = self.gen_bbx(pred_t, fg_instance_bbx, bbx_mask)
            hint = torch.cat(cond['c_concat'], 1)
            control = self.control_model(x=x_noisy, hint=torch.cat((hint, bbx_region), 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            mid_control = control.pop()
            down_control = tuple(control)
            
            mask_label = self.mask_cls(input)
            with open(self.cls_label, 'rb') as f:
                centroid_dict = pickle.load(f)
            _, top64_label = torch.topk(mask_label, self.k, largest=True, sorted=False)
            mask_embeddings = batch['embeddings'].to(self.device)
            for i in range(mask_embeddings.shape[0]):
                for j in range(top64_label.shape[1]):
                    label = top64_label[i,j].item()
                    if label in centroid_dict:
                        mask_embeddings[i,j,:] += torch.tensor(centroid_dict[label]).to(self.device)
                    else:
                        print(f"Warning: label {label} not found in centroid_dict")

            eps = diffusion_model(noisy_latents=x_noisy, timesteps=t, encoder_hidden_states=cond_txt, down_block_additional_residuals=down_control, mid_block_additional_residual=mid_control, mask_embeddings=mask_embeddings)
            return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=1, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True, use_x_T=False,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log_control = c_cat * 2.0 - 1.0
        log["control"] = log_control
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        
        if use_x_T == True:
            x_T = super().q_sample(x_start=z, t=torch.tensor([999], device=self.device, dtype=torch.long), noise=torch.randn_like(z))
        else:
            x_T = None

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta,
                                                     batch=batch)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale >= 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full, x_T=x_T,
                                             batch=batch)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            # log['predicted_mask'] = pred_mask

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, x_T=None, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, x_T=x_T, **kwargs) #add x_T
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        params += list(self.control_model.parameters())
        params += list(self.model.adapter_modules.parameters())
        params += list(self.model.image_proj_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
