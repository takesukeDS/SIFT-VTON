from typing import Any, Optional, Dict
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange
from ldm.modules.attention import SpatialTransformer, ExplicitWarpAttention, HybridWarpAttention, \
    CrossAttentionSIFTLoss, CrossAttentionVis, CrossAttentionSIFTLossVis, CrossAttentionSIFTLossAVE
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.util import exists
from ldm.modules.attention import Normalize, CrossAttention, MemoryEfficientCrossAttention, XFORMERS_IS_AVAILBLE, FeedForward
from utils import get_alphas, compute_snr


class CustomBasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,
        "only-explicit-warp": ExplicitWarpAttention,  # only explicit warp attention
        "hybrid": HybridWarpAttention,  # hybrid attention with explicit warp and softmax attention
        "sift-loss": CrossAttentionSIFTLoss,
        "softmax_vis": CrossAttentionVis,  # visualize attention maps
        "sift-loss_vis": CrossAttentionSIFTLossVis,
        "sift-loss_ave": CrossAttentionSIFTLossAVE
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False,use_loss=True, attn_mode=None):
        super().__init__()
        if attn_mode is None:
            if XFORMERS_IS_AVAILBLE:
                attn_mode = "softmax-xformers"
            else:
                attn_mode = "softmax"
        if attn_mode == "softmax-xformers" and not XFORMERS_IS_AVAILBLE:
            print("Xformers is not available, using softmax attention instead.")
            attn_mode = "softmax"
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.use_loss = use_loss

    def forward(
            self, 
            x, 
            context=None, 
            mask=None, 
            mask1=None, 
            mask2=None, 
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
        ):
        if not (use_attention_tv_loss or use_attention_mask):
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
            x = self.attn2(self.norm2(x), context=context, mask=mask, use_loss=self.use_loss) + x
            x = self.ff(self.norm3(x)) + x
            return x
        elif use_attention_mask:
            x1 = self.attn1(
                self.norm1(x), 
                context=context if self.disable_self_attn else None, 
                mask=mask, 
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=False,
            )
            x = x1 + x
            x2 = self.attn2(  # cross attention
                self.norm2(x), 
                context=context,
                mask=mask,
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=False,
            )
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            return x
        else:
            x1, loss1 = self.attn1(
                self.norm1(x), 
                context=context if self.disable_self_attn else None, 
                mask=mask, 
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=use_attention_tv_loss,
                tv_loss_type=tv_loss_type,
            )
            x = x1 + x
            x2, loss2 = self.attn2(
                self.norm2(x), 
                context=context,
                mask=mask,
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=use_attention_tv_loss,
                use_loss=self.use_loss,
                tv_loss_type=tv_loss_type,
            )
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            loss = loss1 + loss2
            return x, loss
        
class CustomSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,use_loss=True, attn_mode=None):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                CustomBasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint, use_loss=use_loss,
                attn_mode=attn_mode
                ) for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.use_loss = use_loss
    def forward(
            self, 
            x, 
            context=None, 
            mask=None, 
            mask1=None, 
            mask2=None, 
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
    ):
        # note: if no context is given, cross-attention defaults to self-attention
        loss = 0
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if not (use_attention_tv_loss or use_attention_mask):
                x = block(x, context=context[i], mask=mask)
            elif use_attention_mask:
                x = block(
                    x,
                    context=context[i],
                    mask=mask, 
                    mask1=mask1, 
                    mask2=mask2, 
                    use_attention_mask=True,
                    use_attention_tv_loss=False,
                    use_center_loss=False,
                )
            else:
                x, attn_loss = block(
                    x,
                    context=context[i],
                    mask=mask, 
                    mask1=mask1, 
                    mask2=mask2, 
                    use_attention_mask=use_attention_mask,
                    use_attention_tv_loss=use_attention_tv_loss,
                    tv_loss_type=tv_loss_type,
                )
                loss += attn_loss
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        if not (use_attention_tv_loss):
            return x + x_in
        else:
            return x + x_in, loss
class StableVITON(UNetModel):
    def __init__(
        self,
        dim_head_denorm=1,
        use_atv_loss=False,
        cross_attn_handling="none",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        warp_flow_blks = []
        warp_zero_convs = []

        self.encode_output_chs = [
            320,
            320,
            640,
            640,
            640,
            1280, 
            1280, 
            1280, 
            1280
        ]

        self.encode_output_chs2 = [
            320,
            320,
            320,
            320,
            640, 
            640, 
            640,
            1280, 
            1280
        ]

        
        for idx, (in_ch, cont_ch) in enumerate(zip(self.encode_output_chs, self.encode_output_chs2)):
            dim_head = in_ch // self.num_heads
            dim_head = dim_head // dim_head_denorm
            warp_flow_blks.append(CustomSpatialTransformer(
                in_channels=in_ch,
                n_heads=self.num_heads,
                d_head=dim_head,
                depth=self.transformer_depth,
                context_dim=cont_ch,
                use_linear=self.use_linear_in_transformer,
                use_checkpoint=self.use_checkpoint,
                use_loss=idx%3 == 1,
            ))
            warp_zero_convs.append(self.make_zero_conv(in_ch))
        self.warp_flow_blks = nn.ModuleList(reversed(warp_flow_blks))
        self.warp_zero_convs = nn.ModuleList(reversed(warp_zero_convs))
        self.use_atv_loss = use_atv_loss
        self.cross_attn_handling = cross_attn_handling
        if self.cross_attn_handling == "learnable":
            self.attn_scale_mask3 = Parameter(0.5 * torch.ones(len(self.warp_flow_blks), dtype=torch.float32))
        elif self.cross_attn_handling == "zero":
            self.register_parameter('attn_scale_mask3',
                                    Parameter(torch.zeros(len(self.warp_flow_blks), dtype=torch.float32)))
        elif self.cross_attn_handling == "none":
            self.register_parameter('attn_scale_mask3', None)
        else:
            raise ValueError(f"Unknown cross_attn_handling: {self.cross_attn_handling}")
    def make_zero_conv(self, channels):
        return zero_module(conv_nd(2, channels, channels, 1, padding=0))

    @torch.no_grad()
    def load_conv_in(self, state_dict: Dict[str, torch.Tensor]):
        conv_in_weight = state_dict["conv_in.weight"]
        conv_in_bias = state_dict.get("conv_in.bias")
        if conv_in_bias is None:
            self.input_blocks[0][0].bias.copy_(conv_in_bias)
        conv_in_weight = torch.cat([conv_in_weight, torch.zeros(
            conv_in_weight.shape[0], self.in_channels - conv_in_weight.shape[1], *conv_in_weight.shape[2:],
            device=conv_in_weight.device, dtype=conv_in_weight.dtype)], dim=1)
        self.input_blocks[0][0].weight.copy_(conv_in_weight)

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        mask1 = kwargs.get("mask1", None)
        mask2 = kwargs.get("mask2", None)
        mask3 = kwargs.get("mask3", None)
        loss = 0

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:                 
            hint = control.pop()
        # resolution 8 is skipped
        for module in self.output_blocks[:3]:
            control.pop()
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        n_warp = len(self.encode_output_chs)
        for i, (module, warp_blk, warp_zc) in enumerate(zip(self.output_blocks[3:n_warp+3], self.warp_flow_blks, self.warp_zero_convs)):
            if control is None or (h.shape[-2] == 8 and h.shape[-1] == 6):
                assert 0, f"shape is wrong : {h.shape}"
            else:
                hint = control.pop()
                h, attn_loss = self.warp(h, hint, warp_blk, warp_zc, i, mask1=mask1, mask2=mask2, mask3=mask3)
                loss += attn_loss
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        for module in self.output_blocks[n_warp+3:]:
            if control is None:
                h = torch.cat([h, hs.pop()], dim=1)                                          
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.use_atv_loss:
            return self.out(h), loss
        else:
            return self.out(h)
    def warp(self, x, hint, crossattn_layer, zero_conv, i, mask1=None, mask2=None, mask3=None):
        hint = rearrange(hint, "b c h w -> b (h w) c").contiguous()
        if self.use_atv_loss:
            output, attn_loss = crossattn_layer(x, hint, mask1=mask1, mask2=mask2, use_attention_tv_loss=True)
            output = zero_conv(output)
            if self.attn_scale_mask3 is not None:
                mask3 = F.interpolate(mask3, size=output.shape[-2:], mode='bilinear')
                output = output * ((1 - mask3) + self.attn_scale_mask3[i] * mask3)
            return output + x, attn_loss
        else:
            output = crossattn_layer(x, hint)
            output = zero_conv(output)
            if self.attn_scale_mask3 is not None:
                mask3 = F.interpolate(mask3, size=output.shape[-2:], mode='bilinear')
                output = output * ((1 - mask3) + self.attn_scale_mask3[i] * mask3)
            return output + x, 0

class NoZeroConvControlNet(nn.Module):
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
            n_embed=None,  
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            use_VAEdownsample=False,
            cond_first_ch=8,
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
            print(f"Constructor of UNetModel received um_attention_blocks={num_attention_blocks}. "
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
        self.use_VAEdownsample = use_VAEdownsample

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

        self.cond_first_block = TimestepEmbedSequential(
            zero_module(conv_nd(dims, cond_first_ch, model_channels, 3, padding=1))
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
        self._feature_size += ch

    # training only conv_input for hybvton
    def prepare_training(self):
        self.requires_grad_(False)
        self.input_blocks[0].requires_grad_(True)

    @torch.no_grad()
    def load_conv_in(self, state_dict: Dict[str, torch.Tensor]):
        conv_in_weight = state_dict["conv_in.weight"]
        conv_in_bias = state_dict.get("conv_in.bias")
        if conv_in_bias is None:
            self.input_blocks[0][0].bias.copy_(conv_in_bias)
        conv_in_weight = torch.cat([conv_in_weight, torch.zeros(
            conv_in_weight.shape[0], self.in_channels - conv_in_weight.shape[1], *conv_in_weight.shape[2:],
            device=conv_in_weight.device, dtype=conv_in_weight.dtype)], dim=1)
        self.input_blocks[0][0].weight.copy_(conv_in_weight)

    def forward(self, x, hint, timesteps, context, only_mid_control=False, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if not self.use_VAEdownsample:
            guided_hint = self.input_hint_block(hint, emb, context)
        else:
            guided_hint = self.cond_first_block(hint, emb, context)

        outs = []
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                hs.append(h)
                guided_hint = None
            else:
                h = module(h, emb, context)
                hs.append(h)
            outs.append(h)

        h = self.middle_block(h, emb, context)
        outs.append(h)
        return outs, None


class ExplicitWarpBasicTransformerBlock(CustomBasicTransformerBlock):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, use_loss=True, attn_mode=None, sift_loss_scale=0.01, tv_loss_scale=0.001):
        self.sift_loss_scale = sift_loss_scale
        self.tv_loss_scale = tv_loss_scale
        super().__init__(dim, n_heads, d_head, dropout, context_dim, gated_ff, checkpoint, disable_self_attn, use_loss,
                         attn_mode)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            mask1=None,
            mask2=None,
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
            mask3=None,
            explicit_sim=None,
            explicit_ratio=1.0,
            use_sift_loss=False,
            hist_mask=None,
        ):
        if self.attn_mode.endswith("_vis"):
            attn_weight_list = []
            if not (use_attention_tv_loss or use_attention_mask or use_sift_loss):
                x1, attn_weight = self.attn1(self.norm1(x),
                               context=context if self.disable_self_attn else None,
                               mask=mask, mask3=mask3, explicit_sim=explicit_sim, explicit_ratio=explicit_ratio)
                x = x1 + x
                attn_weight_list.append(attn_weight)
                x2, attn_weight = self.attn2(self.norm2(x),
                               context=context, mask=mask,
                               mask3=mask3, explicit_sim=explicit_sim, explicit_ratio=explicit_ratio)
                x = x2 + x
                attn_weight_list.append(attn_weight)
                x = self.ff(self.norm3(x)) + x
                return x, attn_weight_list
            elif use_attention_mask:
                x1, attn_weight = self.attn1(
                    self.norm1(x),
                    context=context if self.disable_self_attn else None,
                    mask=mask,
                    mask1=mask1,
                    mask2=mask2,
                    use_attention_tv_loss=False,
                    mask3=mask3,
                    explicit_sim=explicit_sim,
                    explicit_ratio=explicit_ratio,
                )
                attn_weight_list.append(attn_weight)
                x = x1 + x
                x2, attn_weight = self.attn2(  # cross attention
                    self.norm2(x),
                    context=context,
                    mask=mask,
                    mask1=mask1,
                    mask2=mask2,
                    use_attention_tv_loss=False,
                    mask3=mask3,
                    explicit_sim=explicit_sim,
                    explicit_ratio=explicit_ratio,
                )
                attn_weight_list.append(attn_weight)
                x = x2 + x
                x = self.ff(self.norm3(x)) + x
                return x, attn_weight_list
            else:
                x1, loss1, attn_weight = self.attn1(
                    self.norm1(x),
                    context=context if self.disable_self_attn else None,
                    mask=mask,
                    mask1=mask1,
                    mask2=mask2,
                    use_attention_tv_loss=use_attention_tv_loss,
                    tv_loss_type=tv_loss_type,
                    mask3=mask3,
                    explicit_sim=explicit_sim,
                    explicit_ratio=explicit_ratio,
                    use_sift_loss=use_sift_loss,
                    hist_mask=hist_mask,
                    sift_loss_scale=self.sift_loss_scale,
                )
                attn_weight_list.append(attn_weight)
                x = x1 + x
                x2, loss2, attn_weight = self.attn2(
                    self.norm2(x),
                    context=context,
                    mask=mask,
                    mask1=mask1,
                    mask2=mask2,
                    use_attention_tv_loss=use_attention_tv_loss,
                    use_loss=self.use_loss,
                    tv_loss_type=tv_loss_type,
                    mask3=mask3,
                    explicit_sim=explicit_sim,
                    explicit_ratio=explicit_ratio,
                    use_sift_loss=use_sift_loss,
                    hist_mask=hist_mask,
                    sift_loss_scale=self.sift_loss_scale,
                )
                attn_weight_list.append(attn_weight)
                x = x2 + x
                x = self.ff(self.norm3(x)) + x
                loss = loss1 + loss2
                return x, loss, attn_weight_list

        if not (use_attention_tv_loss or use_attention_mask or use_sift_loss):
            x = self.attn1(self.norm1(x),
                           context=context if self.disable_self_attn else None,
                           mask=mask, mask3=mask3, explicit_sim=explicit_sim, explicit_ratio=explicit_ratio) + x
            x = self.attn2(self.norm2(x),
                           context=context, mask=mask,
                           mask3=mask3, explicit_sim=explicit_sim, explicit_ratio=explicit_ratio) + x
            x = self.ff(self.norm3(x)) + x
            return x
        elif use_attention_mask:
            x1 = self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                mask=mask,
                mask1=mask1,
                mask2=mask2,
                use_attention_tv_loss=False,
                mask3=mask3,
                explicit_sim=explicit_sim,
                explicit_ratio=explicit_ratio,
            )
            x = x1 + x
            x2 = self.attn2(  # cross attention
                self.norm2(x),
                context=context,
                mask=mask,
                mask1=mask1,
                mask2=mask2,
                use_attention_tv_loss=False,
                mask3=mask3,
                explicit_sim=explicit_sim,
                explicit_ratio=explicit_ratio,
            )
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            return x
        else:
            x1, loss1 = self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                mask=mask,
                mask1=mask1,
                mask2=mask2,
                use_attention_tv_loss=use_attention_tv_loss,
                tv_loss_type=tv_loss_type,
                mask3=mask3,
                explicit_sim=explicit_sim,
                explicit_ratio=explicit_ratio,
                use_sift_loss=use_sift_loss,
                hist_mask=hist_mask,
                sift_loss_scale=self.sift_loss_scale,
                tv_loss_scale=self.tv_loss_scale
            )
            x = x1 + x
            x2, loss2 = self.attn2(
                self.norm2(x),
                context=context,
                mask=mask,
                mask1=mask1,
                mask2=mask2,
                use_attention_tv_loss=use_attention_tv_loss,
                use_loss=self.use_loss,
                tv_loss_type=tv_loss_type,
                mask3=mask3,
                explicit_sim=explicit_sim,
                explicit_ratio=explicit_ratio,
                use_sift_loss=use_sift_loss,
                hist_mask=hist_mask,
                sift_loss_scale=self.sift_loss_scale,
                tv_loss_scale=self.tv_loss_scale
            )
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            loss = loss1 + loss2
            return x, loss

class ExplicitWarpSpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,use_loss=True, attn_mode=None, sift_loss_scale=0.01, tv_loss_scale=0.001):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                ExplicitWarpBasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint, use_loss=use_loss,
                    attn_mode=attn_mode,
                    sift_loss_scale=sift_loss_scale,
                    tv_loss_scale=tv_loss_scale
                ) for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.use_loss = use_loss
        self.depth = depth

    def forward(
            self,
            x,
            context=None,
            mask=None,
            mask1=None,
            mask2=None,
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
            mask3=None,
            explicit_sim=None,
            explicit_ratio=1.0,
            use_sift_loss=False,
            hist_mask=None,
            return_attn_weight=False
    ):
        # note: if no context is given, cross-attention defaults to self-attention
        loss = 0
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        attn_weights = []
        for i, block in enumerate(self.transformer_blocks):
            if not (use_attention_tv_loss or use_attention_mask or use_sift_loss):
                x = block(x, context=context[i], mask=mask, mask3=mask3,
                          explicit_sim=explicit_sim, explicit_ratio=explicit_ratio)
                if return_attn_weight:
                    x, attn_weight = x
            elif use_attention_mask:
                x = block(
                    x,
                    context=context[i],
                    mask=mask,
                    mask1=mask1,
                    mask2=mask2,
                    use_attention_mask=True,
                    use_attention_tv_loss=False,
                    use_center_loss=False,
                    mask3=mask3,
                    explicit_sim=explicit_sim,
                    explicit_ratio=explicit_ratio,
                )
                if return_attn_weight:
                    x, attn_weight = x
            else:
                x = block(
                    x,
                    context=context[i],
                    mask=mask,
                    mask1=mask1,
                    mask2=mask2,
                    use_attention_mask=use_attention_mask,
                    use_attention_tv_loss=use_attention_tv_loss,
                    use_sift_loss=use_sift_loss,
                    tv_loss_type=tv_loss_type,
                    mask3=mask3,
                    explicit_sim=explicit_sim,
                    explicit_ratio=explicit_ratio,
                    hist_mask=hist_mask
                )
                attn_loss = x[1]
                if return_attn_weight:
                    attn_weight = x[-1]
                x = x[0]
                loss += attn_loss
            if return_attn_weight:
                attn_weights.append(attn_weight)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        if return_attn_weight:
            if not (use_attention_tv_loss or use_sift_loss):
                return x + x_in, attn_weights if self.use_loss else None
            else:
                return x + x_in, loss, attn_weights if self.use_loss else None
        if not (use_attention_tv_loss or use_sift_loss):
            return x + x_in
        else:
            return x + x_in, loss


class ExplicitWarpBasicTransformerBlockSNR(CustomBasicTransformerBlock):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, use_loss=True, attn_mode=None, sift_loss_scale=0.01, tv_loss_scale=0.001):
        self.sift_loss_scale = sift_loss_scale
        self.tv_loss_scale = tv_loss_scale
        super().__init__(dim, n_heads, d_head, dropout, context_dim, gated_ff, checkpoint, disable_self_attn, use_loss,
                         attn_mode)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            mask1=None,
            mask2=None,
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
            mask3=None,
            explicit_sim=None,
            explicit_ratio=1.0,
            use_sift_loss=False,
            hist_mask=None,
        ):
        tv_loss_res = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        sift_loss_res = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        x1, tv_loss, sift_loss = self.attn1(
            self.norm1(x),
            context=context if self.disable_self_attn else None,
            mask=mask,
            mask1=mask1,
            mask2=mask2,
            use_attention_tv_loss=use_attention_tv_loss,
            tv_loss_type=tv_loss_type,
            mask3=mask3,
            explicit_sim=explicit_sim,
            explicit_ratio=explicit_ratio,
            use_sift_loss=use_sift_loss,
            hist_mask=hist_mask,
            sift_loss_scale=self.sift_loss_scale,
            tv_loss_scale=self.tv_loss_scale
        )
        x = x1 + x
        tv_loss_res += tv_loss
        sift_loss_res += sift_loss
        x2, tv_loss, sift_loss = self.attn2(
            self.norm2(x),
            context=context,
            mask=mask,
            mask1=mask1,
            mask2=mask2,
            use_attention_tv_loss=use_attention_tv_loss,
            use_loss=self.use_loss,
            tv_loss_type=tv_loss_type,
            mask3=mask3,
            explicit_sim=explicit_sim,
            explicit_ratio=explicit_ratio,
            use_sift_loss=use_sift_loss,
            hist_mask=hist_mask,
            sift_loss_scale=self.sift_loss_scale,
            tv_loss_scale=self.tv_loss_scale,
        )
        x = x2 + x
        tv_loss_res += tv_loss
        sift_loss_res += sift_loss
        x = self.ff(self.norm3(x)) + x
        return x, tv_loss_res, sift_loss_res

class ExplicitWarpSpatialTransformerSNR(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,use_loss=True, attn_mode=None, sift_loss_scale=0.01, tv_loss_scale=0.001):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                ExplicitWarpBasicTransformerBlockSNR(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    use_loss=use_loss,
                    attn_mode=attn_mode,
                    sift_loss_scale=sift_loss_scale,
                    tv_loss_scale=tv_loss_scale
                ) for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.use_loss = use_loss
        self.depth = depth

    def forward(
            self,
            x,
            context=None,
            mask=None,
            mask1=None,
            mask2=None,
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
            mask3=None,
            explicit_sim=None,
            explicit_ratio=1.0,
            use_sift_loss=False,
            hist_mask=None,
            return_attn_weight=False
    ):
        # note: if no context is given, cross-attention defaults to self-attention
        tv_loss_res = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        sift_loss_res = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        attn_weights = []
        for i, block in enumerate(self.transformer_blocks):
            x, tv_loss, sift_loss = block(
                x,
                context=context[i],
                mask=mask,
                mask1=mask1,
                mask2=mask2,
                use_attention_mask=use_attention_mask,
                use_attention_tv_loss=use_attention_tv_loss,
                use_sift_loss=use_sift_loss,
                tv_loss_type=tv_loss_type,
                mask3=mask3,
                explicit_sim=explicit_sim,
                explicit_ratio=explicit_ratio,
                hist_mask=hist_mask
            )
            tv_loss_res += tv_loss
            sift_loss_res += sift_loss
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in, tv_loss_res, sift_loss_res


class YahaVTON(UNetModel):
    def __init__(
            self,
            dim_head_denorm=1,
            use_atv_loss=False,
            cross_attn_handling="none",
            attn_mode=None,
            explicit_ratio=1.0,
            learnable_explicit_ratio=False,
            use_sift_loss=False,
            sift_loss_scale=0.01,
            sift_loss_step_thres=1000,
            tv_loss_scale=0.001,
            model_loss_all_layers=False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attn_mode = attn_mode
        self.sift_loss_scale = sift_loss_scale
        self.sift_loss_step_thres = sift_loss_step_thres
        warp_flow_blks = []
        warp_zero_convs = []

        self.encode_output_chs = [
            320,
            320,
            640,
            640,
            640,
            1280,
            1280,
            1280,
            1280
        ]

        self.encode_output_chs2 = [
            320,
            320,
            320,
            320,
            640,
            640,
            640,
            1280,
            1280
        ]
        warp_class = ExplicitWarpSpatialTransformerSNR if self.attn_mode in ["sift-loss_ave"] else ExplicitWarpSpatialTransformer

        for idx, (in_ch, cont_ch) in enumerate(zip(self.encode_output_chs, self.encode_output_chs2)):
            dim_head = in_ch // self.num_heads
            dim_head = dim_head // dim_head_denorm
            warp_flow_blks.append(warp_class(
                in_channels=in_ch,
                n_heads=self.num_heads,
                d_head=dim_head,
                depth=self.transformer_depth,
                context_dim=cont_ch,
                use_linear=self.use_linear_in_transformer,
                use_checkpoint=self.use_checkpoint,
                use_loss=True if model_loss_all_layers else idx % 3 == 1,
                attn_mode=self.attn_mode,
                sift_loss_scale=sift_loss_scale,
                tv_loss_scale=tv_loss_scale
            ))
            warp_zero_convs.append(self.make_zero_conv(in_ch))
        self.warp_flow_blks = nn.ModuleList(reversed(warp_flow_blks))
        self.warp_zero_convs = nn.ModuleList(reversed(warp_zero_convs))
        self.use_atv_loss = use_atv_loss
        self.use_sift_loss = use_sift_loss
        self.model_loss_all_layers = model_loss_all_layers
        self.cross_attn_handling = cross_attn_handling
        if self.cross_attn_handling == "learnable":
            self.attn_scale_mask3 = Parameter(0.5 * torch.ones(len(self.warp_flow_blks), dtype=torch.float32))
        elif self.cross_attn_handling == "zero":
            self.register_parameter('attn_scale_mask3',
                                    Parameter(torch.zeros(len(self.warp_flow_blks), dtype=torch.float32)))
        elif self.cross_attn_handling == "none":
            self.register_parameter('attn_scale_mask3', None)
        elif self.cross_attn_handling == "average":
            self.register_buffer('attn_scale_mask3',
                                    0.5 * torch.ones(len(self.warp_flow_blks), dtype=torch.float32))
        else:
            raise ValueError(f"Unknown cross_attn_handling: {self.cross_attn_handling}")
        self.learnable_explicit_ratio = learnable_explicit_ratio
        if self.learnable_explicit_ratio:
            self.explicit_ratio = Parameter(explicit_ratio * torch.ones(len(self.warp_flow_blks), dtype=torch.float32))
        else:
            self.register_buffer(
            'explicit_ratio',
            explicit_ratio * torch.ones(len(self.warp_flow_blks), dtype=torch.float32)
        )
        self.model_loss_snr = self.attn_mode in ["sift-loss_ave"]
        self.alphas_cumprod = None

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(2, channels, channels, 1, padding=0))

    def get_sift_loss_coef(self, timesteps):
        if self.sift_loss_step_thres is None:
            return 1.0
        if self.sift_loss_step_thres < self.alphas_cumprod.shape[0]:
            return (timesteps < self.sift_loss_step_thres).float()
        return (timesteps / (self.alphas_cumprod.shape[0] - 1)).float()

    def get_atv_loss_coef(self, timesteps):
        if self.sift_loss_step_thres is None:
            return 1.0
        if self.sift_loss_step_thres < self.alphas_cumprod.shape[0]:
            return 1.0
        return 1 - (timesteps / (self.alphas_cumprod.shape[0] - 1)).float()

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        mask1 = kwargs.get("mask1", None)
        mask2 = kwargs.get("mask2", None)
        mask3 = kwargs.get("mask3", None)
        histograms = kwargs.get("histograms", (None, None, None))
        hist_masks = kwargs.get("hist_masks", (None, None, None))
        return_attn_weight = kwargs.get("return_attn_weight", False)
        hist_resolutions = [(16, 12), (32, 24), (64, 48)]
        if histograms[0] is None:
            explicit_sims = [None, None, None]
        else:
            explicit_sims = [hist.view(-1, hres[0] * hres[1], hres[0] * hres[1])
                             for hist, hres in zip(histograms, hist_resolutions)]
        loss = 0
        atv_loss_coef = self.get_atv_loss_coef(timesteps)
        sift_loss_coef = self.get_sift_loss_coef(timesteps)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            hint = control.pop()
        # resolution 8 is skipped
        for module in self.output_blocks[:3]:
            if control is not None:
                control.pop()
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        n_warp = len(self.encode_output_chs)
        if mask3 is not None:
            mask3_resized = [F.interpolate(mask3, shape, mode="nearest") for shape in [(16, 12), (32, 24), (64, 48)]]
        else:
            mask3_resized = [None] * 3
        attn_weight_list = []
        for i, (module, warp_blk, warp_zc) in enumerate(
                zip(self.output_blocks[3:n_warp + 3], self.warp_flow_blks, self.warp_zero_convs)):
            if h.shape[-2] == 8 and h.shape[-1] == 6:
                assert 0, f"shape is wrong : {h.shape}"
            if control is not None:
                hint = control.pop()
                warp_out = self.warp(h, hint, warp_blk, warp_zc, i,
                                     mask1=mask1, mask2=mask2, mask3=mask3_resized[i // 3],
                                     explicit_sim=explicit_sims[i // 3], hist_mask=hist_masks[i // 3],
                                     return_attn_weight=return_attn_weight,
                                     atv_loss_coef=atv_loss_coef, sift_loss_coef=sift_loss_coef)
                if return_attn_weight:
                    h, attn_loss, attn_weight = warp_out
                    attn_weight_list.append(attn_weight)
                else:
                    h, attn_loss = warp_out
                loss += attn_loss
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        for module in self.output_blocks[n_warp + 3:]:
            if control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if return_attn_weight:
            if self.use_atv_loss or self.use_sift_loss:
                return self.out(h), loss, attn_weight_list
            else:
                return self.out(h), attn_weight_list
        if self.use_atv_loss or self.use_sift_loss:
            return self.out(h), loss
        else:
            return self.out(h)

    def warp(self, x, hint, crossattn_layer, zero_conv, i,
             mask1=None, mask2=None, mask3=None, explicit_sim=None, hist_mask=None,
             return_attn_weight=False, sift_loss_coef=None, atv_loss_coef=None):
        hint = rearrange(hint, "b c h w -> b (h w) c").contiguous()
        if not return_attn_weight and sift_loss_coef is not None and atv_loss_coef is not None:
            crossattn_layer_output = crossattn_layer(x, hint,
                                                     mask1=mask1, mask2=mask2, mask3=mask3,
                                                     use_attention_tv_loss=self.use_atv_loss,
                                                     use_sift_loss=self.use_sift_loss, hist_mask=hist_mask,
                                                     explicit_sim=explicit_sim, explicit_ratio=self.explicit_ratio[i],
                                                     return_attn_weight=return_attn_weight)
            output, tv_loss, sift_loss = crossattn_layer_output
            attn_loss = atv_loss_coef * tv_loss + sift_loss_coef * sift_loss

            output = zero_conv(output)
            if self.attn_scale_mask3 is not None:
                mask3 = F.interpolate(mask3, size=output.shape[-2:], mode='bilinear')
                output = output * ((1 - mask3) + self.attn_scale_mask3[i] * mask3)
            warp_output = output + x, attn_loss.sum()
            return warp_output
        if self.use_atv_loss or self.use_sift_loss:
            crossattn_layer_output = crossattn_layer(x, hint,
                                                mask1=mask1, mask2=mask2, mask3=mask3,
                                                use_attention_tv_loss=self.use_atv_loss,
                                                use_sift_loss=self.use_sift_loss, hist_mask=hist_mask,
                                                explicit_sim=explicit_sim, explicit_ratio=self.explicit_ratio[i],
                                                return_attn_weight=return_attn_weight)
            if return_attn_weight:
                output, attn_loss, attn_weight = crossattn_layer_output
            else:
                output, attn_loss = crossattn_layer_output
            output = zero_conv(output)
            if self.attn_scale_mask3 is not None:
                mask3 = F.interpolate(mask3, size=output.shape[-2:], mode='bilinear')
                output = output * ((1 - mask3) + self.attn_scale_mask3[i] * mask3)
            if return_attn_weight:
                warp_output = output + x, attn_loss, attn_weight
            else:
                warp_output = output + x, attn_loss
            return warp_output
        else:
            crossattn_layer_output = crossattn_layer(x, hint, mask3=mask3, explicit_sim=explicit_sim,
                                                     explicit_ratio=self.explicit_ratio[i],
                                                     return_attn_weight=return_attn_weight)
            if return_attn_weight:
                output, attn_weight = crossattn_layer_output
            else:
                output = crossattn_layer_output
            output = zero_conv(output)
            if self.attn_scale_mask3 is not None:
                mask3 = F.interpolate(mask3, size=output.shape[-2:], mode='bilinear')
                output = output * ((1 - mask3) + self.attn_scale_mask3[i] * mask3)
            if return_attn_weight:
                warp_output = output + x, 0, attn_weight
            else:
                warp_output = output + x, 0
            return warp_output

    def set_alphas_cumprod(self, alphas_cumprod):
        self.alphas_cumprod = alphas_cumprod
