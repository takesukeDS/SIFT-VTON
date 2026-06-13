import os
from os.path import join as opj
import omegaconf

import cv2
import einops
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

from utils import tensor2img, resize_mask, compute_snr
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler

class ControlLDM(LatentDiffusion):
    def __init__(
            self, 
            control_stage_config, 
            validation_config, 
            control_key, 
            only_mid_control, 
            use_VAEdownsample=False,
            all_unlocked=False,
            config_name="",
            control_scales=None,
            use_pbe_weight=False,
            u_cond_percent=0.0,
            img_H=512,
            img_W=384,
            imageclip_trainable=True,
            pbe_train_mode=False,
            use_attn_mask=False,
            always_learnable_param=False,
            mask1_key="",
            mask2_key="",
            mask3_key="",
            histogram_keys=(None, None, None),
            *args,
            **kwargs
        ):
        self.control_stage_config = control_stage_config
        self.use_pbe_weight = use_pbe_weight
        self.u_cond_percent = u_cond_percent
        self.img_H = img_H
        self.img_W = img_W
        self.config_name = config_name
        self.imageclip_trainable = imageclip_trainable
        self.pbe_train_mode = pbe_train_mode
        self.use_attn_mask = use_attn_mask
        self.mask1_key = mask1_key
        self.mask2_key = mask2_key
        self.mask3_key = mask3_key
        self.histogram_keys = histogram_keys
        self.always_learnable_param = always_learnable_param
        super().__init__(*args, **kwargs)
        control_stage_config.params["use_VAEdownsample"] = use_VAEdownsample
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        if control_scales is None:
            self.control_scales = [1.0] * 13
        else:
            self.control_scales = control_scales
        self.first_stage_key_cond = kwargs.get("first_stage_key_cond", None)
        self.valid_config = validation_config
        self.use_VAEDownsample = use_VAEdownsample
        self.all_unlocked = all_unlocked
        self.gmm = None
        self.clothflow = None
        self.hybvton = kwargs.get("hybvton", False)
        self.siftvton = kwargs.get("siftvton", False)
        self.snr_gamma = kwargs.get("snr_gamma", None)
        if self.siftvton:
            self.model.diffusion_model.set_alphas_cumprod(self.alphas_cumprod)
        self.resume = kwargs.get("resume", None)
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        if isinstance(self.control_key, omegaconf.listconfig.ListConfig):
            control_lst = []
            for key in self.control_key:
                control = batch[key]
                if bs is not None:
                    control = control[:bs]
                control = control.to(self.device)
                control = einops.rearrange(control, 'b h w c -> b c h w')
                control = control.to(memory_format=torch.contiguous_format).float()
                control_lst.append(control)
            control = control_lst
        else:
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()
            control = [control]
        cond_dict = dict(c_crossattn=[c], c_concat=control)
        if self.first_stage_key_cond is not None:
            first_stage_cond = []
            for key in self.first_stage_key_cond:
                if not "mask" in key:
                    cond, _ = super().get_input(batch, key, *args, **kwargs)
                else:
                    cond, _ = super().get_input(batch, key, no_latent=True, *args, **kwargs)      
                first_stage_cond.append(cond)
            first_stage_cond = torch.cat(first_stage_cond, dim=1)
            cond_dict["first_stage_cond"] = first_stage_cond
        return x, cond_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        return_attn_weight = kwargs.pop("return_attn_weight", False)
        
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        if self.proj_out is not None:
            if cond_txt.shape[-1] == 1024:
                cond_txt = self.proj_out(cond_txt)  # [BS x 1 x 768]
        if self.always_learnable_param:
            cond_txt = self.get_unconditional_conditioning(cond_txt.shape[0])
        if "first_stage_cond" in cond:
            x_noisy = torch.cat([x_noisy, cond["first_stage_cond"]], dim=1)
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control, return_attn_weight=return_attn_weight)
        else:
            if not self.use_VAEDownsample:
                hint = cond["c_concat"]
            else:
                hint = []
                for h in cond["c_concat"]:
                    if h.shape[2] == self.img_H and h.shape[3] == self.img_W:
                        h = self.encode_first_stage(h)
                        h = self.get_first_stage_encoding(h).detach()
                    hint.append(h)
            hint = torch.cat(hint, dim=1)
            control, cond_output = self.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt, only_mid_control=self.only_mid_control)
            if len(control) == len(self.control_scales):
                control = [c * scale for c, scale in zip(control, self.control_scales)]
            mask_h = 64
            mask_w = 48
            mask1 = None
            mask2 = None
            mask3 = None
            if self.use_attn_mask:

                mask1 = self.batch[self.mask1_key][:x_noisy.shape[0]].permute(0,3,1,2)
                mask2 = self.batch[self.mask2_key][:x_noisy.shape[0]].permute(0,3,1,2)

                mask1 = self.mask_resize(mask1, mask_h, mask_w, inverse=True)  # [BS x 1 x H x W]
                mask2 = self.mask_resize(mask2, mask_h, mask_w, inverse=True)  # [BS x 1 x H x W]

            if self.mask3_key:
                mask3 = self.batch[self.mask3_key][:x_noisy.shape[0]].permute(0,3,1,2)
                mask3 = F.interpolate(mask3, (mask_h, mask_w), mode="nearest")

            histograms = []
            hist_masks = []
            for hkey in self.histogram_keys:
                if hkey is not None:
                    histograms.append(self.batch[hkey])
                    hist_masks.append(self.batch[hkey + "_mask"])
                else:
                    histograms.append(None)
                    hist_masks.append(None)

            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control,
                                  mask1=mask1, mask2=mask2, mask3=mask3, histograms=histograms, hist_masks=hist_masks,
                                  return_attn_weight=return_attn_weight)
        attn_weight = None
        if isinstance(eps, tuple) and return_attn_weight:
            if len(eps) > 2:
                attn_weight = eps[-1]
                eps = eps[:-1]
            else:
                eps, attn_weight = eps
        return eps, attn_weight

    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, cond_output_dict = self.apply_model(x_noisy, t, cond)

        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        model_loss = None
        if isinstance(model_output, tuple):
            model_output, model_loss = model_output

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        if self.snr_gamma is not None:
            if self.parameterization != "eps":
                raise NotImplementedError("snr weighting is only implemented for eps prediction in this code")
            ### Ported from diffusers script 2025.10.28 Takemoto###
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(t, self.alphas_cumprod)
            mse_loss_weights = (
                    torch.stack([snr, self.snr_gamma * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = loss_simple.mean(dim=list(range(1, len(loss_simple.shape)))) * mse_loss_weights
            loss = loss.mean()
        else:
            logvar_t = self.logvar[t].to(self.device)
            loss = loss_simple / torch.exp(logvar_t) + logvar_t
            if self.learn_logvar:
                loss_dict.update({f'gamma': loss.mean()})
                loss_dict.update({'logvar': self.logvar.data.mean()})
            loss = self.l_simple_weight * loss.mean()

            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            if self.original_elbo_weight != 0:
                loss_dict.update({f'loss_vlb': loss_vlb})
            loss += (self.original_elbo_weight * loss_vlb)

        if model_loss is not None:
            loss += model_loss
            loss_dict.update({"model_loss": model_loss})
        loss_dict.update({f'{prefix}_loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def mask_resize(self, m, h, w, inverse=False):
        m = F.interpolate(m, (h, w), mode="nearest")
        if inverse:
            m = 1-m
        return m
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        if not self.kwargs["use_imageCLIP"]:
            return self.get_learned_conditioning([""] * N)
        else:
            return self.learnable_vector.repeat(N,1,1)
    @torch.no_grad()
    def get_unconditional_conditioning_cnet(self, N):
        return self.learnable_matrix.repeat(N,1,1,1)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        if self.first_stage_key_cond:
            first_stage_cond = c["first_stage_cond"][:N]
            for key_idx, key in enumerate(self.first_stage_key_cond):
                cond = batch[key]
                if len(cond.shape) == 3:
                    cond = cond[..., None]
                cond = rearrange(cond, "b h w c -> b c h w")
                cond = cond.to(memory_format=torch.contiguous_format).float()
                log[f"first_stage_cond_{key_idx}"] = cond
        c_cat = [i[:N] for i in c["c_concat"]]
        c = c["c_crossattn"][0][:N]
        if c.ndim == 4:
            c = self.get_learned_conditioning(c)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        
        x = batch[self.first_stage_key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        log["input"] = x
        log["reconstruction"] = self.decode_first_stage(z)
        log_c_cat = torch.cat(c_cat, dim=1)
        if torch.all(log_c_cat >= 0):
            log["control"] = log_c_cat * 2.0 - 1.0  
        else:
            log["control"] = log_c_cat
        if not self.kwargs["use_imageCLIP"]:
            log["clip conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], 
            size=16)
        else:
            x = batch[self.cond_stage_key]
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b h w c -> b c h w')
            x = x.to(memory_format=torch.contiguous_format).float()
            log["clip conditioning"] = x

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
            samples, z_denoise_row = self.sample_log(cond={"c_concat": c_cat, "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
        if unconditional_guidance_scale >= 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            cond = {"c_concat": c_cat, "c_crossattn": [c]}
            uc_full = {"c_concat": uc_cat, "c_crossattn": [uc_cross]}
            if self.first_stage_key_cond:
                cond["first_stage_cond"] = first_stage_cond
                uc_full["first_stage_cond"] = first_stage_cond
            samples_cfg, _, cond_output_dict = self.sample_log(cond=cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            if cond_output_dict is not None:
                cond_sample = cond_output_dict["cond_sample"]             
                cond_sample = self.decode_first_stage(cond_sample)
                log[f"cond_sample"] = cond_sample
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps=5, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates, cond_output_dict = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates, cond_output_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        print("=====configure optimizer=====")
        if self.hybvton:
            print("hybvton mode")
            params = list(self.model.diffusion_model.output_blocks.parameters())
            print("Unet output blocks is added")
            params += list(self.model.diffusion_model.out.parameters())
            print("Unet out is added")
            params += list(self.model.diffusion_model.input_blocks[0][0].parameters())
            print("Unet input block is added")
            if self.model.diffusion_model.cross_attn_handling == "learnable":
                params += [self.model.diffusion_model.attn_scale_mask3]
                print("attn_scale_mask3 is added")
            params += list(self.control_model.input_blocks[0][0].parameters())
            print("- control model input block is added")
            opt = torch.optim.AdamW(params, lr=lr)
            print("============================")
            return opt
        if self.siftvton:
            print("siftvton mode")
            params = list(self.control_model.parameters())
            print("control model is added")
            params += list(self.model.diffusion_model.output_blocks.parameters())
            print("Unet output blocks is added")
            params += list(self.model.diffusion_model.out.parameters())
            print("Unet out is added")
            params += list(self.model.diffusion_model.input_blocks[0][0].parameters())
            print("Unet input block is added")
            if self.model.diffusion_model.cross_attn_handling == "learnable":
                params += [self.model.diffusion_model.attn_scale_mask3]
                print("attn_scale_mask3 is added")
            print("- control model input block is added")
            if self.model.diffusion_model.learnable_explicit_ratio:
                params += [self.model.diffusion_model.explicit_ratio]
                print("explicit_ratio is added")
            if hasattr(self.model.diffusion_model, "warp_flow_blks"):
                params += list(self.model.diffusion_model.warp_flow_blks.parameters())
                print(f"warp flow blks is added")
            if hasattr(self.model.diffusion_model, "warp_zero_convs"):
                params += list(self.model.diffusion_model.warp_zero_convs.parameters())
                print(f"warp zero convs is added")
            opt = torch.optim.AdamW(params, lr=lr)
            # print total number of trainable parameters
            total_params = sum(p.numel() for p in params if p.requires_grad)
            print(f"Total trainable parameters: {total_params}")
            print("============================")
            if self.resume:
                print(f"Resuming from {self.resume}, no scheduler is used.")
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 1.0)
                return [opt], [scheduler]
            # since finetuning trained VTON model, we use linearly warmup scheduler
            scheduler = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=0.01,
                total_iters=100
            )
            print("Using LinearLR scheduler with warmup: 100 iters from 1% to 100%")
            return [opt], [scheduler]

        if self.pbe_train_mode:
            print("pbe train mode")
            params = list(self.model.parameters())
            print("- unet is added")
            params += list(self.cond_stage_model.final_ln.parameters())
            print("- cond stage model final ln is added")
            params += list(self.cond_stage_model.mapper.parameters())
            print("- cond stage model mapper is added")
            params += list(self.proj_out.parameters())
            print("- proj_out layer is added")
            params.append(self.learnable_vector)
            print("- learnable vector is added")
            opt = torch.optim.AdamW(params, lr=lr)
            print("============================")
            return opt
        params = list(self.control_model.parameters())
        print("control model is added")
        if self.all_unlocked:
            params += list(self.model.parameters())
            print("Unet is added")
        else:
            if not self.sd_locked:
                params += list(self.model.diffusion_model.output_blocks.parameters())
                print("Unet output blocks is added")
                params += list(self.model.diffusion_model.out.parameters())
                print("Unet out is added")
            if "pbe" in self.config_name:
                if self.unet_config.params.in_channels != 9:
                    params += list(self.model.diffusion_model.input_blocks[0].parameters())
                    print("Unet input block is added")
            else:
                if self.unet_config.params.in_channels != 4:
                    params += list(self.model.diffusion_model.input_blocks[0].parameters())
                    print("Unet input block is added")
            if hasattr(self.model.diffusion_model, "warp_flow_blks"):
                params += list(self.model.diffusion_model.warp_flow_blks.parameters())
                print(f"warp flow blks is added")
            if hasattr(self.model.diffusion_model, "warp_zero_convs"):
                params += list(self.model.diffusion_model.warp_zero_convs.parameters())
                print(f"warp zero convs is added")
        if self.cond_stage_trainable:
            if hasattr(self.cond_stage_model, "final_ln"):
                params += list(self.cond_stage_model.final_ln.parameters())
                print("cond stage model final ln is added")
            if hasattr(self.cond_stage_model, "mapper"):
                params += list(self.cond_stage_model.mapper.parameters())
                print("cond stage model mapper is added")
        if self.proj_out is not None:
            params += list(self.proj_out.parameters())
            print("proj out is added")
        if self.learnable_vector is not None:
            params.append(self.learnable_vector)
            print("learnable vector is added")
        opt = torch.optim.AdamW(params, lr=lr)
        print("============================")
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
        
    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.ddim_sampler = DDIMSampler(self)
        self.validation_gene_dirs = []
        for data_type in ["pair", "unpair"]:
            to_dir = opj(self.valid_config.img_save_dir, f"{data_type}_{self.current_epoch}")
            self.validation_gene_dirs.append(to_dir)
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if batch_idx > 4: return
        self.batch = batch
        data_type = "pair" if dataloader_idx==0 else "unpair"
        z, c = self.get_input(batch, self.first_stage_key)
        x_recon = self.decode_first_stage(z)
        shape = (4, self.img_H//8, self.img_W//8)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = self.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = [self.learnable_vector.repeat(z.shape[0],1,1)]

        uc_full = {"c_concat": None, "c_crossattn": uc_cross}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = self.q_sample(c["first_stage_cond"][:, :4], ts)

        samples, intermediates, _ = self.ddim_sampler.sample(
            self.valid_config.ddim_steps,
            bs,
            shape,
            c,
            x_T=start_code,
            verbose=False,
            eta=self.valid_config.eta,
            mask=None,
            x0=None,
            unconditional_guidance_scale=1.5,
            unconditional_conditioning=uc_full
        )
        x_samples = self.decode_first_stage(samples)
        to_dir = opj(self.valid_config.img_save_dir, f"{data_type}_{self.current_epoch}")
        os.makedirs(to_dir, exist_ok=True)
        for x_sample, cloth, gt, fn, recon in zip(x_samples, batch["cloth"], batch["image"], batch["img_fn"], x_recon):
            x_sample_img = tensor2img(x_sample)
            x_recon_img = tensor2img(recon)
            cloth_img = np.uint8((cloth.detach().cpu()+1)/2 * 255.0)
            gt_img = np.uint8((gt.detach().cpu()+1)/2 * 255.0)
            cloth_save = np.concatenate([x_sample_img, gt_img, cloth_img, x_recon_img], axis=1)

            to_path = opj(to_dir, fn)
            cv2.imwrite(to_path, cloth_save[:,:,::-1])

    def on_before_optimizer_step(self, *args, **kwargs):
        super().on_before_optimizer_step(*args, **kwargs)
        if self.siftvton:
            sch = self.lr_schedulers()
            sch.step()
