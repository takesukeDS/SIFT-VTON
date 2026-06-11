"""SAMPLING ONLY."""
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import partial

from cldm.cldm import ControlLDM
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from ldm.models.diffusion.sampling_util import norm_thresholding
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from networks import make_grid
from utils import ensure_tensor, remove_overlap, bilateral_filter


class PLMSSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.first_n_repaint = kwargs.get("first_n_repaint", None)
        self.resampling_trick = kwargs.get("resampling_trick", False)
        self.resampling_trick_repeat = kwargs.get("resampling_trick_repeat", 10)
        self.return_attn_weight = kwargs.pop("return_attn_weight", False)
        if self.resampling_trick and self.return_attn_weight:
            raise NotImplementedError("resampling_trick with return_attn_weight is not implemented")
        self.attn_weight_list = []
        if self.resampling_trick:
            print(f"use resampling trick with repeat {self.resampling_trick_repeat}")

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               **kwargs
               ):
        # reset attn_weight_list
        self.attn_weight_list = []
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates, cond_output_dict = self.plms_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    )
        return samples, intermediates, cond_output_dict

    @torch.no_grad()
    def plms_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            if not self.resampling_trick:
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

                if mask is not None:
                    assert x0 is not None
                    if i < self.first_n_repaint:
                        img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                        img = img_orig * mask + (1. - mask) * img

                outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        old_eps=old_eps, t_next=ts_next,
                                        dynamic_threshold=dynamic_threshold)
                img, pred_x0, e_t = outs
                old_eps.append(e_t)
                if len(old_eps) >= 4:
                    old_eps.pop(0)
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)
            else:  # resampling_trick 적용
                if i == 0:
                    for r in range(self.resampling_trick_repeat):
                        index = total_steps - i - 1
                        ts = torch.full((b,), step, device=device, dtype=torch.long)
                        ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

                        if mask is not None:
                            assert x0 is not None
                            if i < self.first_n_repaint:
                                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                                img = img_orig * mask + (1. - mask) * img

                        outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                                quantize_denoised=quantize_denoised, temperature=temperature,
                                                noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=unconditional_conditioning,
                                                old_eps=old_eps, t_next=ts_next,
                                                dynamic_threshold=dynamic_threshold)
                        img, pred_x0, e_t = outs
                        old_eps.append(e_t)
                        if len(old_eps) >= 4:
                            old_eps.pop(0)
                        if callback: callback(i)
                        if img_callback: img_callback(pred_x0, i)

                        if r !=9 :
                            img = self.undo(x_t=img, t=ts-1)
                            print(f"resampling trick : [step - {step}]_[repeat - {r}/{self.resampling_trick_repeat}]")
                else:
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=device, dtype=torch.long)
                    ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

                    if mask is not None:
                        assert x0 is not None
                        if i < self.first_n_repaint:
                            img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                            img = img_orig * mask + (1. - mask) * img

                    outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                            quantize_denoised=quantize_denoised, temperature=temperature,
                                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning,
                                            old_eps=old_eps, t_next=ts_next,
                                            dynamic_threshold=dynamic_threshold)
                    img, pred_x0, e_t = outs
                    old_eps.append(e_t)
                    if len(old_eps) >= 4:
                        old_eps.pop(0)
                    if callback: callback(i)
                    if img_callback: img_callback(pred_x0, i)

                    if index % log_every_t == 0 or index == total_steps - 1:
                        intermediates['x_inter'].append(img)
                        intermediates['pred_x0'].append(pred_x0)

        return img, intermediates, None
    def undo(self, x_t, t):
        beta = extract_into_tensor(self.betas, t, x_t.shape)
        x_t_forward = torch.sqrt(1 - beta) * x_t + torch.sqrt(beta) * torch.randn_like(x_t)
        return x_t_forward
    
    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t, attn_weight = self.model.apply_model(x, t, c, return_attn_weight=self.return_attn_weight)
            else:
                model_t, attn_weight = self.model.apply_model(x,t,c, return_attn_weight=self.return_attn_weight)
                model_uncond, _ = self.model.apply_model(x,t,unconditional_conditioning)

                if isinstance(model_t, tuple):
                    model_t, _ = model_t
                if isinstance(model_uncond, tuple):
                    model_uncond, _ = model_uncond
                e_t = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t, attn_weight

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            if dynamic_threshold is not None:
                pred_x0 = norm_thresholding(pred_x0, dynamic_threshold)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t, attn_weight = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next, _ = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        self.attn_weight_list.append(attn_weight)
        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t


class PLMSSamplerHybvton(PLMSSampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.bilateral_kernel_size = kwargs.get("bilateral_kernel_size", 4)
        self.bilateral_sigma_d = kwargs.get("bilateral_sigma_d", 5.0)
        self.bilateral_sigma_r = kwargs.get("bilateral_sigma_r", 0.06)
        self.bilateral_filter_iterations = kwargs.get("bilateral_filter_iterations", 4)
        self.num_erode_iterations = kwargs.get("num_erode_iterations", 1)
        self.erode_kernel_size = kwargs.get("erode_kernel_size", 21)
        self.tocg = kwargs.get("tocg", None)
        if self.tocg is None:
            print(f"Warning: ToCG is not provided. Sampling without refinement.")
        self.timestep_threshold = kwargs.get("timestep_threshold", 1000)
        self.display_cond = kwargs.get("display_cond", False)
        self.save_dir_cond = kwargs.get("save_dir_cond", "display_cond")
        if self.display_cond:
            os.makedirs(self.save_dir_cond, exist_ok=True)
        self.extract_torso = kwargs.get("extract_torso", False)
        self.use_preprocessed = kwargs.get("use_preprocessed", False)
        self.only_one_refinement = kwargs.get("only_one_refinement", False)

    @torch.no_grad()
    def sample(self,
               batch,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates, cond_output_dict = self.plms_sampling(batch, conditioning, size,
                                                                      callback=callback,
                                                                      img_callback=img_callback,
                                                                      quantize_denoised=quantize_x0,
                                                                      mask=mask, x0=x0,
                                                                      ddim_use_original_steps=False,
                                                                      noise_dropout=noise_dropout,
                                                                      temperature=temperature,
                                                                      score_corrector=score_corrector,
                                                                      corrector_kwargs=corrector_kwargs,
                                                                      x_T=x_T,
                                                                      log_every_t=log_every_t,
                                                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                                                      unconditional_conditioning=unconditional_conditioning,
                                                                      dynamic_threshold=dynamic_threshold,
                                                                      )
        return samples, intermediates, cond_output_dict

    @torch.no_grad()
    def plms_sampling(self, batch, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []
        if self.use_preprocessed and self.bilateral_filter_iterations > 0:
            print(f"Warning: Using preprocessed images with additional bilateral filtering")

        # with only_one_refinement, diffusion steps before timestep threshold are without warped cloth
        if self.only_one_refinement and self.tocg is not None:
            batch["agn"] = batch["agn_orig"]
            batch["agn_mask"] = batch["agn_mask_orig"]
            batch["hybvton_warped_mask"] = torch.zeros_like(batch["hybvton_warped_mask"])

        if self.display_cond:
            for name in ["agn", "agn_mask", "hybvton_warped_mask", "densepose_torso_mask"]:
                if name == "densepose_torso_mask":
                    img_tmp = batch.get(name)
                    if img_tmp is None:
                        continue
                    img_tmp = img_tmp[0].permute(1, 2, 0).repeat(1, 1, 3)
                elif "mask" in name:
                    img_tmp = batch[name][0].repeat(1, 1, 3)
                else:
                    img_tmp = batch[name][0]
                img_tmp = (img_tmp + 1) / 2
                img_tmp = img_tmp.cpu().numpy()
                img_tmp = (img_tmp * 255).astype(np.uint8)
                Image.fromarray(img_tmp).save(
                    f"{self.save_dir_cond}/{batch['img_fn'][0]}_{batch['cloth_fn'][0]}_{name}_before.png")

        do_refinement = True
        for i, step in enumerate(iterator):
            if not self.resampling_trick:
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

                if mask is not None:
                    assert x0 is not None
                    if i < self.first_n_repaint:
                        img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                        img = img_orig * mask + (1. - mask) * img

                if self.tocg is not None and do_refinement and step < self.timestep_threshold:
                    if self.only_one_refinement:
                        do_refinement = False
                    input_timesteps = ensure_tensor(step)
                    if input_timesteps.dim() == 0:
                        input_timesteps = input_timesteps.unsqueeze(0)
                    input_timesteps = input_timesteps.expand(b).cuda()
                    warped_cloth_orig = batch['warped_cloth_orig']
                    warped_mask_orig = batch['warped_mask_orig']
                    parse_agnostic = batch['parse_agnostic']
                    densepose = batch['image_densepose'].permute(0, 3, 1, 2)
                    to_be_warped = batch['warped_cloth_processed'] if self.use_preprocessed else batch['warped_cloth_orig']

                    # down
                    pre_clothes_mask_down = F.interpolate(warped_mask_orig, size=(256, 192), mode='nearest')
                    input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
                    clothes_down = F.interpolate(warped_cloth_orig, size=(256, 192), mode='bilinear')
                    densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

                    noisy_latents = TF.resize(img, (256, 192),
                                              interpolation=TF.InterpolationMode.BICUBIC)
                    # multi-task inputs
                    input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                    input2 = torch.cat([input_parse_agnostic_down, densepose_down, noisy_latents], 1)
                    flow_list, fake_segmap, _, warped_clothmask_paired = self.tocg(input1, input2, input_timesteps)
                    # warped cloth
                    N, _, iH, iW = warped_cloth_orig.shape
                    flow = F.interpolate(
                        flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0,2,3,1)
                    flow_norm = torch.cat(
                        [flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)],
                        3)

                    grid = make_grid(N, iH, iW, flow.device)
                    warped_grid = grid + flow_norm
                    warped_cloth = F.grid_sample(to_be_warped, warped_grid, padding_mode='border')
                    warped_clothmask = F.grid_sample(warped_mask_orig, warped_grid, padding_mode='border')
                    fake_segmap = F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear')
                    warped_clothmask = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask,
                                                      inference=True)
                    if self.extract_torso:
                        warped_clothmask = warped_clothmask * batch["densepose_torso_mask"]
                    warped_cloth = warped_cloth * warped_clothmask + torch.zeros_like(warped_cloth) * (
                                1 - warped_clothmask)

                    if self.num_erode_iterations > 0:
                        for _ in range(self.num_erode_iterations):
                            warped_clothmask = 1 - F.max_pool2d(1 - warped_clothmask, self.erode_kernel_size, stride=1,
                                         padding=self.erode_kernel_size // 2)
                        warped_cloth = warped_cloth * warped_clothmask + \
                                       torch.zeros_like(warped_cloth) * (1 - warped_clothmask)

                    if self.bilateral_filter_iterations > 0:
                        for _ in range(self.bilateral_filter_iterations):
                            warped_cloth = bilateral_filter(warped_cloth, self.bilateral_kernel_size,
                                                            self.bilateral_sigma_d, self.bilateral_sigma_r)

                    warped_cloth = warped_cloth.permute(0, 2, 3, 1)
                    warped_clothmask = warped_clothmask.bool().permute(0, 2, 3, 1) & (batch["agn_mask_orig"] < 0.5)
                    warped_clothmask = warped_clothmask.float()
                    batch["agn_mask"] = batch["agn_mask_orig"] + warped_clothmask
                    batch["agn"] = batch["agn_orig"] * (1 - warped_clothmask) + warped_cloth * warped_clothmask
                    batch["hybvton_warped_mask"] = warped_clothmask

                    if self.display_cond:
                        for name in ["agn", "agn_mask", "hybvton_warped_mask"]:
                            if "mask" in name:
                                img_tmp = batch[name][0].repeat(1, 1, 3)
                            else:
                                img_tmp = batch[name][0]
                            img_tmp = (img_tmp + 1) / 2
                            img_tmp = img_tmp.cpu().numpy()
                            img_tmp = (img_tmp * 255).astype(np.uint8)
                            Image.fromarray(img_tmp).save(
                                f"{self.save_dir_cond}/{batch['img_fn'][0]}_{batch['cloth_fn'][0]}_{name}_{step}.png")

                    first_stage_cond = []
                    for key in self.model.first_stage_key_cond:
                        if not "mask" in key:
                            cond_tmp, _ = super(ControlLDM, self.model).get_input(batch, key)
                        else:
                            cond_tmp, _ = super(ControlLDM, self.model).get_input(batch, key, no_latent=True)
                        first_stage_cond.append(cond_tmp)
                    first_stage_cond = torch.cat(first_stage_cond, dim=1)
                    cond["first_stage_cond"] = first_stage_cond


                outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          old_eps=old_eps, t_next=ts_next,
                                          dynamic_threshold=dynamic_threshold)
                img, pred_x0, e_t = outs
                old_eps.append(e_t)
                if len(old_eps) >= 4:
                    old_eps.pop(0)
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)
            else:  # resampling_trick 적용
                raise NotImplementedError("We do not support resampling trick in this version.")

        return img, intermediates, None


class PLMSSamplerYaha(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.first_n_repaint = kwargs.get("first_n_repaint", None)
        self.resampling_trick = kwargs.get("resampling_trick", False)
        self.resampling_trick_repeat = kwargs.get("resampling_trick_repeat", 10)
        if self.resampling_trick:
            print(f"use resampling trick with repeat {self.resampling_trick_repeat}")

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates, cond_output_dict = self.plms_sampling(conditioning, size,
                                                                      callback=callback,
                                                                      img_callback=img_callback,
                                                                      quantize_denoised=quantize_x0,
                                                                      mask=mask, x0=x0,
                                                                      ddim_use_original_steps=False,
                                                                      noise_dropout=noise_dropout,
                                                                      temperature=temperature,
                                                                      score_corrector=score_corrector,
                                                                      corrector_kwargs=corrector_kwargs,
                                                                      x_T=x_T,
                                                                      log_every_t=log_every_t,
                                                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                                                      unconditional_conditioning=unconditional_conditioning,
                                                                      dynamic_threshold=dynamic_threshold,
                                                                      )
        return samples, intermediates, cond_output_dict

    @torch.no_grad()
    def plms_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            if not self.resampling_trick:
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

                if mask is not None:
                    assert x0 is not None
                    if i < self.first_n_repaint:
                        img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                        img = img_orig * mask + (1. - mask) * img

                outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          old_eps=old_eps, t_next=ts_next,
                                          dynamic_threshold=dynamic_threshold)
                img, pred_x0, e_t = outs
                old_eps.append(e_t)
                if len(old_eps) >= 4:
                    old_eps.pop(0)
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)
            else:  # resampling_trick 적용
                if i == 0:
                    for r in range(self.resampling_trick_repeat):
                        index = total_steps - i - 1
                        ts = torch.full((b,), step, device=device, dtype=torch.long)
                        ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device,
                                             dtype=torch.long)

                        if mask is not None:
                            assert x0 is not None
                            if i < self.first_n_repaint:
                                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                                img = img_orig * mask + (1. - mask) * img

                        outs = self.p_sample_plms(img, cond, ts, index=index,
                                                  use_original_steps=ddim_use_original_steps,
                                                  quantize_denoised=quantize_denoised, temperature=temperature,
                                                  noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                  corrector_kwargs=corrector_kwargs,
                                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                                  unconditional_conditioning=unconditional_conditioning,
                                                  old_eps=old_eps, t_next=ts_next,
                                                  dynamic_threshold=dynamic_threshold)
                        img, pred_x0, e_t = outs
                        old_eps.append(e_t)
                        if len(old_eps) >= 4:
                            old_eps.pop(0)
                        if callback: callback(i)
                        if img_callback: img_callback(pred_x0, i)

                        if r != 9:
                            img = self.undo(x_t=img, t=ts - 1)
                            print(f"resampling trick : [step - {step}]_[repeat - {r}/{self.resampling_trick_repeat}]")
                else:
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=device, dtype=torch.long)
                    ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device,
                                         dtype=torch.long)

                    if mask is not None:
                        assert x0 is not None
                        if i < self.first_n_repaint:
                            img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                            img = img_orig * mask + (1. - mask) * img

                    outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                              quantize_denoised=quantize_denoised, temperature=temperature,
                                              noise_dropout=noise_dropout, score_corrector=score_corrector,
                                              corrector_kwargs=corrector_kwargs,
                                              unconditional_guidance_scale=unconditional_guidance_scale,
                                              unconditional_conditioning=unconditional_conditioning,
                                              old_eps=old_eps, t_next=ts_next,
                                              dynamic_threshold=dynamic_threshold)
                    img, pred_x0, e_t = outs
                    old_eps.append(e_t)
                    if len(old_eps) >= 4:
                        old_eps.pop(0)
                    if callback: callback(i)
                    if img_callback: img_callback(pred_x0, i)

                    if index % log_every_t == 0 or index == total_steps - 1:
                        intermediates['x_inter'].append(img)
                        intermediates['pred_x0'].append(pred_x0)

        return img, intermediates, None

    def undo(self, x_t, t):
        beta = extract_into_tensor(self.betas, t, x_t.shape)
        x_t_forward = torch.sqrt(1 - beta) * x_t + torch.sqrt(beta) * torch.randn_like(x_t)
        return x_t_forward

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t, _ = self.model.apply_model(x, t, c)
            else:
                model_t, cond_output_dict = self.model.apply_model(x, t, c)
                model_uncond, cond_output_dict = self.model.apply_model(x, t, unconditional_conditioning)

                if isinstance(model_t, tuple):
                    model_t, _ = model_t
                if isinstance(model_uncond, tuple):
                    model_uncond, _ = model_uncond
                e_t = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            if dynamic_threshold is not None:
                pred_x0 = norm_thresholding(pred_x0, dynamic_threshold)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t