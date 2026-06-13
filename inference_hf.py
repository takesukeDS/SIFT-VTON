"""
HuggingFace Hub-compatible inference for SIFT-VTON.

Load from HuggingFace Hub:
    python inference_hf.py \
        --repo_id takesukeDS/SIFT-VTON \
        --data_root_dir ~/data/zalando-hd-resized \
        --save_dir ./output \
        --phase test \
        --start_from_noised_agn \
        --cfg_scale 1.5 \
        --repaint

Load from local files:
    python inference_hf.py \
        --config_path ./configs/SIFT-VTON_sift_loss_ave.yaml \
        --model_load_path ./ckpts/siftvton.ckpt \
        --data_root_dir ~/data/zalando-hd-resized \
        --save_dir ./output \
        --phase test \
        --start_from_noised_agn \
        --cfg_scale 1.5 \
        --repaint
"""

import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img, set_seed


def build_args():
    parser = argparse.ArgumentParser()

    # HuggingFace Hub loading (alternative to --config_path + --model_load_path)
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace Hub repo ID, e.g. 'takesukeDS/SIFT-VTON'")
    parser.add_argument("--config_filename", type=str, default="config.yaml",
                        help="Filename of the config inside the Hub repo")
    parser.add_argument("--model_filename", type=str, default="model.ckpt",
                        help="Filename of the checkpoint inside the Hub repo")
    parser.add_argument("--revision", type=str, default=None,
                        help="Hub revision (branch / tag / commit hash)")

    # Local loading
    parser.add_argument("--config_path", type=str, default=None,
                        help="Local path to config yaml (used when --repo_id is not set)")
    parser.add_argument("--model_load_path", type=str, default=None,
                        help="Local path to model checkpoint (used when --repo_id is not set)")

    # Inference options
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./DATA/zalando-hd-resized")
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./samples")
    parser.add_argument("--resampling_trick", action="store_true")
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--start_from_noised_agn", action="store_true")
    parser.add_argument("--seed", type=int, default=1235)
    parser.add_argument("--cfg_scale", type=float, default=1.0)

    args = parser.parse_args()

    if args.repo_id is None and (args.config_path is None or args.model_load_path is None):
        parser.error("Provide either --repo_id or both --config_path and --model_load_path.")

    return args


def resolve_paths(args):
    """Download config and checkpoint from HF Hub when --repo_id is set."""
    if args.repo_id is None:
        return args.config_path, args.model_load_path

    from huggingface_hub import hf_hub_download

    print(f"Downloading config from Hub: {args.repo_id}/{args.config_filename}")
    config_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.config_filename,
        revision=args.revision,
    )

    print(f"Downloading checkpoint from Hub: {args.repo_id}/{args.model_filename}")
    model_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.model_filename,
        revision=args.revision,
    )

    return config_path, model_path


@torch.no_grad()
def main(args):
    config_path, model_load_path = resolve_paths(args)

    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    config = OmegaConf.load(config_path)
    config.model.params.img_H = img_H
    config.model.params.img_W = img_W
    config.model.params.unet_config.params.use_sift_loss = False
    params = config.model.params

    model = create_model(config_path=None, config=config)
    load_cp = torch.load(model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    model.load_state_dict(load_cp)
    model = model.cuda()
    model.eval()

    sampler = PLMSSampler(model, resampling_trick=args.resampling_trick)
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        phase=args.phase,
        is_paired=not args.unpair,
        is_sorted=True,
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H // 8, img_W // 8)
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        z, c = model.get_input(batch, params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = [model.learnable_vector.repeat(bs, 1, 1)]
        uc_full = {"c_concat": None, "c_crossattn": uc_cross}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = None
        if args.start_from_noised_agn:
            start_code = model.q_sample(c["first_stage_cond"][:, :4], ts)

        samples, _, _ = sampler.sample(
            args.denoise_steps,
            bs,
            shape,
            c,
            x_T=start_code,
            verbose=False,
            eta=args.eta,
            unconditional_guidance_scale=args.cfg_scale,
            unconditional_conditioning=uc_full,
        )

        x_samples = model.decode_first_stage(samples)
        for sample_idx, (x_sample, fn, cloth_fn) in enumerate(zip(x_samples, batch["img_fn"], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample, round=True)
            if args.repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy() + 1) / 2 * 255 + 0.5)
                repaint_agn_mask_img = batch["agn_mask_orig"][sample_idx].cpu().numpy()
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1 - repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img + 0.5)
            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            cv2.imwrite(to_path, x_sample_img[:, :, ::-1])


if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)
    main(args)
