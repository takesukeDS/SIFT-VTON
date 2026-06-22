# Design: Publish Training Code for SIFT-VTON

Date: 2026-06-23
Branch: orph_siftvton

## Goal

Add training code (`train_siftvton.py`) and training instructions (README.md) to the public release branch, completing the open-research certificate requirements for ICPR2026.

## Files

### Add to branch

| File | Action | Notes |
|---|---|---|
| `train_siftvton.py` | Copy `train_yahavton.py`, rename only | Zero "yahavton" occurrences in content; no internal edits needed |
| `cldm/logger.py` | Add (currently untracked) | Required by `from cldm.logger import ImageLogger` |

`configs/SIFT-VTON_sift_loss_ave.yaml` is already staged; `--config_name SIFT-VTON_sift_loss_ave` works as-is.

### Out of scope

- `visualize_attn_siftvton.py` (may be added later if reviewers request it)
- `wandb_config.yaml` template (WandB is optional via `--wandb_config_path`; no README mention needed)
- `train.sh` (lab-specific paths; not published)

## README.md changes

### 1. TODO checkboxes (lines 7–15)

Tick all now-complete items:

- `[x] Filtered SIFT correspondences(json) on VITON-HD dataset`
- `[x] Code for training SIFT-VTON`
- `[x] Code for inference`
- `[x] Trained weights of SIFT-VTON`
- `[x] Instructions for training`
- `[x] Instructions for inference`

### 2. New `## Training` section

Inserted before `## Citation`. Contents:

1. **Pretrained checkpoint**: Instruct users to download StableVITON's pretrained checkpoint from [StableVITON](https://github.com/rlawjdghek/stableviton). Specific filename/URL is a placeholder for manual fill-in.
2. **Data prep**: One-liner referencing the existing "For SIFT matching and filtering" section — `sift_matching/` under `train/` is required when using `--use_sift_loss`.
3. **Example command**:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_siftvton.py \
    --config_name SIFT-VTON_sift_loss_ave \
    --data_root_dir [VITON-HD dataset dir] \
    --pretrained_path [StableVITON checkpoint path] \
    --save_root_dir [output dir] \
    --save_name train_siftvton \
    --transform_size hflip shiftscale \
    --transform_color hsv bright_contrast \
    --use_sift_loss \
    --sift_loss_scale 0.0005 \
    --snr_gamma 5.0 \
    --accum_iter 8 \
    --batch_size 4 \
    --save_every_n_epochs 20 \
    --valid_epoch_freq 10
```

Key flags:

| Flag | Value | Purpose |
|---|---|---|
| `--use_sift_loss` | (toggle) | Enable SIFT correspondence supervision on cross-attention |
| `--sift_loss_scale` | `0.0005` | Weight of SIFT loss relative to diffusion loss |
| `--snr_gamma` | `5.0` | Min-SNR loss weighting |
| `--accum_iter` | `8` | Gradient accumulation steps (effective batch = batch_size × n_gpus × accum_iter) |

## Constraints

- WandB is already optional in code (`--wandb_config_path` defaults to None); not mentioned in README.
- StableVITON checkpoint URL/filename placeholder must be filled in manually.
