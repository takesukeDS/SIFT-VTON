---
license: cc-by-nc-sa-4.0
tags:
  - virtual-try-on
  - diffusion
  - image-to-image
  - computer-vision
datasets:
  - viton-hd
pipeline_tag: image-to-image
arxiv: 2605.01296
---

# SIFT-VTON: Geometric Correspondence Supervision on Cross-Attention for Virtual Try-On

**ICPR 2026**

SIFT-VTON is a diffusion-based virtual try-on model that uses SIFT feature correspondences between a garment image and a person image to supervise cross-attention maps during training, improving geometric alignment in the generated results.

Paper: [arXiv:2605.01296](https://arxiv.org/abs/2605.01296)

This model is derived from [StableVITON](https://github.com/rlawjdghek/stableviton) and built on a Stable Diffusion backbone.

The code repository is available at [takesukeDS/SIFT-VTON](https://github.com/takesukeDS/SIFT-VTON).

## Model Files

| File | Description |
|---|---|
| `model.ckpt` | Model checkpoint |
| `config.yaml` | Model architecture config |

## Requirements

Clone the code repository and set up the environment:

```bash
git clone https://github.com/takesukeDS/SIFT-VTON
cd SIFT-VTON

conda create -n siftvton python==3.12.8 -y
conda activate siftvton

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib einops omegaconf yacs
pip install pytorch-lightning==2.5.2
pip install open-clip-torch==3.1.0
pip install diffusers==0.34.0
pip install scipy==1.16.1
pip install transformers==4.55.0
conda install -c anaconda ipython -y
pip install scikit-image clean-fid albumentations==2.0.8
pip3 install -U xformers==0.0.31.post1
pip install tensorboard
pip install accelerate==1.10.0
pip install numpy==2.2.6
```

## Data

Download the [VITON-HD dataset](https://github.com/shadow2496/VITON-HD) and prepare the following directory structure:

```
[data_root_dir]
└── test
    |-- image
    |-- image-densepose
    |-- agnostic-v3.2
    |-- agnostic-mask
    |-- cloth
    |-- cloth-mask
```

A pairs file `siftvton_test_pairs.txt` is also required under `[data_root_dir]`, listing image and cloth filenames one pair per line:
```
image_00001.jpg cloth_00001.jpg
image_00002.jpg cloth_00002.jpg
...
```

## Inference

```bash
python inference_hf.py \
    --repo_id takesukeDS/SIFT-VTON \
    --data_root_dir [data_root_dir] \
    --save_dir [output_dir] \
    --phase test \
    --batch_size 4 \
    --start_from_noised_agn \
    --cfg_scale 1.5 \
    --repaint
```

The model and config are downloaded automatically from this Hub repository on the first run and cached locally under `~/.cache/huggingface/hub/`.

### Key inference arguments

| Argument | Default | Description |
|---|---|---|
| `--repo_id` | — | This Hub repo (`takesukeDS/SIFT-VTON`) |
| `--phase` | `test` | `test` for the test split, `train` for the training split |
| `--cfg_scale` | `1.0` | Classifier-free guidance scale |
| `--denoise_steps` | `50` | Number of PLMS denoising steps |
| `--start_from_noised_agn` | off | Start denoising from noised agnostic image instead of pure noise (recommended) |
| `--repaint` | off | Paste back the unmasked region from the original image after generation (recommended) |
| `--unpair` | off | Run unpaired inference (person and garment from different samples) |
| `--batch_size` | `16` | Batch size |
| `--seed` | `1235` | Random seed |

## Citation

```bibtex
@misc{takemoto2026siftvton,
  title         = {{SIFT-VTON}: Geometric Correspondence Supervision on Cross-Attention for Virtual Try-On},
  author        = {Takemoto, Kosuke and Koshinaka, Takafumi},
  year          = {2026},
  eprint        = {2605.01296},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.01296}
}
```

## License

Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.
