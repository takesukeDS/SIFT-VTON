# [ICPR2026] SIFT-VTON: Geometric Correspondence Supervision on Cross-Attention for Virtual Try-On

![teaser](assets/teaser.jpg)&nbsp;

This repository is derived from [StableVITON](https://github.com/rlawjdghek/stableviton).

## TODO
- [x] Code for preprocessing
- [x] Filtered SIFT correspondences(json) on VITON-HD dataset
- [x] Code for training SIFT-VTON
- [x] Code for inference 
- [x] Trained weights of SIFT-VTON
- [x] Instructions for preprocessing
- [x] Instructions for training 
- [x] Instructions for inference 

## Environments
```bash
git clone https://github.com/takesukeDS/SIFT-VTON
cd SIFT-VTON

conda create -n siftvton python==3.12.8 -y
conda activate siftvton

# install packages
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

## Weights and Data Preparation
Our weights are publicly available on huggingface [takesuke/SIFT-VTON](https://huggingface.co/takesuke/SIFT-VTON), 
and inference_hf.py automatically downloads the weights from huggingface when you run inference.


### For inference (minimum data preparation)
You can download the VITON-HD dataset from [here](https://github.com/shadow2496/VITON-HD).<br>

Pairs definition files for the train/val/test splits (`siftvton_train_pairs.txt`, `siftvton_val_pairs.txt`, `siftvton_test_pairs.txt`) are available on the [huggingface model repo](https://huggingface.co/takesuke/SIFT-VTON). Download them and place them directly under the dataset root directory:
```bash
hf download takesuke/SIFT-VTON siftvton_train_pairs.txt siftvton_val_pairs.txt siftvton_test_pairs.txt --local-dir [VITON-HD dataset dir]
```

For inference, the following VITON-HD dataset structure is required:
```
[VITON-HD dataset dir]
|-- train
|   |-- image
|   |-- image-densepose
|   |-- agnostic-v3.2
|   |-- agnostic-mask
|   |-- cloth
|   |-- cloth-mask
|   |-- gt_cloth_warped_mask (for ATV loss)
|-- test
|   |-- image
|   |-- image-densepose
|   |-- agnostic-v3.2
|   |-- agnostic-mask
|   |-- cloth
|   |-- cloth-mask
|-- siftvton_train_pairs.txt
|-- siftvton_val_pairs.txt
|-- siftvton_test_pairs.txt
```

### For SIFT matching and filtering (Optional)
If you want to compute SIFT correspondences manually for training,
download Fine-grained Parsing in [GP-VTON](https://github.com/xiezhy6/GP-VTON) for VITON-HD dataset.
We used garment parts segmentation provided by GP-VTON for better spatial matching.
We refer the directory where you unzipped the cloth parsing as `[cloth_parsing_dir]` in the following instructions.

## Preprocessing: SIFT matching and filtering for training on VITON-HD dataset
The code below saves the filtered SIFT correspondences in json for each image pair in the VITON-HD dataset. 
```
python save_sift_matching.py --save_dir [output dir] --data_root_dir [VITON-HD dataset dir] --data_type train --cloth_segmentation_base_dir [cloth_parsing_dir] 
```
Move `[output dir]/train` into `[VITON-HD dataset dir]/train` as `sift_matching` like below if you want to run training code:
```
mv [output dir]/train [VITON-HD dataset dir]/train/sift_matching
```

Alternatively, you can download the precomputed SIFT correspondences for the train split from the [huggingface model repo](https://huggingface.co/takesuke/SIFT-VTON) and place it under the `train` directory of the VITON-HD dataset:
```bash
hf download takesuke/SIFT-VTON sift_matching.zip --local-dir [VITON-HD dataset dir]/train
cd [VITON-HD dataset dir]/train
unzip sift_matching.zip
```

## Training

Download StableVITON's pretrained checkpoint for VITON-HD from [StableVITON](https://github.com/rlawjdghek/stableviton) and place it under `ckpts/`. 

SIFT matching data (`sift_matching/` under `[VITON-HD dataset dir]/train/`) is required — see [For SIFT matching and filtering](#for-sift-matching-and-filtering-optional) above.

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_siftvton.py \
    --config_name SIFT-VTON_sift_loss_ave \
    --data_root_dir [VITON-HD dataset dir] \
    --pretrained_path ckpts/[StableVITON checkpoint filename] \
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

| Flag | Value | Description |
|---|---|---|
| `--use_sift_loss` | toggle | Enable SIFT correspondence supervision on cross-attention |
| `--sift_loss_scale` | `0.0005` | Weight of SIFT loss relative to diffusion loss |
| `--snr_gamma` | `5.0` | Min-SNR loss weighting |
| `--accum_iter` | `8` | Gradient accumulation steps (effective batch = `batch_size × n_gpus × accum_iter`) |

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
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
