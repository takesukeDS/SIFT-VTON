# [ICPR2026] SIFT-VTON: Geometric Correspondence Supervision on Cross-Attention for Virtual Try-On

![teaser](assets/teaser.jpg)&nbsp;

This repository is derived from [StableVITON](https://github.com/rlawjdghek/stableviton).

## TODO
- [x] Code for preprocessing
- [ ] Filtered SIFT correspondences(json) on VITON-HD dataset
- [ ] Code for training SIFT-VTON
- [ ] Code for inference 
- [ ] Trained weights of SIFT-VTON
- [x] Instructions for preprocessing
- [ ] Instructions for training 
- [ ] Instructions for inference 

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

## Weights and Data
Our weights are not publicly available yet.
You can download the VITON-HD dataset from [here](https://github.com/shadow2496/VITON-HD).<br>
Additionally, we used garment parts segmentation provided by GP-VTON for better spatial matching.
Download Fine-grained Parsing in [GP-VTON](https://github.com/xiezhy6/GP-VTON) for VITON-HD dataset.
We refer the directory where you unzipped the cloth parsing as `[cloth_parsing_dir]` in the following instructions. 

For inference, the following VITON-HD dataset structure is required:
```
train
|-- image
|-- image-densepose
|-- agnostic
|-- agnostic-mask
|-- cloth
|-- cloth_mask
|-- gt_cloth_warped_mask (for ATV loss)

test
|-- image
|-- image-densepose
|-- agnostic
|-- agnostic-mask
|-- cloth
|-- cloth_mask
```

## Preprocessing: SIFT matching and filtering for training on VITON-HD dataset
The code below saves the filtered SIFT correspondences in json for each image pair in the VITON-HD dataset. 
```
python save_sift_matching.py --save_dir [output dir] --data_root_dir [VITON-HD dataset dir] --data_type train --cloth_segmentation_base_dir [cloth_parsing_dir] 
```
Move `[output dir]/train` into `[VITON-HD dataset dir]/train` as `sift_matching` like below:
```
mv [output dir]/train [VITON-HD dataset dir]/train/sift_matching
```

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
