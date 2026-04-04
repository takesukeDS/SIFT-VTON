import os
import random
from os.path import join as opj

import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
import json

from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torchvision import transforms


def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None
):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = cv2.imread(p)
    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img
def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img


DENSEPOSE_SEGM_RGB_TORSO = [ 20,  80, 194]
DENSEPOSE_SEGM_RGB_RIGHT_ARM = [
    [170, 189, 105], #right_arm_upper_inside
    [216, 186,  86], #right_arm_upper_outside
    [240, 199,  60], #right_arm_lower_inside
    [251, 220,  36], #right_arm_lower_outside
]
DENSEPOSE_SEGM_RGB_LEFT_ARM = [
    [145, 191, 116], #left_arm_upper_inside
    [192, 188,  96], #left_arm_upper_outside
    [228, 192,  74], #left_arm_lower_inside
    [252, 206,  46], #left_arm_lower_outside
]


DENSEPOSE_SEGM_RGB_RIGHT_ARM_RED = [
    170, 216, 240, 251
]
DENSEPOSE_SEGM_RGB_LEFT_ARM_RED = [
    145, 192, 228, 252
]
DENSEPOSE_SEGM_RGB_TORSO_RED = [
    20
]

GPVTON_SEGM_TORSO_RED = [
    5, 6, 7
]
GPVTON_SEGM_LEFT_SLEEVE_RED = [
    21
]
GPVTON_SEGM_RIGHT_SLEEVE_RED = [
    22
]

def gpvton_to_left_sleeve_mask(segm_np):
    segm_np_red = segm_np[:, :, 0]
    mask_sleeve = np.isin(segm_np_red, GPVTON_SEGM_LEFT_SLEEVE_RED)
    return mask_sleeve

def gpvton_to_right_sleeve_mask(segm_np):
    segm_np_red = segm_np[:, :, 0]
    mask_sleeve = np.isin(segm_np_red, GPVTON_SEGM_RIGHT_SLEEVE_RED)
    return mask_sleeve

def gpvton_to_torso_mask(segm_np):
    segm_np_red = segm_np[:, :, 0]
    mask_torso = np.isin(segm_np_red, GPVTON_SEGM_TORSO_RED)
    return mask_torso


def densepose_to_armmask(segm_np):
    segm_np_red = segm_np[:, :, 0]
    mask_arm = np.isin(segm_np_red, DENSEPOSE_SEGM_RGB_RIGHT_ARM_RED + DENSEPOSE_SEGM_RGB_LEFT_ARM_RED)
    return mask_arm

def densepose_to_leftarmmask(segm_np):
    segm_np_red = segm_np[:, :, 0]
    mask_arm = np.isin(segm_np_red, DENSEPOSE_SEGM_RGB_LEFT_ARM_RED)
    return mask_arm

def densepose_to_rightarmmask(segm_np):
    segm_np_red = segm_np[:, :, 0]
    mask_arm = np.isin(segm_np_red, DENSEPOSE_SEGM_RGB_RIGHT_ARM_RED)
    return mask_arm


def densepose_to_torsomask(segm_np):
    segm_np_red = segm_np[:, :, 0]
    mask_torso = np.isin(segm_np_red, DENSEPOSE_SEGM_RGB_TORSO_RED)
    return mask_torso


def round_tuple(t):
  return tuple(map(lambda x: round(x), t))

def transpose_pt(pt):
  return (pt[1], pt[0])

def gaussian_filter(image, sigma, kernel_size=3):
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        image = image.numpy()
    res = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    if is_tensor:
        res = torch.from_numpy(res)
    return res

class VITONHDDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W,
            phase,
            is_paired=True, 
            is_sorted=False,
            transform_size=None, 
            transform_color=None,
            torso_extraction_method="none",
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if phase in ["train", "val"] else "test"
        self.is_test = phase in ["val", "test"]
        self.phase = phase
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0
        self.torso_extraction_method = torso_extraction_method
        self.use_hybvton_densepose_torso = kwargs.get("use_hybvton_densepose_torso", False)

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]
        transform_hflip_lst = []
    
        if transform_size is not None:
            if "hflip" in transform_size:
                transform_hflip_lst.append(A.HorizontalFlip(p=0.5))

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))

        self.transform_crop_person = A.Compose(
                transform_crop_person_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image",
                                    "hybvton_warped_cloth": "image",
                                    "hybvton_warped_mask": "image",
                                    }
        )
        self.transform_crop_cloth = A.Compose(
                transform_crop_cloth_lst,
                additional_targets={"cloth_mask":"image"}
        )

        self.transform_size = A.Compose(
                transform_size_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "cloth":"image", 
                                    "cloth_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image",
                                    "image_densepose_hybvton":"image",
                                    }
            )
        self.transform_hflip = A.Compose(
                transform_hflip_lst,
                additional_targets={"agn":"image",
                                    "agn_mask":"image",
                                    "cloth":"image",
                                    "cloth_mask":"image",
                                    "cloth_mask_warped":"image",
                                    "cloth_warped":"image",
                                    "image_densepose":"image",
                                    "image_parse":"image",
                                    "gt_cloth_warped_mask":"image",
                                    "hybvton_warped_cloth": "image",
                                    "hybvton_warped_mask": "image",
                                    }
            )
        #### spatial aug <<<<

        #### non-spatial aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
                elif t == "bright_contrast":
                    transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={"agn":"image", 
                                    "cloth":"image",  
                                    "cloth_warped":"image",
                                    }
            )
        #### non-spatial aug <<<<
                    
        assert not (self.phase == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"hybvton_{self.phase}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        if self.transform_size is None and self.transform_color is None:
            raise NotImplementedError("Never reached by original code")
            agn = imread(
                opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]), 
                self.img_H, 
                self.img_W
            )
            agn_mask = imread(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
            )
            cloth = imread(
                opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]), 
                self.img_H, 
                self.img_W
            )

            gt_cloth_warped_mask = imread(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]), 
                self.img_H, 
                self.img_W, 
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)

            hybvton_warped_cloth = imread(
                opj(self.drd, self.data_type, "hybvton_warped_cloth_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
                self.img_H,
                self.img_W,
            )
            hybvton_warped_mask = imread(
                opj(self.drd, self.data_type, "hybvton_warped_mask_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
                self.img_H,
                self.img_W,
                is_mask=True
            )

            image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
            image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)
            hybvton_warped_mask = (hybvton_warped_mask / 255 * agn_mask / 255)
            agn = agn * (1 - hybvton_warped_mask[:,:,None]) + hybvton_warped_cloth * hybvton_warped_mask[:,:,None]
            agn = agn.astype(np.uint8)
            hybvton_warped_mask = (hybvton_warped_mask * 255).astype(np.uint8)
            agn_mask_orig = 255 - agn_mask
            agn_mask = np.clip(agn_mask - hybvton_warped_mask, 0, 255)
            agn_mask = 255 - agn_mask

        else:
            agn = imread_for_albu(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
            agn_mask = imread_for_albu(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), is_mask=True)
            cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
            cloth_mask = imread_for_albu(opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]), is_mask=True, cloth_mask_check=True)
            
            gt_cloth_warped_mask = imread_for_albu(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]),
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)
            hybvton_warped_cloth = imread_for_albu(
                opj(self.drd, self.data_type, "hybvton_warped_cloth_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
            )
            hybvton_warped_mask = imread_for_albu(
                opj(self.drd, self.data_type, "hybvton_warped_mask_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
                is_mask=True
            )
                
            image = imread_for_albu(opj(self.drd, self.data_type, "image", self.im_names[idx]))
            image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))
            if self.use_hybvton_densepose_torso:
                image_densepose_hybvton = imread_for_albu(
                    opj(self.drd, self.data_type, "image-densepose_hybvton", self.im_names[idx].replace(".jpg", ".png")))

            if self.transform_size is not None:
                kwargs_for_transform = dict(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    cloth=cloth,
                    cloth_mask=cloth_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                )
                if self.use_hybvton_densepose_torso:
                    kwargs_for_transform["image_densepose_hybvton"] = image_densepose_hybvton
                transformed = self.transform_size(**kwargs_for_transform)
                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                image_densepose=transformed["image_densepose"]
                gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]
                if self.use_hybvton_densepose_torso:
                    image_densepose_hybvton=transformed["image_densepose_hybvton"]

                cloth=transformed["cloth"]
                cloth_mask=transformed["cloth_mask"]

            hybvton_warped_mask = (hybvton_warped_mask / 255 * agn_mask / 255)
            agn_mask_orig = 255 - agn_mask
            agn_orig = agn

            if self.torso_extraction_method != "none":
                densepose_for_torso = image_densepose_hybvton if self.use_hybvton_densepose_torso else image_densepose
                if self.torso_extraction_method == "torso_segment":
                    densepose_torso_mask = densepose_for_torso[:, :, 0] == DENSEPOSE_SEGM_RGB_TORSO[0]
                    densepose_torso_mask = densepose_torso_mask.astype(np.float32)
                    hybvton_warped_mask = hybvton_warped_mask * densepose_torso_mask
                elif self.torso_extraction_method == "arm_elimination":
                    arm_mask = densepose_to_armmask(densepose_for_torso).astype(np.float32)
                    hybvton_warped_mask = hybvton_warped_mask * (1 - arm_mask)

            agn = agn * (1 - hybvton_warped_mask[:, :, None]) + hybvton_warped_cloth * hybvton_warped_mask[:, :, None]
            agn = agn.astype(np.uint8)
            hybvton_warped_mask = (hybvton_warped_mask * 255).astype(np.uint8)
            agn_mask = np.clip(agn_mask - hybvton_warped_mask, 0, 255)

            if self.transform_hflip is not None:
                transformed = self.transform_hflip(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    cloth=cloth,
                    cloth_mask=cloth_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    hybvton_warped_cloth=hybvton_warped_cloth,
                    hybvton_warped_mask=hybvton_warped_mask,
                )

                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                image_densepose=transformed["image_densepose"]
                gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]
                hybvton_warped_cloth=transformed["hybvton_warped_cloth"]
                hybvton_warped_mask=transformed["hybvton_warped_mask"]

                cloth=transformed["cloth"]
                cloth_mask=transformed["cloth_mask"]

            if self.transform_crop_person is not None:
                transformed_image = self.transform_crop_person(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    hybvton_warped_cloth=hybvton_warped_cloth,
                    hybvton_warped_mask=hybvton_warped_mask,
                )

                image=transformed_image["image"]
                agn=transformed_image["agn"]
                agn_mask=transformed_image["agn_mask"]
                image_densepose=transformed_image["image_densepose"]
                gt_cloth_warped_mask=transformed_image["gt_cloth_warped_mask"]
                hybvton_warped_cloth=transformed_image["hybvton_warped_cloth"]
                hybvton_warped_mask=transformed_image["hybvton_warped_mask"]

            if self.transform_crop_cloth is not None:
                transformed_cloth = self.transform_crop_cloth(
                    image=cloth,
                    cloth_mask=cloth_mask
                )

                cloth=transformed_cloth["image"]
                cloth_mask=transformed_cloth["cloth_mask"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=image, 
                    agn=agn, 
                    cloth=cloth,
                )

                image=transformed["image"]
                agn=transformed["agn"]
                cloth=transformed["cloth"]

                agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)
                
            agn = norm_for_albu(agn)
            agn_orig = norm_for_albu(agn_orig)
            agn_mask_orig = norm_for_albu(agn_mask_orig, is_mask=True)
            agn_mask_orig = (agn_mask_orig > 0.5).astype(np.float32)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            cloth = norm_for_albu(cloth)
            cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
            image = norm_for_albu(image)
            image_densepose = norm_for_albu(image_densepose)
            gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)
            hybvton_warped_cloth = norm_for_albu(hybvton_warped_cloth)
            hybvton_warped_mask = norm_for_albu(hybvton_warped_mask, is_mask=True)
            
        return dict(
            agn=agn,
            agn_orig=agn_orig,
            agn_mask=agn_mask,
            agn_mask_orig=agn_mask_orig,
            cloth=cloth,
            cloth_mask=cloth_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
            hybvton_warped_cloth=hybvton_warped_cloth,
            hybvton_warped_mask=hybvton_warped_mask,
        )


class VITONHDDatasetWithGAN(VITONHDDataset):
    # parse map
    labels = {
        0: ['background', [0, 10]],
        1: ['hair', [1, 2]],
        2: ['face', [4, 13]],
        3: ['upper', [5, 6, 7]],
        4: ['bottom', [9, 12]],
        5: ['left_arm', [14]],
        6: ['right_arm', [15]],
        7: ['left_leg', [16]],
        8: ['right_leg', [17]],
        9: ['left_shoe', [18]],
        10: ['right_shoe', [19]],
        11: ['socks', [8]],
        12: ['noise', [3, 11]]
    }

    def __init__(self, data_root_dir, img_H, img_W, phase, is_paired=True, is_sorted=False, transform_size=None,
                 transform_color=None, semantic_nc=None, use_preprocessed=False, **kwargs):
        super().__init__(data_root_dir, img_H, img_W, phase, is_paired, is_sorted, transform_size, transform_color,
                         **kwargs)
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.semantic_nc = semantic_nc
        self.use_preprocessed = use_preprocessed

    def build_parse_agnostic(self, idx):
        # load parsing image
        parse_name = opj(self.drd, self.data_type, 'image-parse-v3', self.im_names[idx]).replace('.jpg', '.png')
        im_parse_pil_big = Image.open(parse_name)
        im_parse_pil = TF.resize(im_parse_pil_big,
                                 (int(self.img_H*self.resize_ratio_H), int(self.img_W*self.resize_ratio_W)),
                                 interpolation=InterpolationMode.NEAREST)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))


        parse_map = torch.FloatTensor(20, self.img_H, self.img_W).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.img_H, self.img_W).zero_()

        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.img_H, self.img_W).zero_()
        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # load image-parse-agnostic
        image_parse_agnostic = Image.open(
            osp.join(parse_name.replace('image-parse-v3', 'image-parse-agnostic-v3.2')))
        image_parse_agnostic = transforms.Resize(self.img_W, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()

        parse_agnostic_map = torch.FloatTensor(20, self.img_H, self.img_W).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.img_H, self.img_W).zero_()
        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        return new_parse_agnostic_map

    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        agn = imread_for_albu(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
        agn_mask = imread_for_albu(
            opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")),
            is_mask=True)
        cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
        cloth_mask = imread_for_albu(opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]),
                                     is_mask=True, cloth_mask_check=True)

        gt_cloth_warped_mask = imread_for_albu(
            opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]),
            is_mask=True
        ) if not self.is_test else np.zeros_like(agn_mask)
        hybvton_warped_cloth = imread_for_albu(
            opj(self.drd, self.data_type, "hybvton_warped_cloth_" + self.pair_key,
                self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
        )
        hybvton_warped_mask = imread_for_albu(
            opj(self.drd, self.data_type, "hybvton_warped_mask_" + self.pair_key,
                self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
            is_mask=True
        )

        image = imread_for_albu(opj(self.drd, self.data_type, "image", self.im_names[idx]))
        image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))
        image_densepose_hybvton = imread_for_albu(opj(
            self.drd, self.data_type, "image-densepose_hybvton", self.im_names[idx].replace(".jpg", ".png")))
        densepose_torso_mask = image_densepose_hybvton[:, :, 0] == DENSEPOSE_SEGM_RGB_TORSO[0]
        densepose_torso_mask = densepose_torso_mask.astype(np.uint8)[:,:,None]

        if self.transform_size is not None:
            transformed = self.transform_size(
                image=image,
                agn=agn,
                agn_mask=agn_mask,
                cloth=cloth,
                cloth_mask=cloth_mask,
                image_densepose=image_densepose,
                gt_cloth_warped_mask=gt_cloth_warped_mask,
                densepose_torso_mask=densepose_torso_mask,
            )
            image = transformed["image"]
            agn = transformed["agn"]
            agn_mask = transformed["agn_mask"]
            image_densepose = transformed["image_densepose"]
            gt_cloth_warped_mask = transformed["gt_cloth_warped_mask"]
            densepose_torso_mask = transformed["densepose_torso_mask"]

            cloth = transformed["cloth"]
            cloth_mask = transformed["cloth_mask"]

        # Hybvton: images and masks we add for refinement will have format c h w
        densepose_torso_mask = densepose_torso_mask.astype(np.float32).transpose(2, 0, 1)
        hybvton_warped_mask = (hybvton_warped_mask / 255 * agn_mask / 255)
        agn_mask_orig = 255 - agn_mask
        agn_orig = agn
        agn = agn * (1 - hybvton_warped_mask[:, :, None]) + hybvton_warped_cloth * hybvton_warped_mask[:, :, None]
        agn = agn.astype(np.uint8)
        agn_orig = agn_orig.astype(np.uint8)
        hybvton_warped_mask = (hybvton_warped_mask * 255).astype(np.uint8)
        agn_mask = np.clip(agn_mask - hybvton_warped_mask, 0, 255)

        if self.transform_hflip is not None:
            transformed = self.transform_hflip(
                image=image,
                agn=agn,
                agn_mask=agn_mask,
                cloth=cloth,
                cloth_mask=cloth_mask,
                image_densepose=image_densepose,
                gt_cloth_warped_mask=gt_cloth_warped_mask,
                hybvton_warped_cloth=hybvton_warped_cloth,
                hybvton_warped_mask=hybvton_warped_mask,
            )

            image = transformed["image"]
            agn = transformed["agn"]
            agn_mask = transformed["agn_mask"]
            image_densepose = transformed["image_densepose"]
            gt_cloth_warped_mask = transformed["gt_cloth_warped_mask"]
            hybvton_warped_cloth = transformed["hybvton_warped_cloth"]
            hybvton_warped_mask = transformed["hybvton_warped_mask"]

            cloth = transformed["cloth"]
            cloth_mask = transformed["cloth_mask"]

        if self.transform_crop_person is not None:
            transformed_image = self.transform_crop_person(
                image=image,
                agn=agn,
                agn_mask=agn_mask,
                image_densepose=image_densepose,
                gt_cloth_warped_mask=gt_cloth_warped_mask,
                hybvton_warped_cloth=hybvton_warped_cloth,
                hybvton_warped_mask=hybvton_warped_mask,
            )

            image = transformed_image["image"]
            agn = transformed_image["agn"]
            agn_mask = transformed_image["agn_mask"]
            image_densepose = transformed_image["image_densepose"]
            gt_cloth_warped_mask = transformed_image["gt_cloth_warped_mask"]
            hybvton_warped_cloth = transformed_image["hybvton_warped_cloth"]
            hybvton_warped_mask = transformed_image["hybvton_warped_mask"]

        if self.transform_crop_cloth is not None:
            transformed_cloth = self.transform_crop_cloth(
                image=cloth,
                cloth_mask=cloth_mask
            )

            cloth = transformed_cloth["image"]
            cloth_mask = transformed_cloth["cloth_mask"]

        agn_mask = 255 - agn_mask
        if self.transform_color is not None:
            transformed = self.transform_color(
                image=image,
                agn=agn,
                cloth=cloth,
            )

            image = transformed["image"]
            agn = transformed["agn"]
            cloth = transformed["cloth"]

            agn = agn * agn_mask[:, :, None].astype(np.float32) / 255.0 + 128 * (
                        1 - agn_mask[:, :, None].astype(np.float32) / 255.0)

        agn = norm_for_albu(agn)
        agn_orig = norm_for_albu(agn_orig)
        agn_mask_orig = norm_for_albu(agn_mask_orig, is_mask=True)
        agn_mask = norm_for_albu(agn_mask, is_mask=True)
        cloth = norm_for_albu(cloth)
        cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
        image = norm_for_albu(image)
        image_densepose = norm_for_albu(image_densepose)
        gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)
        hybvton_warped_cloth = norm_for_albu(hybvton_warped_cloth)
        hybvton_warped_mask = norm_for_albu(hybvton_warped_mask, is_mask=True)

        # original warped cloth and mask
        warped_cloth_pil = Image.open(osp.join(self.drd, self.data_type, f'hybvton_warped_cloth_{self.pair_key}_orig',
                                               self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][
                                                   idx].replace(".jpg", ".png")))
        warped_cloth_pil = TF.resize(warped_cloth_pil,
                                     (int(self.img_H*self.resize_ratio_H), int(self.img_W*self.resize_ratio_W)),
                                     interpolation=InterpolationMode.BICUBIC)
        warped_cloth_np = np.array(warped_cloth_pil)

        warped_mask_pil = Image.open(osp.join(self.drd, self.data_type, f'hybvton_warped_mask_{self.pair_key}_orig',
                                              self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][
                                                  idx].replace(".jpg", ".png")))
        warped_mask_pil = TF.resize(warped_mask_pil,
                                    (int(self.img_H*self.resize_ratio_H), int(self.img_W*self.resize_ratio_W)),
                                    interpolation=InterpolationMode.NEAREST)
        warped_mask_np = np.array(warped_mask_pil)
        warped_mask_orig = torch.from_numpy(warped_mask_np >= 0.5).float()[None]
        warped_cloth_orig = self.transform(warped_cloth_np)

        parse_agnostic = self.build_parse_agnostic(idx)

        warped_cloth_processed = 0
        if self.use_preprocessed:
            warped_cloth_processed_pil = Image.open(
                osp.join(self.drd, self.data_type, f'hybvton_warped_cloth_{self.pair_key}_processed',
                         self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][
                             idx].replace(".jpg", ".png")))
            warped_cloth_processed_pil = TF.resize(warped_cloth_processed_pil,
                                         (int(self.img_H * self.resize_ratio_H), int(self.img_W * self.resize_ratio_W)),
                                         interpolation=InterpolationMode.BICUBIC)
            warped_cloth_processed = self.transform(warped_cloth_processed_pil)


        return dict(
            agn=agn,
            agn_orig=agn_orig,
            agn_mask=agn_mask,
            cloth=cloth,
            cloth_mask=cloth_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
            hybvton_warped_cloth=hybvton_warped_cloth,
            hybvton_warped_mask=hybvton_warped_mask,
            agn_mask_orig=agn_mask_orig,
            warped_mask_orig=warped_mask_orig,
            warped_cloth_orig=warped_cloth_orig,
            parse_agnostic=parse_agnostic,
            densepose_torso_mask=densepose_torso_mask,
            warped_cloth_processed=warped_cloth_processed,
        )

LATENT_SIZES = [(16, 12), (32, 24), (64, 48)]
class YahaVITONHDDataset(Dataset):
    def __init__(
            self,
            data_root_dir,
            img_H,
            img_W,
            phase,
            is_paired=True,
            is_sorted=False,
            transform_size=None,
            transform_color=None,
            torso_extraction_method="none",
            hist_to_density=True,
            hist_only_max=False,
            hist_blured=False,
            use_sift_matching=False,
            use_explicit_warping=False,
            seed=None,
            exclude_image_ids=tuple(),
            **kwargs
    ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        if img_H != 512 or img_W != 384:
            raise NotImplementedError("Currently only support 512x384 input")
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if phase in ["train", "val"] else "test"
        self.is_test = phase in ["val", "test"]
        self.phase = phase
        self.seed = seed
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0
        self.hflip_p = 0.5
        self.torso_extraction_method = torso_extraction_method
        self.use_hybvton_densepose_torso = kwargs.get("use_hybvton_densepose_torso", False)
        self.hist_to_density = hist_to_density
        self.hist_only_max = hist_only_max
        self.hist_blured = hist_blured
        if self.hist_only_max and self.hist_blured:
            raise NotImplementedError("hist_only_max and hist_blured cannot be true at the same time")
        self.use_sift_matching = use_sift_matching
        if use_sift_matching and (self.pair_key == "unpaired" or self.data_type == "test"):
            raise NotImplementedError("SIFT matching only supports paired training set")
        if use_explicit_warping:
            raise NotImplementedError("Explicit warping is not supported now")
        self.use_explicit_warping = use_explicit_warping
        if use_sift_matching and use_explicit_warping:
            raise NotImplementedError("SIFT matching and explicit warping cannot be used at the same time")

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H * self.resize_ratio_H), int(img_W * self.resize_ratio_W))]
        transform_hflip_lst = []

        if transform_size is not None:
            if "hflip" in transform_size:
                # The probability of hflip is self.hflip_p and determined outside albmentations
                transform_hflip_lst.append(A.HorizontalFlip(p=1.0))

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(
                    A.ShiftScaleRotate(rotate_limit=10, shift_limit=0.2, scale_limit=(-0.2, 0.2),
                                       border_mode=cv2.BORDER_CONSTANT, p=0.5))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=2, shift_limit=0.2, scale_limit=(-0.2, 0.2),
                                                                   border_mode=cv2.BORDER_CONSTANT, p=0.5))

        self.transform_crop_person = A.Compose(
            transform_crop_person_lst,
            additional_targets={"agn": "image",
                                "agn_mask": "image",
                                "cloth_mask_warped": "image",
                                "cloth_warped": "image",
                                "image_densepose": "image",
                                "image_parse": "image",
                                "gt_cloth_warped_mask": "image",
                                "image_densepose_hybvton": "mask",
                                "hybvton_warped_cloth": "image",
                                "hybvton_warped_mask": "image",
                                },
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False,),
            seed=seed
        )
        self.transform_crop_cloth = A.Compose(
            transform_crop_cloth_lst,
            additional_targets={"cloth_mask":"image"},
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, ),
            seed=seed
        )

        self.transform_size = A.Compose(
            transform_size_lst,
            additional_targets={"agn": "image",
                                "agn_mask": "image",
                                "cloth": "image",
                                "cloth_mask": "image",
                                "cloth_mask_warped": "image",
                                "cloth_warped": "image",
                                "image_densepose": "image",
                                "image_parse": "image",
                                "gt_cloth_warped_mask": "image",
                                "image_densepose_hybvton": "mask",
                                },
            seed=seed
        )
        self.transform_hflip = A.Compose(
            transform_hflip_lst,
            additional_targets={"agn": "image",
                                "agn_mask": "image",
                                "cloth": "image",
                                "cloth_mask": "image",
                                "cloth_mask_warped": "image",
                                "cloth_warped": "image",
                                "image_densepose": "image",
                                "image_parse": "image",
                                "gt_cloth_warped_mask": "image",
                                "image_densepose_hybvton": "mask",
                                "hybvton_warped_cloth": "image",
                                "hybvton_warped_mask": "image",
                                "keypoints_cloth": "keypoints",
                                },
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, ),
            seed=seed
        )
        #### spatial aug <<<<

        #### non-spatial aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5, 5, 5, p=0.5))
                elif t == "bright_contrast":
                    transform_color_lst.append(
                        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={"agn": "image",
                                    "cloth": "image",
                                    "cloth_warped": "image",
                                    },
                seed=seed
            )
        #### non-spatial aug <<<<

        assert not (self.phase == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"

        im_names = []
        c_names = []
        with open(opj(self.drd, f"yahavton_{self.phase}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                if osp.splitext(im_name)[0] in exclude_image_ids:
                    continue
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names

        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)

    def get_item_from_cname(self, cloth_name):
        if cloth_name not in self.c_names[self.pair_key]:
            raise ValueError(f"Cloth name {cloth_name} not found in dataset")
        idx = self.c_names[self.pair_key].index(cloth_name)
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        if self.transform_size is None and self.transform_color is None:
            raise NotImplementedError("Never reached by original code")
            agn = imread(
                opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]),
                self.img_H,
                self.img_W
            )
            agn_mask = imread(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")),
                self.img_H,
                self.img_W,
                is_mask=True,
            )
            cloth = imread(
                opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]),
                self.img_H,
                self.img_W
            )

            gt_cloth_warped_mask = imread(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]),
                self.img_H,
                self.img_W,
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)

            yahavton_warped_mask = imread(
                opj(self.drd, self.data_type, "yahavton_warped_mask_" + self.pair_key,
                    self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].replace(".jpg", ".png")),
                self.img_H,
                self.img_W,
                is_mask=True
            )

            image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
            image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H,
                                     self.img_W)
            yahavton_warped_mask = (yahavton_warped_mask / 255 * agn_mask / 255)
            agn = agn * (1 - yahavton_warped_mask[:, :, None]) + yahavton_warped_cloth * yahavton_warped_mask[:, :, None]
            agn = agn.astype(np.uint8)
            yahavton_warped_mask = (yahavton_warped_mask * 255).astype(np.uint8)
            agn_mask_orig = 255 - agn_mask
            agn_mask = 255 - agn_mask

        else:
            agn = imread_for_albu(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
            agn_mask = imread_for_albu(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")),
                is_mask=True)
            cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
            cloth_mask = imread_for_albu(opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]),
                                         is_mask=True, cloth_mask_check=True)

            gt_cloth_warped_mask = imread_for_albu(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]),
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)

            yahavton_warp_flow = None
            sift_matches = []
            yahavton_warped_mask = np.ones((self.img_H, self.img_W), dtype=np.uint8) * 255
            sift_keypoints_person = sift_keypoints_cloth = np.array([])
            if self.use_sift_matching:
                with open(opj(self.drd, self.data_type, "sift_matching",
                              self.im_names[idx].replace(".jpg", "_sift_matches.json"))) as f:
                    sift_matches = json.load(f)
                sift_matches = np.array(sift_matches).round()
                if len(sift_matches) > 0:
                    sift_keypoints_person = sift_matches[:, 0, :]
                    sift_keypoints_cloth = sift_matches[:, 1, :]
            if self.use_explicit_warping:
                yahavton_warped_mask = imread_for_albu(
                    opj(self.drd, self.data_type, "yahavton_warped_mask_" + self.pair_key,
                        self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].split(".")[0] + "_torso_mask.png"),
                    is_mask=True
                )
                yahavton_warp_flow = torch.load(
                    opj(self.drd, self.data_type, "yahavton_warp_flow_" + self.pair_key,
                        self.im_names[idx].split(".")[0] + "_" + self.c_names[self.pair_key][idx].split(".")[0] + "_torso_flow.pt"),
                    map_location=torch.device("cpu")
                ).numpy()[0]

            image = imread_for_albu(opj(self.drd, self.data_type, "image", self.im_names[idx]))
            image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))
            image_densepose_hybvton = None
            if self.use_hybvton_densepose_torso:
                image_densepose_hybvton = imread_for_albu(
                    opj(self.drd, self.data_type, "image-densepose_hybvton",
                        self.im_names[idx].replace(".jpg", ".png")))

            if self.transform_size is not None:
                kwargs_for_transform = dict(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    cloth=cloth,
                    cloth_mask=cloth_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    image_densepose_hybvton=image_densepose_hybvton
                )
                if self.use_hybvton_densepose_torso:
                    kwargs_for_transform["image_densepose_hybvton"] = image_densepose_hybvton
                transformed = self.transform_size(**kwargs_for_transform)
                image = transformed["image"]
                agn = transformed["agn"]
                agn_mask = transformed["agn_mask"]
                image_densepose = transformed["image_densepose"]
                gt_cloth_warped_mask = transformed["gt_cloth_warped_mask"]
                if self.use_hybvton_densepose_torso:
                    image_densepose_hybvton = transformed["image_densepose_hybvton"]

                cloth = transformed["cloth"]
                cloth_mask = transformed["cloth_mask"]
            if yahavton_warped_mask is not None:
                yahavton_warped_mask = (yahavton_warped_mask / 255 * agn_mask / 255)
            agn_mask_orig = 255 - agn_mask
            agn_orig = agn

            if self.torso_extraction_method != "none":
                densepose_for_torso = image_densepose_hybvton if self.use_hybvton_densepose_torso else image_densepose
                if self.torso_extraction_method == "torso_segment":
                    densepose_torso_mask = densepose_for_torso[:, :, 0] == DENSEPOSE_SEGM_RGB_TORSO[0]
                    densepose_torso_mask = densepose_torso_mask.astype(np.float32)
                    yahavton_warped_mask = yahavton_warped_mask * densepose_torso_mask
                elif self.torso_extraction_method == "arm_elimination":
                    arm_mask = densepose_to_armmask(densepose_for_torso).astype(np.float32)
                    yahavton_warped_mask = yahavton_warped_mask * (1 - arm_mask)

            agn = agn.astype(np.uint8)
            if yahavton_warped_mask is not None:
                yahavton_warped_mask = (yahavton_warped_mask * 255).astype(np.uint8)

            if self.transform_hflip is not None and random.random() < self.hflip_p:
                if self.use_explicit_warping:
                    transformed = self.transform_hflip(
                        image=image,
                        agn=agn,
                        agn_mask=agn_mask,
                        cloth=cloth,
                        cloth_mask=cloth_mask,
                        image_densepose=image_densepose,
                        gt_cloth_warped_mask=gt_cloth_warped_mask,
                        yahavton_warped_mask=yahavton_warped_mask,
                        yahavton_warp_flow=yahavton_warp_flow,
                        image_densepose_hybvton=image_densepose_hybvton,
                        keypoints=sift_keypoints_person,
                        keypoints_cloth=sift_keypoints_cloth
                    )
                    yahavton_warped_mask = transformed["yahavton_warped_mask"]
                    yahavton_warp_flow = transformed["yahavton_warp_flow"]
                    yahavton_warp_flow[:, :, 0] = -yahavton_warp_flow[:, :, 0]  # flip x
                else:
                    transformed = self.transform_hflip(
                        image=image,
                        agn=agn,
                        agn_mask=agn_mask,
                        cloth=cloth,
                        cloth_mask=cloth_mask,
                        image_densepose=image_densepose,
                        gt_cloth_warped_mask=gt_cloth_warped_mask,
                        image_densepose_hybvton=image_densepose_hybvton,
                        keypoints=sift_keypoints_person,
                        keypoints_cloth=sift_keypoints_cloth
                    )

                image = transformed["image"]
                agn = transformed["agn"]
                agn_mask = transformed["agn_mask"]
                image_densepose = transformed["image_densepose"]
                gt_cloth_warped_mask = transformed["gt_cloth_warped_mask"]
                image_densepose_hybvton = transformed["image_densepose_hybvton"]

                cloth = transformed["cloth"]
                cloth_mask = transformed["cloth_mask"]
                sift_keypoints_person = transformed["keypoints"]
                sift_keypoints_cloth = transformed["keypoints_cloth"]

            if self.transform_crop_person is not None and len(self.transform_crop_person.transforms) != 0:
                transformed_image = self.transform_crop_person(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    yahavton_warped_mask=yahavton_warped_mask,
                    image_densepose_hybvton=image_densepose_hybvton,
                    keypoints=sift_keypoints_person
                )

                image = transformed_image["image"]
                agn = transformed_image["agn"]
                agn_mask = transformed_image["agn_mask"]
                image_densepose = transformed_image["image_densepose"]
                gt_cloth_warped_mask = transformed_image["gt_cloth_warped_mask"]
                yahavton_warped_mask = transformed_image["yahavton_warped_mask"]
                image_densepose_hybvton = transformed_image["image_densepose_hybvton"]
                sift_keypoints_person = transformed_image["keypoints"]

            if self.transform_crop_cloth is not None and len(self.transform_crop_cloth.transforms) != 0:
                transformed_cloth = self.transform_crop_cloth(
                    image=cloth,
                    cloth_mask=cloth_mask,
                    keypoints=sift_keypoints_cloth
                )

                cloth = transformed_cloth["image"]
                cloth_mask = transformed_cloth["cloth_mask"]
                sift_keypoints_cloth = transformed_cloth["keypoints"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=image,
                    agn=agn,
                    cloth=cloth,
                )

                image = transformed["image"]
                agn = transformed["agn"]
                cloth = transformed["cloth"]

                agn = agn * agn_mask[:, :, None].astype(np.float32) / 255.0 + 128 * (
                            1 - agn_mask[:, :, None].astype(np.float32) / 255.0)

            histogram_output = [torch.zeros(ls[0], ls[1], ls[0], ls[1], dtype=torch.float32) for ls in LATENT_SIZES]
            hist_masks = [torch.zeros(ls[0], ls[1], dtype=torch.float32) for ls in LATENT_SIZES]
            gaussian_sigma_scale = gss = 1/2**0.5
            if self.use_sift_matching and len(sift_matches) > 0:
                sift_matches = np.stack([sift_keypoints_person, sift_keypoints_cloth], axis=1)
                def filter_match_out_of_image(m):
                    loc1_truth = (m[0, 0] >= 0) and (m[0, 0] < self.img_W) and (m[0, 1] >= 0) and (m[0, 1] < self.img_H)
                    loc2_truth = (m[1, 0] >= 0) and (m[1, 0] < self.img_W) and (m[1, 1] >= 0) and (m[1, 1] < self.img_H)
                    return loc1_truth and loc2_truth
                sift_matches = np.array([m for m in sift_matches if filter_match_out_of_image(m)])
                # SIFT matching
                image_size = (512, 384)
                for l_idx, latent_size in enumerate(LATENT_SIZES):
                    scale_factor = image_size[0] / latent_size[0]
                    latent_bins = [[None] * latent_size[1] for _ in range(latent_size[0])]
                    latent_hist = [[0] * latent_size[1] for _ in range(latent_size[0])]
                    for loc1, loc2 in sift_matches:
                        loc1 = np.floor(loc1 / scale_factor).astype(np.int32)
                        loc1 = transpose_pt(loc1)
                        if latent_bins[loc1[0]][loc1[1]] is None:
                            latent_bins[loc1[0]][loc1[1]] = []
                        latent_bins[loc1[0]][loc1[1]].append(transpose_pt(loc2))
                        latent_hist[loc1[0]][loc1[1]] = latent_hist[loc1[0]][loc1[1]] + 1
                    latent_hist = np.array(latent_hist)
                    histograms = []
                    for i in range(latent_size[0]):
                        for j in range(latent_size[1]):
                            if latent_hist[i][j] == 0:
                                histograms.append(np.zeros((latent_size[0], latent_size[1])))
                                continue
                            hist, edges = np.histogramdd(
                                np.array(latent_bins[i][j]),
                                bins=latent_size,
                                range=[(0, image_size[0]), (0, image_size[1])],
                                density=False
                            )
                            if self.hist_blured:
                                hist = gaussian_filter(hist, sigma=gss**(2 - l_idx))
                            if self.hist_only_max:
                                m_hist = hist.max()
                                hist = (hist == m_hist).astype(np.float32)
                            if self.hist_to_density:
                                hist = hist / hist.sum()
                            histograms.append(hist)
                    histograms = np.array(histograms)
                    histograms = histograms.reshape(latent_size[0], latent_size[1], latent_size[0], latent_size[1])
                    histograms = torch.from_numpy(histograms).float()
                    histogram_output[l_idx] = histograms
                    hist_masks[l_idx] = (histograms.sum(dim=(-1, -2)) > 0).float()
            elif self.use_explicit_warping:
                yahavton_warp_flow = torch.from_numpy(yahavton_warp_flow).float()
                for l_idx, msize in enumerate(LATENT_SIZES):
                    torso_flow_reshaped = yahavton_warp_flow.reshape(msize[0], self.img_H // msize[0],
                                                                     msize[1], self.img_W // msize[1], 2)
                    torso_flow_reshaped = torso_flow_reshaped.permute(0, 2, 1, 3, 4)
                    torso_flow_reshaped = torso_flow_reshaped.clip(-1, 1)
                    histograms = []
                    hsize = torso_flow_reshaped.size(0)
                    wsize = torso_flow_reshaped.size(1)
                    for i in range(hsize):
                        for j in range(wsize):
                            # Estimated flow specifies a location with (x, y) i.e. (horizontal, vertical)
                            histogram_obj = torch.histogramdd(torso_flow_reshaped[i, j],
                                                              bins=[msize[1], msize[0]],
                                                              range=[-1.0, 1.0, -1.0, 1.0],
                                                              density=False)
                            hist = histogram_obj.hist.transpose(-2, -1)
                            if self.hist_blured:
                                hist = gaussian_filter(hist, sigma=gss**(2 - l_idx))
                            if self.hist_only_max:
                                m_hist = hist.max()
                                hist = (hist == m_hist).to(torch.float32)
                            if self.hist_to_density:
                                hist = hist / hist.sum()
                            histograms.append(hist)
                    histograms = torch.stack(histograms).reshape(hsize, wsize, msize[0], msize[1])
                    histogram_output[l_idx] = histograms

            densepose_for_torso = image_densepose_hybvton if self.use_hybvton_densepose_torso else image_densepose
            densepose_torso_mask = densepose_for_torso[:, :, 0] == DENSEPOSE_SEGM_RGB_TORSO[0]
            densepose_torso_mask = densepose_torso_mask.astype(np.uint8) * 255
            densepose_torso_mask = norm_for_albu(densepose_torso_mask, is_mask=True)
            agn = norm_for_albu(agn)
            agn_orig = norm_for_albu(agn_orig)
            agn_mask_orig = norm_for_albu(agn_mask_orig, is_mask=True)
            # agn_mask_orig = (agn_mask_orig > 0.5).astype(np.float32)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            densepose_torso_mask = densepose_torso_mask * (1 - agn_mask)
            cloth = norm_for_albu(cloth)
            cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
            image = norm_for_albu(image)
            image_densepose = norm_for_albu(image_densepose)
            gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)
            if self.use_explicit_warping:
                yahavton_warped_mask = norm_for_albu(yahavton_warped_mask, is_mask=True)

        return dict(
            agn=agn,
            agn_orig=agn_orig,
            agn_mask=agn_mask,
            agn_mask_orig=agn_mask_orig,
            cloth=cloth,
            cloth_mask=cloth_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
            yahavton_warped_mask=yahavton_warped_mask,
            hist16=histogram_output[0],
            hist32=histogram_output[1],
            hist64=histogram_output[2],
            hist16_mask=hist_masks[0],
            hist32_mask=hist_masks[1],
            hist64_mask=hist_masks[2],
            densepose_torso_mask=densepose_torso_mask,
        )