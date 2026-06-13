import argparse
import os
import cv2
import numpy as np
import PIL
import os.path as osp
import json
from tqdm import tqdm

from dataset import gpvton_to_torso_mask, densepose_to_torsomask, densepose_to_leftarmmask, \
    gpvton_to_left_sleeve_mask, densepose_to_rightarmmask, gpvton_to_right_sleeve_mask, densepose_to_armmask
from utils import sift_match, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--data_type", type=str, choices=["train", "test"])
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--verbose", action="store_true", help="If true, print progress info.")
    parser.add_argument("--lowe_ratio", type=float, default=0.80, help="Lowe's ratio for SIFT matching.")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--show_first_n", type=int, default=5, help="Show first n images for debugging.")
    parser.add_argument("--cloth_segmentation_base_dir", type=str, default=None, help="Directory of cloth segmentation masks.")
    parser.add_argument("--seed", type=int, default=1235, help="Random seed.")
    parser.add_argument("--target", type=str, default="all", choices=["all", "torso", "left", "right"], help="Target part to match.")
    return parser.parse_args()

def main(args):
    save_dir = osp.join(args.save_dir, args.data_type)
    os.makedirs(save_dir, exist_ok=True)
    image_dir = osp.join(args.data_root_dir, args.data_type, "image")
    cloth_dir = osp.join(args.data_root_dir, args.data_type, "cloth")
    segm_dir = osp.join(args.data_root_dir, args.data_type, "image-densepose_siftvton")
    ag_mask_dir = osp.join(args.data_root_dir, args.data_type, "agnostic-mask")
    image_name_list = os.listdir(image_dir)
    print(f"Found {len(image_name_list)} images in {args.data_type} set.")
    # filter by extension
    image_name_list = [name for name in image_name_list if name.endswith(".jpg")]
    print(f"After filtering, {len(image_name_list)} images are used.")
    print(f"Example image names: {image_name_list[:5]}")
    if args.cloth_segmentation_base_dir is not None:
        gpvton_seg_dir = osp.join(args.cloth_segmentation_base_dir, args.data_type, "cloth_parse-bytedance")
        print(f"Using cloth segmentation masks from {gpvton_seg_dir}.")
    else:
        gpvton_seg_dir = None
    target_parts = ["torso", "right", "left"] if args.target == "all" else [args.target]

    for idx, image_fname in tqdm(enumerate(image_name_list), disable=args.disable_tqdm):
        img1_path = osp.join(image_dir, image_fname)
        img2_path = osp.join(cloth_dir, image_fname)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = cv2.resize(img1, (args.img_W, args.img_H))
        img2 = cv2.resize(img2, (args.img_W, args.img_H))

        # densepose segm
        segm = PIL.Image.open(osp.join(segm_dir, image_fname.replace(".jpg", ".png")))
        segm = segm.resize((args.img_W, args.img_H), PIL.Image.NEAREST)
        segm_np = np.array(segm)

        # gpvton segm
        if args.cloth_segmentation_base_dir is not None:
            gpvton_seg_path = osp.join(gpvton_seg_dir, image_fname.replace(".jpg", ".png"))
            gpvton_seg = cv2.imread(gpvton_seg_path)
            gpvton_seg = cv2.cvtColor(gpvton_seg, cv2.COLOR_BGR2RGB)
        else:
            # temporary gpvton segm
            gpvton_seg = np.ones_like(img1)

        # impainting mask
        ag_mask = cv2.imread(osp.join(ag_mask_dir, image_fname.replace(".jpg", "_mask.png")))
        ag_mask = cv2.resize(ag_mask, (args.img_W, args.img_H))
        ag_mask = cv2.cvtColor(ag_mask, cv2.COLOR_BGR2GRAY)
        ag_mask_bool = ag_mask > 128
        img1_org = img1
        img2_org = img2
        matche_loc_list_all = []
        for target_part in target_parts:
            match target_part:
                case "torso":
                    segm_torso_np = densepose_to_torsomask(segm_np)
                    segm_mask_np = segm_torso_np
                    segm_cloth_mask_np = gpvton_to_torso_mask(gpvton_seg)
                case "left":
                    segm_leftarm_np = densepose_to_leftarmmask(segm_np)
                    segm_mask_np = segm_leftarm_np
                    segm_cloth_mask_np = gpvton_to_left_sleeve_mask(gpvton_seg)
                case "right":
                    segm_rightarm_np = densepose_to_rightarmmask(segm_np)
                    segm_mask_np = segm_rightarm_np
                    segm_cloth_mask_np = gpvton_to_right_sleeve_mask(gpvton_seg)
                case "all":
                    segm_torso_np = segm_np[:, :, 0] == 20
                    segm_arm_np = densepose_to_armmask(segm_np)
                    segm_mask_np = np.logical_or(segm_torso_np, segm_arm_np)
                    segm_cloth_mask_np = np.ones_like(segm_mask_np)
                case _:
                    raise ValueError(f"target_part {target_part} not supported")
            img1 = np.zeros_like(img1)
            and_mask = segm_mask_np[:, :] & ag_mask_bool[:, :]
            img1[and_mask] = img1_org[and_mask]
            img1 = img1.astype(np.uint8)
            img2 = np.ones_like(img2) * 255
            img2[segm_cloth_mask_np] = img2_org[segm_cloth_mask_np]
            img2 = img2.astype(np.uint8)

            matches, keypoints1, keypoints2 = sift_match(img1, img2, lowe_ratio=args.lowe_ratio, verbose=args.verbose)
            matche_loc_list = [[keypoints1[m.queryIdx].pt, keypoints2[m.trainIdx].pt] for m in matches]
            matche_loc_list_all += matche_loc_list
            if idx < args.show_first_n:
                matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                matched_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(osp.join(args.save_dir,
                                     image_fname.replace(".jpg", f"_{target_part}_sift_matches.jpg")),
                            matched_image)
        with open(osp.join(save_dir, image_fname.replace(".jpg", "_sift_matches.json")), "w") as f:
            json.dump(matche_loc_list_all, f)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)