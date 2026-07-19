"""Evaluation script reproducing Table 1 of the SIFT-VTON paper.

Paired mode  : SSIM + LPIPS between predictions and ground-truth test images.
Unpaired mode: FID + KID (clean-fid) between predictions and test/image.

Implementations and preprocessing mirror the evaluation used for the paper:
- LPIPS: torchmetrics LearnedPerceptualImagePatchSimilarity (AlexNet), inputs in [-1, 1]
- SSIM : torchmetrics StructuralSimilarityIndexMeasure (data_range=1.0), inputs in [0, 1]
- FID/KID: cleanfid.fid.compute_fid / compute_kid, folder vs folder
- Images read with OpenCV, converted to RGB, resized to 384x512 (W x H, bilinear)
"""
import argparse
import csv
import json
import os.path as osp

import cv2
import numpy as np
import torch
from tqdm import tqdm

RESIZE_SIZE = (384, 512)  # (W, H) as passed to cv2.resize


def normalize_from_rgb(image):
    return (image / 127.5) - 1.0


def denormalize_image(image):
    return (image + 1.0) / 2.0


def read_image_by_cv2(image_path, resize_size=RESIZE_SIZE):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = cv2.resize(image, resize_size, interpolation=cv2.INTER_LINEAR)
    return normalize_from_rgb(image)


def to_tensor(image, device):
    return torch.from_numpy(image).permute(2, 0, 1)[None].to(device)


def load_agnostic_mask(data_root_dir, person_id, device, resize_size=RESIZE_SIZE):
    mask_path = osp.join(data_root_dir, "test", "agnostic-mask", f"{person_id}_00_mask.png")
    mask = cv2.imread(mask_path)
    if mask is None:
        raise FileNotFoundError(f"Could not read agnostic mask: {mask_path}")
    mask = cv2.resize(mask, resize_size)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = torch.from_numpy(mask).float() / 255
    return mask[None].to(device)


def evaluate_paired(args, device):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_func = LearnedPerceptualImagePatchSimilarity().to(device)
    ssim_func = StructuralSimilarityIndexMeasure(reduction="sum", data_range=1.0).to(device)

    pair_list_path = args.pair_list or osp.join(args.data_root_dir, "test_pairs.txt")
    with open(pair_list_path) as f:
        pair_rows = list(csv.reader(f, delimiter=" "))

    gt_dir = osp.join(args.data_root_dir, "test", "image")
    lpips_total, ssim_total = 0.0, 0.0
    worst_lpips, worst_lpips_id = 0.0, None
    worst_ssim, worst_ssim_id = 1.0, None

    for fname_person, _fname_cloth in tqdm(pair_rows, desc="paired eval"):
        person_id = fname_person.split("_")[0]
        # Paired evaluation: the person's own cloth, so both ids are person_id.
        pred_fname = args.pred_fname_format.format(person_id=person_id, cloth_id=person_id)
        pred_path = osp.join(args.pred_dir, pred_fname)
        if not osp.exists(pred_path):
            raise FileNotFoundError(
                f"Missing prediction: {pred_path} (from --pred_fname_format "
                f"'{args.pred_fname_format}'). Refusing to skip files: silent "
                f"skips would bias the reported mean.")
        pred = to_tensor(read_image_by_cv2(pred_path), device)
        gt = to_tensor(read_image_by_cv2(osp.join(gt_dir, fname_person)), device)

        if args.restrict_region:
            mask = load_agnostic_mask(args.data_root_dir, person_id, device)
            pred = pred * mask
            gt = gt * mask

        with torch.no_grad():
            lpips_cur = lpips_func(pred, gt).item()
            ssim_cur = ssim_func(denormalize_image(pred), denormalize_image(gt)).item()

        if lpips_cur > worst_lpips:
            worst_lpips, worst_lpips_id = lpips_cur, person_id
        if ssim_cur < worst_ssim:
            worst_ssim, worst_ssim_id = ssim_cur, person_id
        if args.verbose:
            print(f"{person_id}: lpips {lpips_cur:.4f}, ssim {ssim_cur:.4f}")
        lpips_total += lpips_cur
        ssim_total += ssim_cur

    n = len(pair_rows)
    print(f"worst lpips {worst_lpips:.4f} ({worst_lpips_id}), "
          f"worst ssim {worst_ssim:.4f} ({worst_ssim_id})")
    return {"ssim": ssim_total / n, "lpips": lpips_total / n, "num_images": n}


def evaluate_unpaired(args, device):
    from cleanfid import fid as cleanfid_fid
    import glob

    ref_dir = osp.join(args.data_root_dir, "test", "image")
    n_ref = len(glob.glob(osp.join(ref_dir, "*")))
    n_pred = len(glob.glob(osp.join(args.pred_dir, "*")))
    if n_ref != n_pred:
        print(f"WARNING: image count mismatch: {n_ref} reference vs {n_pred} predictions")

    fid_score = cleanfid_fid.compute_fid(ref_dir, args.pred_dir, device=device)
    kid_score = cleanfid_fid.compute_kid(ref_dir, args.pred_dir, device=device)
    return {"fid": fid_score, "kid": kid_score, "kid_x1000": kid_score * 1000,
            "num_reference": n_ref, "num_predictions": n_pred}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root_dir", required=True,
                        help="VITON-HD dataset root (expects test/image, test/agnostic-mask)")
    parser.add_argument("--pred_dir", required=True,
                        help="Directory of generated images (inference output pair/ or unpair/)")
    parser.add_argument("--mode", required=True, choices=["paired", "unpaired"])
    parser.add_argument("--pair_list", default=None,
                        help="Pairs file (default: <data_root_dir>/test_pairs.txt); paired mode only")
    parser.add_argument("--pred_fname_format", default="{person_id}_00_{cloth_id}_00.jpg",
                        help="Prediction filename pattern; paired mode only")
    parser.add_argument("--restrict_region", action="store_true",
                        help="Mask pred and GT with the agnostic mask before SSIM/LPIPS "
                             "(NOT used for Table 1)")
    parser.add_argument("--verbose", action="store_true", help="Print per-image metrics")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("WARNING: CUDA not available, running on CPU (slow)")

    if args.mode == "paired":
        results = evaluate_paired(args, device)
    else:
        results = evaluate_unpaired(args, device)

    results["settings"] = {"mode": args.mode, "pred_dir": args.pred_dir,
                           "restrict_region": args.restrict_region,
                           "resize_wh": list(RESIZE_SIZE)}
    print(json.dumps(results, indent=2))
    out_path = osp.join(osp.dirname(osp.normpath(args.pred_dir)),
                        f"evaluate_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
