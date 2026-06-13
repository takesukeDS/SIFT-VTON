import json
import argparse
import random

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
import cv2
from scipy.spatial import KDTree
from functools import partial

def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
def load_args(from_path, is_test=True):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(from_path, "r") as f:
        args.__dict__ = json.load(f)
    args.is_test = is_test
    if "E_name" not in args.__dict__.keys():
        args.E_name = "basic"
    return args   
def tensor2img(x, round=False):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    # x = (x+1)/2
    # x = np.clip(x, 0, 1)
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    adjust = 0.5 if round else 0
    x = np.uint8(x*255.0 + adjust)
    if x.shape[-1] == 1:  # gray sclae
        x = np.concatenate([x,x,x], axis=-1)
    return x
def resize_mask(m, shape):
    m = F.interpolate(m, shape)
    m[m > 0.5] = 1
    m[m < 0.5] = 0
    return m

def remove_overlap(seg_out, warped_cm, inference=False):
    assert len(warped_cm.shape) == 4
    overlapped_region = (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True)
    if inference:
        overlapped_region = (overlapped_region > 0.5).float()
    warped_cm = warped_cm - overlapped_region * warped_cm
    return warped_cm

def bilateral_filter(image, kernel_size, sigma_d, sigma_r):
    """Bilateral filter implementation.
        Args:
            image: input float tensor with shape [bsz, ch, height, width]
            kernel_size: int. we assume it is odd.
            sigma_d: float. standard deviation for distance.
            sigma_r: float or tensor. standard deviation for range.
    """
    padding = (kernel_size - 1) // 2
    # distance
    bsz, ch, height, width = image.shape
    if isinstance(sigma_r, float):
        sigma_r = torch.tensor([sigma_r]).expand(bsz)
    sigma_r = sigma_r.to(image.device)
    height_pad = height + 2 * padding
    width_pad = width + 2 * padding
    # gaussian on spacial distance
    grid_x, grid_y = torch.meshgrid(torch.arange(width_pad), torch.arange(height_pad),
                                    indexing='xy')
    grid_x = grid_x.float().to(image.device)
    grid_y = grid_y.float().to(image.device)
    unfold_grid = nn.Unfold(kernel_size=kernel_size)
    grid_x_unfolded = unfold_grid(grid_x[None, None])
    grid_y_unfolded = unfold_grid(grid_y[None, None])
    grid_x_unfolded = grid_x_unfolded.transpose(1, 2).reshape(height * width, 1, kernel_size ** 2)
    grid_y_unfolded = grid_y_unfolded.transpose(1, 2).reshape(height * width, 1, kernel_size ** 2)
    center_index = kernel_size ** 2 // 2
    diff_x_unfolded = grid_x_unfolded - grid_x_unfolded[:, :, center_index][:, :, None]
    diff_y_unfolded = grid_y_unfolded - grid_y_unfolded[:, :, center_index][:, :, None]
    dist_unfolded = diff_x_unfolded ** 2 + diff_y_unfolded ** 2
    gaussian_dist = torch.exp(-dist_unfolded / (2 * sigma_d ** 2))

    # gaussian on range
    unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
    image_unfolded = unfold(image)
    image_unfolded = image_unfolded.transpose(1, 2).reshape(bsz, height * width, ch, kernel_size ** 2)
    center_value = image_unfolded[:, :, :, center_index]
    diff_value = image_unfolded - center_value[:, :, :, None]
    dist_value = diff_value ** 2
    gaussian_value = torch.exp(-dist_value / (2 * sigma_r[:, None, None, None] ** 2))

    # bilateral filter
    bilateral_weight = gaussian_dist[None] * gaussian_value
    result_unfolded = torch.sum(bilateral_weight * image_unfolded, dim=-1)
    z_constant = bilateral_weight.sum(dim=-1)
    result_unfolded = result_unfolded / z_constant
    result = result_unfolded.transpose(1, 2).reshape(bsz, ch, height, width)
    return result


def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    # numpy
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



# SIFT-VTON SIFT matching and filtering
ANGLE_THRES = 45
def filter_angle(keypoints1, keypoints2, match):
  idx1 = match.queryIdx
  idx2 = match.trainIdx
  angle1 = keypoints1[idx1].angle
  angle2 = keypoints2[idx2].angle
  diff = abs(angle1 - angle2)
  if diff > 180:
    diff = 360 - diff
  return diff < ANGLE_THRES

def round_tuple(t):
  return tuple(map(lambda x: round(x), t))

def transpose_pt(pt):
  return (pt[1], pt[0])

SCALE_RATIO_THRES = 2.25
def filter_scale(keypoints1, keypoints2, match):
  idx1 = match.queryIdx
  idx2 = match.trainIdx
  size1 = keypoints1[idx1].size
  size2 = keypoints2[idx2].size
  scale_ratio = size1 / size2
  if 1/SCALE_RATIO_THRES <= scale_ratio <= SCALE_RATIO_THRES:
    return True
  return False

# unused
PIXEL_DIST_THRES = 200
def filter_value(keypoints1, keypoints2, img1, img2, match):
  idx1 = match.queryIdx
  idx2 = match.trainIdx
  loc1 = transpose_pt(round_tuple(keypoints1[idx1].pt))
  loc2 = transpose_pt(round_tuple(keypoints2[idx2].pt))
  value1 = img1[loc1].astype(np.int16)
  value2 = img2[loc2].astype(np.int16)
  dist = np.linalg.norm(value1 - value2, ord=1)
  if dist < PIXEL_DIST_THRES:
    return True
  return False

PIXEL_HUE_DIFF_THRES = 15
PIXEL_SAT_EFFECTIVE_RATIO = 0.1
PIXEL_VAL_DIFF_RATIO = 0.4
PIXEL_VAL_EFFECTIVE_RATIO = 0.2
def filter_hsv(keypoints1, keypoints2, img1, img2, match):
  idx1 = match.queryIdx
  idx2 = match.trainIdx
  loc1 = transpose_pt(round_tuple(keypoints1[idx1].pt))
  loc2 = transpose_pt(round_tuple(keypoints2[idx2].pt))
  value1 = img1[loc1]
  value2 = img2[loc2]
  h1, s1, v1 = cv2.cvtColor(value1[None, None], cv2.COLOR_RGB2HSV)[0, 0]
  h2, s2, v2 = cv2.cvtColor(value2[None, None], cv2.COLOR_RGB2HSV)[0, 0]
  v1 = v1.astype(np.int16)
  v2 = v2.astype(np.int16)
  diff = abs(v1 - v2)
  if diff > PIXEL_VAL_DIFF_RATIO * 255:
    return False
  if s1 < 255 * PIXEL_SAT_EFFECTIVE_RATIO and s2 < 255 * PIXEL_SAT_EFFECTIVE_RATIO:
    return True
  if v1 < 255 * PIXEL_VAL_EFFECTIVE_RATIO and v2 < 255 * PIXEL_VAL_EFFECTIVE_RATIO:
    return True

  h1 = h1.astype(np.int16)
  h2 = h2.astype(np.int16)
  diff = abs(h1 - h2)
  if diff > 90:
    diff = 180 - diff
  if diff < PIXEL_HUE_DIFF_THRES:
    return True
  return False

PIXEL_DIST_MULTIPLIER2 = 1.4
NUM_FILTER_NEAREST_ITTER = 1
def filter_nearest_dist_from2(keypoints1, keypoints2, tree, matches, sift_match,
                              num_neighbors=1):
  idx1 = sift_match.queryIdx
  idx2 = sift_match.trainIdx
  pt1 = keypoints1[idx1].pt
  pt2 = keypoints2[idx2].pt

  dist2, ii = tree.query(pt2, k=num_neighbors + 1)

  for i in range(num_neighbors):
    if dist2[i+1] == np.inf:
      continue
    pt1_2 = keypoints1[matches[ii[i+1]].queryIdx].pt
    dist1 = np.linalg.norm(np.array(pt1) - np.array(pt1_2))
    if dist1 > PIXEL_DIST_MULTIPLIER2 * dist2[i+1]:
      return False
  return True

PIXEL_DIST_MULTIPLIER1 = 1.8
def filter_nearest_dist_from1(keypoints1, keypoints2, tree, matches, sift_match, num_neighbors=1):
  idx1 = sift_match.queryIdx
  idx2 = sift_match.trainIdx
  pt1 = keypoints1[idx1].pt
  pt2 = keypoints2[idx2].pt
  dist1, ii = tree.query(pt1, k=num_neighbors + 1)

  for i in range(num_neighbors):
    if dist1[i+1] == np.inf:
      continue
    pt2_2 = keypoints2[matches[ii[i+1]].trainIdx].pt
    dist2 = np.linalg.norm(np.array(pt2) - np.array(pt2_2))
    if dist2 > PIXEL_DIST_MULTIPLIER1 * dist1[i+1]:
      return False
  return True

import math

LOCATION_EPS = 1 * 2**0.5
def remove_duplicate_match(keypoints1, keypoints2, matches):
  matches_filtered = []
  def match_in_filtered(match):
    match_point1 = keypoints1[match.queryIdx].pt
    match_point2 = keypoints2[match.trainIdx].pt
    for filtered_match in matches_filtered:
      cur_point1 = keypoints1[filtered_match.queryIdx].pt
      cur_point2 = keypoints2[filtered_match.trainIdx].pt
      dist1 = np.linalg.norm(np.array(match_point1) - np.array(cur_point1))
      dist2 = np.linalg.norm(np.array(match_point2) - np.array(cur_point2))
      if dist1 == 0.0 or dist2 == 0.0:
          return True
      if dist1 < LOCATION_EPS and dist2 < LOCATION_EPS:
        return True
    return False
  for match in matches:
    if not match_in_filtered(match):
      matches_filtered.append(match)
  return matches_filtered

# SLOPE_STD_MULTIPLIER = 2.58
# SLOPE_STD_MULTIPLIER = 1.96
SLOPE_STD_MULTIPLIER = 1.65
def remove_slope_outliers(keypoints1, keypoints2, matches):
  slopes = []
  for match in matches:
    idx1 = match.queryIdx
    idx2 = match.trainIdx
    pt1 = keypoints1[idx1].pt
    pt2 = keypoints2[idx2].pt
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    slopes.append(slope)
  slope_mean = np.mean(slopes)
  slope_std = np.std(slopes)
  def check_slope(slope):
    return abs(slope - slope_mean) < SLOPE_STD_MULTIPLIER * slope_std
  matches_filtered = [matches[idx] for idx, slope in enumerate(slopes) if check_slope(slope)]
  return matches_filtered

SLOPE_ANGLE_THRES = np.pi / 12
def remove_angle_outliers(keypoints1, keypoints2, matches):
    if len(matches) < 2:
        return matches, None
    slopes = []
    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        pt1 = keypoints1[idx1].pt
        pt2 = keypoints2[idx2].pt
        sl = - (pt2[1] - pt1[1]) / (pt2[0] - pt1[0] + 384)
        slopes.append(sl)
    def check_slope(idx, slope):
        slopes_tmp = slopes.copy()
        slopes_tmp.pop(idx)
        slope_mean_tmp = np.mean(slopes_tmp)
        base_angle_tmp = math.atan(slope_mean_tmp)
        return abs(base_angle_tmp - math.atan(slope)) < SLOPE_ANGLE_THRES
    matches_filtered = [matches[idx] for idx, slope in enumerate(slopes) if check_slope(idx, slope)]
    return matches_filtered, None


class HomographyRANSAC:
    def __init__(self, threshold=2.0, confidence=0.99, max_iterations=1000):
        """
        RANSAC specifically optimized for homography transformations

        Args:
            threshold: Reprojection error threshold in pixels
            confidence: Desired confidence level (0.99 = 99%)
            max_iterations: Maximum number of iterations
        """
        self.threshold = threshold
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.min_samples = 4  # Homography needs minimum 4 point correspondences

    def estimate_homography_transform(self, src_pts, dst_pts):
        """
        Estimate homography transformation from point correspondences

        Homography transformation: [x'] = H [x]
                                   [y']     [y]
                                   [w']     [1]

        where x'/w', y'/w' are the final 2D coordinates

        Args:
            src_pts: Source points (Nx2)
            dst_pts: Destination points (Nx2)

        Returns:
            3x3 homography matrix or None if failed
        """
        if src_pts.shape[0] < 4:
            return None

        n_points = src_pts.shape[0]

        # Build the coefficient matrix A
        # For each point correspondence, we get 2 equations
        A = []

        for i in range(n_points):
            x, y = src_pts[i]
            x_prime, y_prime = dst_pts[i]

            # First equation: x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
            # Rearranged: h11*x + h12*y + h13 - h31*x*x' - h32*y*x' - h33*x' = 0
            A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])

            # Second equation: y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)
            # Rearranged: h21*x + h22*y + h23 - h31*x*y' - h32*y*y' - h33*y' = 0
            A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])

        A = np.array(A)

        try:
            # Solve using SVD (homogeneous least squares)
            # The solution is the last column of V (corresponding to smallest singular value)
            U, S, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)

            # Normalize so that H[2,2] = 1
            H = H / H[2, 2]

            return H

        except np.linalg.LinAlgError:
            return None

    def apply_homography_transform(self, homography_matrix, points):
        """
        Apply homography transformation to points

        Args:
            homography_matrix: 3x3 homography matrix
            points: Nx2 array of points

        Returns:
            Transformed points (Nx2)
        """
        # Convert to homogeneous coordinates
        homogeneous_pts = np.hstack([points, np.ones((points.shape[0], 1))])

        # Apply transformation
        transformed_homogeneous = (homography_matrix @ homogeneous_pts.T).T

        # Convert back from homogeneous coordinates (divide by w)
        # Handle potential division by zero
        w = transformed_homogeneous[:, 2]
        w = np.where(np.abs(w) < 1e-8, 1e-8, w)  # Avoid division by zero

        transformed = transformed_homogeneous[:, :2] / w[:, np.newaxis]

        return transformed

    def compute_reprojection_error(self, homography_matrix, src_pts, dst_pts):
        """
        Compute reprojection errors for homography transformation

        Args:
            homography_matrix: 3x3 homography matrix
            src_pts: Source points
            dst_pts: Destination points

        Returns:
            Array of reprojection errors (distances)
        """
        # Transform source points
        transformed_pts = self.apply_homography_transform(homography_matrix, src_pts)

        # Compute Euclidean distances
        errors = np.linalg.norm(transformed_pts - dst_pts, axis=1)

        return errors

    def adaptive_iterations(self, n_points, current_best_inliers=0):
        """
        Compute adaptive number of iterations based on current best result
        """
        if n_points < self.min_samples:
            return 0

        # Estimate outlier ratio
        if current_best_inliers > 0:
            inlier_ratio = current_best_inliers / n_points
            outlier_ratio = 1 - inlier_ratio
        else:
            outlier_ratio = 0.5  # Conservative initial estimate

        outlier_ratio = max(0.01, min(0.99, outlier_ratio))  # Clamp to reasonable range

        # Probability that a random sample of 4 points are all inliers
        prob_all_inliers = (1 - outlier_ratio) ** self.min_samples

        if prob_all_inliers > 1e-10:
            # Number of iterations to have confidence% chance of success
            num_iter = int(np.log(1 - self.confidence) / np.log(1 - prob_all_inliers))
            return min(num_iter, self.max_iterations)
        else:
            return self.max_iterations

    def eliminate_outliers(self, src_pts, dst_pts, verbose=False):
        """
        Main RANSAC function for eliminating outliers using homography model

        Args:
            src_pts: Source points (Nx2)
            dst_pts: Destination points (Nx2)
            verbose: Print progress information

        Returns:
            inlier_mask: Boolean mask for inlier correspondences
            inlier_indices: Indices of inlier correspondences
            best_homography: Best homography transformation found
            stats: Dictionary with statistics
        """
        n_points = src_pts.shape[0]

        if n_points < self.min_samples:
            empty_mask = np.zeros(n_points, dtype=bool)
            return empty_mask, [], None, {"iterations": 0, "inliers": 0}

        best_inliers = []
        best_homography = None
        max_inliers = 0

        iteration = 0

        if verbose:
            print(f"Starting RANSAC with {n_points} correspondences")
            print(f"Threshold: {self.threshold} pixels")

        while iteration < self.max_iterations:
            # Adaptive termination
            if iteration > 100 and iteration % 100 == 0:
                required_iterations = self.adaptive_iterations(n_points, max_inliers)
                if iteration >= required_iterations:
                    if verbose:
                        print(f"Early termination at iteration {iteration}")
                    break

            # Randomly sample 4 correspondences
            try:
                sample_indices = random.sample(range(n_points), self.min_samples)
            except ValueError:
                break

            sample_src = src_pts[sample_indices]
            sample_dst = dst_pts[sample_indices]

            # Estimate homography transformation
            homography_matrix = self.estimate_homography_transform(sample_src, sample_dst)

            if homography_matrix is None:
                iteration += 1
                continue

            # Compute errors for all points
            try:
                errors = self.compute_reprojection_error(homography_matrix, src_pts, dst_pts)
            except:
                iteration += 1
                continue

            # Find inliers
            inlier_indices = np.where(errors < self.threshold)[0]
            n_inliers = len(inlier_indices)

            # Update best model if current is better
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_inliers = inlier_indices
                best_homography = homography_matrix.copy()

                if verbose and iteration % 500 == 0:
                    print(f"Iteration {iteration}: Found {n_inliers} inliers ({n_inliers/n_points*100:.1f}%)")

            iteration += 1

        # Create inlier mask
        inlier_mask = np.zeros(n_points, dtype=bool)
        if len(best_inliers) > 0:
            inlier_mask[best_inliers] = True

            # Refine the transformation using all inliers
            if len(best_inliers) >= 4:
                refined_homography = self.estimate_homography_transform(
                    src_pts[best_inliers],
                    dst_pts[best_inliers]
                )
                if refined_homography is not None:
                    best_homography = refined_homography

        stats = {
            "iterations": iteration,
            "inliers": len(best_inliers),
            "inlier_ratio": len(best_inliers) / n_points if n_points > 0 else 0,
            "outlier_elimination_rate": 1 - (len(best_inliers) / n_points) if n_points > 0 else 0
        }

        if verbose:
            print(f"Final results after {iteration} iterations:")
            print(f"  Inliers: {len(best_inliers)}/{n_points} ({stats['inlier_ratio']*100:.1f}%)")
            print(f"  Outliers eliminated: {stats['outlier_elimination_rate']*100:.1f}%")

        return inlier_mask, best_inliers, best_homography, stats

    def filter_opencv_matches(self, matches, kp1, kp2, verbose=False):
        """
        Filter OpenCV matches using homography RANSAC

        Args:
            matches: List of cv2.DMatch objects
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            verbose: Print statistics

        Returns:
            good_matches: Filtered matches
            homography_matrix: Estimated homography transformation
            stats: Statistics dictionary
        """
        if len(matches) < self.min_samples:
            return [], None, {"error": "Not enough matches"}

        # Extract point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Apply RANSAC
        inlier_mask, inlier_indices, homography_matrix, stats = self.eliminate_outliers(
            src_pts, dst_pts, verbose=verbose
        )

        # Filter matches based on inliers
        good_matches = [matches[i] for i in inlier_indices]

        return good_matches, homography_matrix, stats

def filter_homography_ransac(src_pts, dst_pts, threshold, confidence=0.99, max_iterations=1000):
    """
    Convenience function to filter correspondences using homography RANSAC

    Args:
        src_pts: Source points (Nx2)
        dst_pts: Destination points (Nx2)
        threshold: Reprojection error threshold

    Returns:
        detected_inliers: Set of inlier indices
    """
    # Initialize homography RANSAC
    homography_ransac = HomographyRANSAC(
        threshold=threshold,
        confidence=confidence,
        max_iterations=max_iterations
    )

    # Eliminate outliers
    inlier_mask, inlier_indices, estimated_homography, stats = homography_ransac.eliminate_outliers(
        src_pts, dst_pts, verbose=False
    )

    # Evaluate results
    detected_inliers = set(inlier_indices)
    return detected_inliers


RANSAC_THRESHOLD = 12.0
def sift_match(img1, img2, lowe_ratio=0.75, cross_check=True, verbose=False):
    """
    Match two images using SIFT and return the matched keypoints.
    :param img1: First image (numpy array).
    :param img2: Second image (numpy array).
    :param ratio: Lowe's ratio test threshold.
    :param cross_check: Whether to use cross-checking. not used if ratio < 1
    :return: list of match objects, and two list of keypoints.
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    if len(keypoints1) < 2 or len(keypoints2) < 2:
        if verbose:
            print("Not enough keypoints found in one of the images.(< 2)")
        return [], keypoints1, keypoints2

    if lowe_ratio < 1:
        # filter matches using lowe's ratio test

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)

        matches = good_matches
        if verbose:
            print(f"Found {len(matches)} matches after Lowe's ratio test")
    else:
        # Create a Brute Force Matcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)
        if verbose:
            print(f"Found {len(matches)} matches after brute-force matching with cross_check={cross_check}")

    matches = sorted(matches, key=lambda x: x.distance)
    # filtering matches
    filtering_functions_1st = [filter_angle, filter_scale]
    for filter_func in filtering_functions_1st:
        matches = list(filter(partial(filter_angle, keypoints1, keypoints2), matches))
        if verbose:
            print(f"Found {len(matches)} matches after {filter_func.__name__}")


    # filtering hue, saturation, value
    matches = list(filter(partial(filter_hsv, keypoints1, keypoints2, img1, img2), matches))
    if verbose:
        print(f"Found {len(matches)} matches after filter_hsv")

    matches = remove_duplicate_match(keypoints1, keypoints2, matches)

    # detecting outliers
    if len(matches) >= 15:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        detected_inliers = filter_homography_ransac(src_pts, dst_pts, threshold=RANSAC_THRESHOLD)
        matches = [matches[i] for i in detected_inliers]
        if verbose:
            print(f"Found {len(matches)} matches after filter_homography_ransac")
    elif len(matches) > 1:
        matches, _ = remove_angle_outliers(keypoints1, keypoints2, matches)
        if verbose:
            print(f"Found {len(matches)} matches after remove_angle_outliers")

        filtering_functions_2nd = [filter_nearest_dist_from2, filter_nearest_dist_from1]
        for filter_func in filtering_functions_2nd:
            for i in range(NUM_FILTER_NEAREST_ITTER):
                if len(matches) < 2:
                    break
                if filter_func == filter_nearest_dist_from2:
                    locations = np.array([keypoints2[m.trainIdx].pt for m in matches])
                else:
                    locations = np.array([keypoints1[m.queryIdx].pt for m in matches])
                tree = KDTree(locations, copy_data=True)
                matches = list(filter(partial(filter_func, keypoints1, keypoints2, tree, matches.copy()),
                                          matches))
                if verbose:
                    print(f"Found {len(matches)} matches after {filter_func.__name__}, itter {i}")
    if len(matches) < 2:
        if verbose:
            print("Not enough matches after filtering.(< 2)")
        return [], keypoints1, keypoints2

    return matches, keypoints1, keypoints2


### Ported from diffusers script 2025.10.28 Takemoto###
def compute_snr(timesteps, alphas_cumprod):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    # Get alphas and sigmas.
    alpha = get_alphas(timesteps, alphas_cumprod)
    sigma = get_sigmas(timesteps, alphas_cumprod)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def get_alphas(timesteps, alphas_cumprod):
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)
    return alpha


def get_sigmas(timesteps, alphas_cumprod):
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
    return sigma
