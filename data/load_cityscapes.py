"""Cityscapes preparation for human-only semantic segmentation.

This script prepares Cityscapes into five K-Fold buckets and writes
images and binary masks into a folder structure suitable for training.

Input (assumed to exist):
	data/cityscapes/
		Cityscape Dataset/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
		Fine Annotations/gtFine/{train,val}/<city>/*_gtFine_labelIds.png

Output (created by this script):
	data/cityscapes/new_data/
		fold_0/image, fold_0/mask
		fold_1/image, fold_1/mask
		...
		fold_4/image, fold_4/mask

Masks are converted to binary semantic masks with values {0, 1} for
Cityscapes classes: person (24) and rider (25).

All images and masks are center-cropped to 512x1024; if center-cropping
is not possible, the script falls back to resizing (masks use nearest-neighbor).

Run from repo root:
	python data/load_cityscapes.py
"""

from __future__ import annotations

import os
import random
from typing import Iterable

import cv2
import numpy as np
from sklearn.model_selection import KFold


TARGET_HEIGHT = 512
TARGET_WIDTH = 1024
N_SPLITS = 5
DEFAULT_SEED = 42

PERSON_ID = 24
RIDER_ID = 25


def create_dir(path: str) -> None:
	"""Create a directory if it does not exist.

	Args:
		path: Directory path.
	"""
	os.makedirs(path, exist_ok=True)


def is_dir_nonempty(path: str) -> bool:
	"""Return True if a directory exists and contains at least one file."""
	if not os.path.isdir(path):
		return False

	for _root, _dirs, files in os.walk(path):
		if files:
			return True
	return False


def center_crop_or_resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
	"""Center-crop an image to (height, width), fallback to resize if crop not possible."""
	h, w = image.shape[:2]
	if h >= height and w >= width:
		y0 = (h - height) // 2
		x0 = (w - width) // 2
		return image[y0 : y0 + height, x0 : x0 + width]

	return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def center_crop_or_resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
	"""Center-crop a mask to (height, width), fallback to resize if crop not possible.

	Resizing uses nearest-neighbor to preserve discrete labels.
	"""
	h, w = mask.shape[:2]
	if h >= height and w >= width:
		y0 = (h - height) // 2
		x0 = (w - width) // 2
		return mask[y0 : y0 + height, x0 : x0 + width]

	return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def iter_cityscapes_image_paths(left_root: str) -> list[str]:
	"""Return a list of Cityscapes image paths under leftImg8bit/train+val.

	Args:
		left_root: Path to leftImg8bit root.
	"""
	if not os.path.isdir(left_root):
		raise FileNotFoundError(f"leftImg8bit directory not found: {left_root}")

	image_paths: list[str] = []
	for root, _dirs, files in os.walk(left_root):
		rel_root = os.path.relpath(root, left_root)
		parts = rel_root.split(os.sep)
		if not parts:
			continue
		split_name = parts[0]
		if split_name not in {"train", "val"}:
			continue

		for filename in files:
			if not filename.endswith("_leftImg8bit.png"):
				continue
			image_paths.append(os.path.join(root, filename))

	if not image_paths:
		raise RuntimeError("No Cityscapes images found under leftImg8bit/train or val.")

	return sorted(image_paths)


def image_to_mask_path(image_path: str, left_root: str, gt_root: str) -> str:
	"""Map a Cityscapes image path to its labelIds mask path.

	Args:
		image_path: Path to an image under leftImg8bit.
		left_root: Path to leftImg8bit root.
		gt_root: Path to gtFine root.

	Returns:
		Expected mask path under gtFine.
	"""
	rel_path = os.path.relpath(image_path, left_root)
	rel_dir = os.path.dirname(rel_path)
	filename = os.path.basename(image_path)

	if not filename.endswith("_leftImg8bit.png"):
		raise ValueError(f"Unexpected Cityscapes image filename: {filename}")

	mask_filename = filename.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
	return os.path.join(gt_root, rel_dir, mask_filename)


def list_image_mask_pairs(left_root: str, gt_root: str) -> list[tuple[str, str, str]]:
	"""List paired (image, mask) samples in Cityscapes.

	Args:
		left_root: Path to leftImg8bit root.
		gt_root: Path to gtFine root.

	Returns:
		List of tuples (base_name, image_path, mask_path).

	Raises:
		RuntimeError: If no valid pairs are found.
	"""
	if not os.path.isdir(gt_root):
		raise FileNotFoundError(f"gtFine directory not found: {gt_root}")

	image_paths = iter_cityscapes_image_paths(left_root)
	pairs: list[tuple[str, str, str]] = []
	missing_masks: list[str] = []

	for image_path in image_paths:
		mask_path = image_to_mask_path(image_path, left_root, gt_root)
		if not os.path.exists(mask_path):
			missing_masks.append(image_path)
			continue

		filename = os.path.basename(image_path)
		base_name = filename.replace("_leftImg8bit.png", "")
		pairs.append((base_name, image_path, mask_path))

	if missing_masks:
		print(f"Warning: {len(missing_masks)} images without a matching mask were skipped.")

	if not pairs:
		raise RuntimeError("No (image, mask) pairs found. Check dataset structure.")

	return pairs


def make_human_binary_mask(label_ids: np.ndarray) -> np.ndarray:
	"""Convert labelIds to a binary mask for person/rider only.

	Args:
		label_ids: Cityscapes labelIds mask (H, W), integer values.

	Returns:
		Binary uint8 mask with values {0, 1}.
	"""
	mask = (label_ids == PERSON_ID) | (label_ids == RIDER_ID)
	return mask.astype(np.uint8)


def write_fold(
	fold_name: str,
	pairs: Iterable[tuple[str, str, str]],
	out_images_dir: str,
	out_masks_dir: str,
) -> int:
	"""Write a fold to disk.

	Args:
		fold_name: Human-readable fold label for logging.
		pairs: Iterable of (base_name, image_path, mask_path).
		out_images_dir: Output directory for images.
		out_masks_dir: Output directory for masks.

	Returns:
		Number of samples written.
	"""
	create_dir(out_images_dir)
	create_dir(out_masks_dir)

	written = 0
	for base, image_path, mask_path in pairs:
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		if image is None:
			raise RuntimeError(f"Failed to read image: {image_path}")

		label_ids = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		if label_ids is None:
			raise RuntimeError(f"Failed to read mask: {mask_path}")

		mask_bin = make_human_binary_mask(label_ids)

		image_out = center_crop_or_resize_image(image, TARGET_HEIGHT, TARGET_WIDTH)
		mask_out = center_crop_or_resize_mask(mask_bin, TARGET_HEIGHT, TARGET_WIDTH)
		mask_out = (mask_out > 0).astype(np.uint8)

		if image_out.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
			raise AssertionError(f"Image size mismatch for {image_path}: {image_out.shape}")
		if mask_out.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
			raise AssertionError(f"Mask size mismatch for {mask_path}: {mask_out.shape}")

		out_image_path = os.path.join(out_images_dir, f"{base}.png")
		out_mask_path = os.path.join(out_masks_dir, f"{base}.png")

		ok_img = cv2.imwrite(out_image_path, image_out)
		ok_msk = cv2.imwrite(out_mask_path, mask_out)
		if not ok_img:
			raise RuntimeError(f"Failed to write image: {out_image_path}")
		if not ok_msk:
			raise RuntimeError(f"Failed to write mask: {out_mask_path}")

		written += 1

	print(f"{fold_name}: wrote {written} samples")
	return written


def assert_split_counts_equal(images_dir: str, masks_dir: str) -> None:
	"""Assert that image and mask counts match for a split."""
	image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
	mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]
	if len(image_files) != len(mask_files):
		raise AssertionError(
			f"Mismatched counts in {os.path.dirname(images_dir)}: "
			f"{len(image_files)} images vs {len(mask_files)} masks"
		)


def spot_check_masks(masks_dir: str) -> None:
	"""Spot-check one mask file for size and binary values."""
	mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]
	if not mask_files:
		return

	sample_path = os.path.join(masks_dir, mask_files[0])
	sample_mask = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
	if sample_mask is None:
		raise RuntimeError(f"Failed to read written mask: {sample_path}")

	if sample_mask.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
		raise AssertionError(f"Written mask has wrong size: {sample_path}")

	unique_vals = set(np.unique(sample_mask).tolist())
	if not unique_vals.issubset({0, 1}):
		raise AssertionError(f"Output mask is not binary {unique_vals} in {sample_path}")


def main() -> int:
	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

	dataset_root = os.path.join(project_root, "data", "cityscapes")
	left_root = os.path.join(dataset_root, "Cityscape Dataset", "leftImg8bit")
	gt_root = os.path.join(dataset_root, "Fine Annotations", "gtFine")
	output_root = os.path.join(dataset_root, "new_data")

	if is_dir_nonempty(output_root):
		print(
			f"Refusing to continue: output directory already exists and is not empty: {output_root}\n"
			"Delete/move it first if you want to regenerate the dataset splits."
		)
		return 1

	pairs = list_image_mask_pairs(left_root, gt_root)
	print(f"Found {len(pairs)} paired samples")

	rng = random.Random(DEFAULT_SEED)
	rng.shuffle(pairs)

	kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=DEFAULT_SEED)

	for fold_idx, (_train_idx, test_idx) in enumerate(kfold.split(pairs)): # type: ignore
		fold_name = f"Fold {fold_idx}"
		fold_pairs = [pairs[i] for i in test_idx]

		fold_root = os.path.join(output_root, f"fold_{fold_idx}")
		fold_images_dir = os.path.join(fold_root, "image")
		fold_masks_dir = os.path.join(fold_root, "mask")

		write_fold(fold_name, fold_pairs, fold_images_dir, fold_masks_dir)
		assert_split_counts_equal(fold_images_dir, fold_masks_dir)
		spot_check_masks(fold_masks_dir)

	print("Done.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
