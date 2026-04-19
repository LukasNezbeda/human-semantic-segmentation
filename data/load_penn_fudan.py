"""
Prepare the PennFudanPed dataset for semantic segmentation.

Creates 5 K-fold buckets with paired images and binary masks.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class Pair:
	"""Image/mask pair with a shared base name."""

	base: str
	image_path: str
	mask_path: str


INPUT_ROOT_DEFAULT = os.path.join("data", "penn_fudan", "PennFudanPed")
IMAGE_SUBDIR = "PNGImages"
MASK_SUBDIR = "PedMasks"
OUTPUT_ROOT_DEFAULT = os.path.join("data", "penn_fudan", "new_data")
K_SPLITS = 5
RANDOM_STATE = 42
OUTPUT_SIZE = 512


def create_dir(path: str) -> None:
	"""Create a directory if it does not exist."""
	if not os.path.exists(path):
		os.makedirs(path)


def is_non_empty_dir(path: str) -> bool:
	"""Return True if a directory exists and contains any entries."""
	if not os.path.isdir(path):
		return False
	return any(os.scandir(path))


def collect_pairs(input_root: str) -> tuple[list[Pair], list[str]]:
	"""Collect image/mask pairs from the dataset.

	Args:
		input_root: Root directory containing the PennFudanPed dataset.

	Returns:
		A tuple of (pairs, missing_masks), where pairs are sorted by base name.
	"""
	image_dir = os.path.join(input_root, IMAGE_SUBDIR)
	mask_dir = os.path.join(input_root, MASK_SUBDIR)
	if not os.path.isdir(image_dir):
		raise ValueError(f"Image directory not found: {image_dir}")
	if not os.path.isdir(mask_dir):
		raise ValueError(f"Mask directory not found: {mask_dir}")
	image_paths = sorted(
		[
			os.path.join(image_dir, name)
			for name in os.listdir(image_dir)
			if name.lower().endswith(".png")
		]
	)

	pairs: list[Pair] = []
	missing_masks: list[str] = []

	for image_path in image_paths:
		base = os.path.splitext(os.path.basename(image_path))[0]
		mask_path = os.path.join(mask_dir, f"{base}_mask.png")
		if not os.path.exists(mask_path):
			missing_masks.append(base)
			continue
		pairs.append(Pair(base=base, image_path=image_path, mask_path=mask_path))

	pairs.sort(key=lambda item: item.base)
	return pairs, missing_masks


def center_crop_or_resize(
	image: np.ndarray,
	size: int,
	interpolation: int,
) -> np.ndarray:
	"""Center-crop to size or resize if input is too small.

	Args:
		image: Image array.
		size: Target size for both height and width.
		interpolation: OpenCV interpolation to use for resizing.

	Returns:
		Output array with shape (size, size, ...) or (size, size).
	"""
	height, width = image.shape[:2]
	if height >= size and width >= size:
		top = (height - size) // 2
		left = (width - size) // 2
		return image[top : top + size, left : left + size]
	return cv2.resize(image, (size, size), interpolation=interpolation)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
	"""Convert mask to binary values {0,1}.

	Args:
		mask: Mask array.

	Returns:
		Binary uint8 mask with values {0,1}.
	"""
	binary = (mask > 0).astype(np.uint8)
	return binary


def prepare_pair(
	pair: Pair,
	output_image_path: str,
	output_mask_path: str,
	size: int,
) -> None:
	"""Load, process, and save one image/mask pair.

	Args:
		pair: Pair metadata.
		output_image_path: Output image path.
		output_mask_path: Output mask path.
		size: Output spatial size.
	"""
	image = cv2.imread(pair.image_path, cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError(f"Failed to read image: {pair.image_path}")

	mask = cv2.imread(pair.mask_path, cv2.IMREAD_GRAYSCALE)
	if mask is None:
		raise ValueError(f"Failed to read mask: {pair.mask_path}")

	image = center_crop_or_resize(image, size, interpolation=cv2.INTER_LINEAR)
	mask = binarize_mask(mask)
	mask = center_crop_or_resize(mask, size, interpolation=cv2.INTER_NEAREST)
	mask = binarize_mask(mask)

	cv2.imwrite(output_image_path, image)
	cv2.imwrite(output_mask_path, mask)


def write_folds(
	pairs: Sequence[Pair],
	output_root: str,
	size: int,
	k_splits: int,
	random_state: int,
) -> list[list[Pair]]:
	"""Write K folds to disk.

	Args:
		pairs: Ordered list of pairs.
		output_root: Root directory for output data.
		size: Output size.
		k_splits: Number of folds.
		random_state: Random seed for KFold.

	Returns:
		List of folds, each fold is a list of pairs.
	"""
	kfold = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
	folds: list[list[Pair]] = []

	for fold_index, (_, test_indices) in enumerate(kfold.split(pairs)): # type: ignore
		fold_pairs = [pairs[i] for i in test_indices]
		fold_dir = os.path.join(output_root, f"fold_{fold_index}")
		image_dir = os.path.join(fold_dir, "image")
		mask_dir = os.path.join(fold_dir, "mask")

		create_dir(image_dir)
		create_dir(mask_dir)

		for pair in fold_pairs:
			filename = f"{pair.base}.png"
			output_image_path = os.path.join(image_dir, filename)
			output_mask_path = os.path.join(mask_dir, filename)
			prepare_pair(pair, output_image_path, output_mask_path, size)

		folds.append(fold_pairs)

	return folds


def validate_outputs(output_root: str, folds: Sequence[Sequence[Pair]]) -> None:
	"""Validate that each fold has matching image and mask counts.

	Args:
		output_root: Output root directory.
		folds: Fold metadata list.
	"""
	for fold_index, _ in enumerate(folds):
		fold_dir = os.path.join(output_root, f"fold_{fold_index}")
		image_dir = os.path.join(fold_dir, "image")
		mask_dir = os.path.join(fold_dir, "mask")
		image_count = len([name for name in os.listdir(image_dir) if name.endswith(".png")])
		mask_count = len([name for name in os.listdir(mask_dir) if name.endswith(".png")])
		if image_count != mask_count:
			raise RuntimeError(
				f"Mismatched counts in fold {fold_index}: {image_count} images vs {mask_count} masks"
			)


def spot_check_masks(output_root: str, fold_count: int, seed: int = 42) -> None:
	"""Spot-check a few masks for binary values.

	Args:
		output_root: Output root directory.
		fold_count: Number of folds to scan.
		seed: Random seed.
	"""
	rng = random.Random(seed)
	for fold_index in range(fold_count):
		mask_dir = os.path.join(output_root, f"fold_{fold_index}", "mask")
		mask_names = [name for name in os.listdir(mask_dir) if name.endswith(".png")]
		if not mask_names:
			continue
		sample_name = rng.choice(mask_names)
		mask_path = os.path.join(mask_dir, sample_name)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		if mask is None:
			raise RuntimeError(f"Failed to read mask for validation: {mask_path}")
		unique_values = np.unique(mask)
		if not set(unique_values.tolist()).issubset({0, 1}):
			raise RuntimeError(f"Mask not binary: {mask_path}")


def summarize(folds: Sequence[Sequence[Pair]], output_root: str) -> None:
	"""Print a summary of folds and example paths.

	Args:
		folds: Fold list with pairs.
		output_root: Output root directory.
	"""
	total = sum(len(fold) for fold in folds)
	print(f"Total paired samples: {total}")
	for fold_index, fold_pairs in enumerate(folds):
		print(f"Fold {fold_index} count: {len(fold_pairs)}")
		if fold_pairs:
			first_pair = fold_pairs[0]
			image_path = os.path.join(output_root, f"fold_{fold_index}", "image", f"{first_pair.base}.png")
			mask_path = os.path.join(output_root, f"fold_{fold_index}", "mask", f"{first_pair.base}.png")
			print(f"Fold {fold_index} example image: {image_path}")
			print(f"Fold {fold_index} example mask:  {mask_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: CLI arguments.

	Returns:
		Parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Prepare PennFudanPed data into 5 K-fold buckets."
	)
	parser.add_argument(
		"--input-root",
		default=INPUT_ROOT_DEFAULT,
		help="Path to PennFudanPed root directory.",
	)
	parser.add_argument(
		"--output-root",
		default=OUTPUT_ROOT_DEFAULT,
		help="Path to output root directory.",
	)
	parser.add_argument(
		"--size",
		type=int,
		default=OUTPUT_SIZE,
		help="Output size for width and height.",
	)
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	"""Run dataset preparation.

	Args:
		argv: Optional CLI argument list.

	Returns:
		Exit code.
	"""
	args = parse_args(argv if argv is not None else sys.argv[1:])

	if is_non_empty_dir(args.output_root):
		print(
			"Output directory exists and is not empty. "
			"Remove contents or choose a new output root to proceed."
		)
		return 1

	if not os.path.isdir(args.input_root):
		print(f"Input root not found: {args.input_root}")
		return 1

	pairs, missing_masks = collect_pairs(args.input_root)
	if missing_masks:
		print(f"Skipping {len(missing_masks)} images without matching masks.")
		for base in missing_masks[:5]:
			print(f"Missing mask for: {base}")
		if len(missing_masks) > 5:
			print("Additional missing masks omitted...")

	if not pairs:
		print("No valid image/mask pairs found.")
		return 1

	create_dir(args.output_root)
	folds = write_folds(
		pairs,
		args.output_root,
		args.size,
		K_SPLITS,
		RANDOM_STATE,
	)
	validate_outputs(args.output_root, folds)
	spot_check_masks(args.output_root, K_SPLITS, seed=RANDOM_STATE)
	summarize(folds, args.output_root)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
