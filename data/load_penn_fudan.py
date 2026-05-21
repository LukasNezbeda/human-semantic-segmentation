"""Prepare the PennFudanPed dataset for semantic segmentation.

Creates a deterministic 70/15/15 train/valid/test split with paired images and
binary masks. Each output sample is center-cropped to 512x512 or resized as a
fallback if cropping is not possible.

Input (expected):
	data/penn_fudan/PennFudanPed/PNGImages/*.png
	data/penn_fudan/PennFudanPed/PedMasks/*_mask.png

Output (created):
	data/penn_fudan/new_data/{train,valid,test}/{image,mask}/*.png
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


def center_crop_or_resize(image: np.ndarray, size: int, interpolation: int) -> np.ndarray:
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
	"""Convert a mask to binary values {0,1}.

	Args:
		mask: Mask array.

	Returns:
		Binary uint8 mask with values {0,1}.
	"""
	return (mask > 0).astype(np.uint8)


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

	if not cv2.imwrite(output_image_path, image):
		raise RuntimeError(f"Failed to write image: {output_image_path}")
	if not cv2.imwrite(output_mask_path, mask):
		raise RuntimeError(f"Failed to write mask: {output_mask_path}")


def split_pairs(
	pairs: Sequence[Pair],
	train_ratio: float,
	valid_ratio: float,
	test_ratio: float,
	seed: int,
) -> tuple[list[Pair], list[Pair], list[Pair]]:
	"""Split paired samples into train/valid/test deterministically.

	Rounding policy:
		n_test = floor(test_ratio * n)
		n_valid = floor(valid_ratio * n)
		n_train = n - n_valid - n_test

	Args:
		pairs: Paired samples.
		train_ratio: Train fraction.
		valid_ratio: Validation fraction.
		test_ratio: Test fraction.
		seed: Random seed.

	Returns:
		(train_pairs, valid_pairs, test_pairs)
	"""
	n = len(pairs)
	if n == 0:
		return [], [], []
	if train_ratio < 0 or valid_ratio < 0 or test_ratio < 0:
		raise ValueError("Split ratios must be non-negative")
	if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
		raise ValueError("Split ratios must sum to 1.0")

	shuffled = list(pairs)
	rng = random.Random(seed)
	rng.shuffle(shuffled)

	n_test = int(test_ratio * n)
	n_valid = int(valid_ratio * n)
	n_train = n - n_valid - n_test

	train_pairs = shuffled[:n_train]
	valid_pairs = shuffled[n_train : n_train + n_valid]
	test_pairs = shuffled[n_train + n_valid :]
	return train_pairs, valid_pairs, test_pairs


def write_split(
	split_name: str,
	pairs: Sequence[Pair],
	output_root: str,
	size: int,
) -> None:
	"""Write one split to disk.

	Args:
		split_name: One of "train", "valid", "test".
		pairs: Pairs to write.
		output_root: Output root directory.
		size: Output size.
"""
	split_dir = os.path.join(output_root, split_name)
	image_dir = os.path.join(split_dir, "image")
	mask_dir = os.path.join(split_dir, "mask")

	create_dir(image_dir)
	create_dir(mask_dir)

	for pair in pairs:
		filename = f"{pair.base}.png"
		output_image_path = os.path.join(image_dir, filename)
		output_mask_path = os.path.join(mask_dir, filename)
		prepare_pair(pair, output_image_path, output_mask_path, size)


def validate_outputs(output_root: str) -> None:
	"""Validate that each split has matching image and mask counts."""
	for split_name in ("train", "valid", "test"):
		split_dir = os.path.join(output_root, split_name)
		image_dir = os.path.join(split_dir, "image")
		mask_dir = os.path.join(split_dir, "mask")
		if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
			raise RuntimeError(f"Missing output directories for split: {split_name}")

		image_count = len(
			[name for name in os.listdir(image_dir) if name.lower().endswith(".png")]
		)
		mask_count = len(
			[name for name in os.listdir(mask_dir) if name.lower().endswith(".png")]
		)
		if image_count != mask_count:
			raise RuntimeError(
				f"Mismatched counts in {split_name}: {image_count} images vs {mask_count} masks"
			)


def spot_check_masks(output_root: str, seed: int, samples_per_split: int = 3) -> None:
	"""Spot-check a few output masks for binary values.

	Args:
		output_root: Output root directory.
		seed: Random seed.
		samples_per_split: Number of masks to sample per split.
	"""
	rng = random.Random(seed)
	for split_name in ("train", "valid", "test"):
		mask_dir = os.path.join(output_root, split_name, "mask")
		mask_names = [name for name in os.listdir(mask_dir) if name.lower().endswith(".png")]
		if not mask_names:
			continue

		for _ in range(min(samples_per_split, len(mask_names))):
			sample_name = rng.choice(mask_names)
			mask_path = os.path.join(mask_dir, sample_name)
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			if mask is None:
				raise RuntimeError(f"Failed to read mask for validation: {mask_path}")
			unique_values = np.unique(mask)
			if not set(unique_values.tolist()).issubset({0, 1}):
				raise RuntimeError(f"Mask not binary: {mask_path}")


def summarize(
	output_root: str,
	train_pairs: Sequence[Pair],
	valid_pairs: Sequence[Pair],
	test_pairs: Sequence[Pair],
) -> None:
	"""Print a summary and example paths."""
	total = len(train_pairs) + len(valid_pairs) + len(test_pairs)
	print(f"Total paired samples written: {total}")
	print(f"Train count: {len(train_pairs)}")
	print(f"Valid count: {len(valid_pairs)}")
	print(f"Test count:  {len(test_pairs)}")

	def _print_example(split_name: str, pairs: Sequence[Pair]) -> None:
		if not pairs:
			return
		first = pairs[0]
		image_path = os.path.join(output_root, split_name, "image", f"{first.base}.png")
		mask_path = os.path.join(output_root, split_name, "mask", f"{first.base}.png")
		print(f"{split_name} example image: {image_path}")
		print(f"{split_name} example mask:  {mask_path}")

	_print_example("train", train_pairs)
	_print_example("valid", valid_pairs)
	_print_example("test", test_pairs)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: CLI arguments.

	Returns:
		Parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Prepare PennFudanPed data into train/valid/test folders."
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
	parser.add_argument(
		"--seed",
		type=int,
		default=RANDOM_STATE,
		help="Seed for deterministic splitting.",
	)
	parser.add_argument("--train-ratio", type=float, default=0.70)
	parser.add_argument("--valid-ratio", type=float, default=0.15)
	parser.add_argument("--test-ratio", type=float, default=0.15)
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
			"Remove contents or choose a new output root to proceed.",
			file=sys.stderr,
		)
		return 1

	if not os.path.isdir(args.input_root):
		print(f"Input root not found: {args.input_root}", file=sys.stderr)
		return 1

	pairs, missing_masks = collect_pairs(args.input_root)
	if missing_masks:
		print(f"Skipping {len(missing_masks)} images without matching masks.")
		for base in missing_masks[:5]:
			print(f"Missing mask for: {base}")
		if len(missing_masks) > 5:
			print("Additional missing masks omitted...")

	if not pairs:
		print("No valid image/mask pairs found.", file=sys.stderr)
		return 1

	create_dir(args.output_root)
	train_pairs, valid_pairs, test_pairs = split_pairs(
		pairs,
		train_ratio=float(args.train_ratio),
		valid_ratio=float(args.valid_ratio),
		test_ratio=float(args.test_ratio),
		seed=int(args.seed),
	)

	write_split("train", train_pairs, args.output_root, int(args.size))
	write_split("valid", valid_pairs, args.output_root, int(args.size))
	write_split("test", test_pairs, args.output_root, int(args.size))
	validate_outputs(args.output_root)
	spot_check_masks(args.output_root, seed=int(args.seed))
	summarize(args.output_root, train_pairs, valid_pairs, test_pairs)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
