"""
Prepare the Cityscapes dataset for human-only semantic segmentation.

Creates train/valid/test splits with binary masks for person and rider.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Pair:
	"""Image/mask pair with a shared base name."""

	base: str
	image_path: str
	mask_path: str


INPUT_ROOT_DEFAULT = os.path.join("data", "cityscapes")
IMAGE_ROOT_NAME = "Cityscape Dataset"
IMAGE_SUBDIR = "leftImg8bit"
MASK_ROOT_NAME = "Fine Annotations"
MASK_SUBDIR = "gtFine"
IMAGE_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_labelIds.png"
OUTPUT_ROOT_DEFAULT = os.path.join("data", "cityscapes", "new_data")
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 1024
RANDOM_STATE = 42
CITYSCAPES_SPLITS = ("train", "val", "test")
OUTPUT_SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
PERSON_LABEL_IDS = (24, 25)


def create_dir(path: str) -> None:
	"""Create a directory if it does not exist."""
	if not os.path.exists(path):
		os.makedirs(path)


def is_non_empty_dir(path: str) -> bool:
	"""Return True if a directory exists and contains any entries."""
	if not os.path.isdir(path):
		return False
	return any(os.scandir(path))


def split_dirs(input_root: str, split: str) -> tuple[str, str]:
	"""Return image and mask directories for a split.

	Args:
		input_root: Root directory containing Cityscapes data.
		split: Dataset split name (train/val/test).

	Returns:
		Tuple of (image_dir, mask_dir).
	"""
	image_dir = os.path.join(input_root, IMAGE_ROOT_NAME, IMAGE_SUBDIR, split)
	mask_dir = os.path.join(input_root, MASK_ROOT_NAME, MASK_SUBDIR, split)
	return image_dir, mask_dir


def image_name_to_base(name: str) -> str | None:
	"""Extract the shared base name from a Cityscapes image filename.

	Args:
		name: Filename such as *_leftImg8bit.png.

	Returns:
		Base name without suffix or None if the suffix does not match.
	"""
	if not name.endswith(IMAGE_SUFFIX):
		return None
	return name[: -len(IMAGE_SUFFIX)]


def base_to_mask_name(base: str) -> str:
	"""Create a mask filename from a base name."""
	return f"{base}{MASK_SUFFIX}"


def collect_pairs(input_root: str, split: str) -> tuple[list[Pair], list[str]]:
	"""Collect image/mask pairs for a split.

	Args:
		input_root: Root directory containing Cityscapes data.
		split: Dataset split name.

	Returns:
		Tuple of (pairs, missing_masks), where pairs are sorted by base name.
	"""
	image_dir, mask_dir = split_dirs(input_root, split)
	if not os.path.isdir(image_dir):
		raise ValueError(f"Image directory not found: {image_dir}")
	if not os.path.isdir(mask_dir):
		raise ValueError(f"Mask directory not found: {mask_dir}")

	pairs: list[Pair] = []
	missing_masks: list[str] = []

	for root, _, files in os.walk(image_dir):
		for filename in files:
			base = image_name_to_base(filename)
			if base is None:
				continue
			image_path = os.path.join(root, filename)
			rel_dir = os.path.relpath(root, image_dir)
			mask_subdir = mask_dir if rel_dir == "." else os.path.join(mask_dir, rel_dir)
			mask_path = os.path.join(mask_subdir, base_to_mask_name(base))
			if not os.path.exists(mask_path):
				missing_masks.append(base)
				continue
			pairs.append(Pair(base=base, image_path=image_path, mask_path=mask_path))

	pairs.sort(key=lambda item: item.base)
	missing_masks.sort()
	return pairs, missing_masks


def center_crop_or_resize(
	image: np.ndarray,
	target_height: int,
	target_width: int,
	interpolation: int,
) -> np.ndarray:
	"""Center-crop to target size or resize if input is too small.

	Args:
		image: Image array.
		target_height: Target height.
		target_width: Target width.
		interpolation: OpenCV interpolation for resizing.

	Returns:
		Output array with shape (target_height, target_width, ...) or
		(target_height, target_width).
	"""
	height, width = image.shape[:2]
	if height >= target_height and width >= target_width:
		top = (height - target_height) // 2
		left = (width - target_width) // 2
		return image[top : top + target_height, left : left + target_width]
	return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def binarize_mask_from_label_ids(mask: np.ndarray) -> np.ndarray:
	"""Convert labelId mask to binary values {0,1} for person and rider.

	Args:
		mask: LabelId mask array.

	Returns:
		Binary uint8 mask with values {0,1}.
	"""
	binary = np.isin(mask, PERSON_LABEL_IDS).astype(np.uint8)
	return binary


def prepare_pair(
	pair: Pair,
	output_image_path: str,
	output_mask_path: str,
	target_height: int,
	target_width: int,
) -> None:
	"""Load, process, and save one image/mask pair.

	Args:
		pair: Pair metadata.
		output_image_path: Output image path.
		output_mask_path: Output mask path.
		target_height: Output height.
		target_width: Output width.
	"""
	image = cv2.imread(pair.image_path, cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError(f"Failed to read image: {pair.image_path}")

	mask = cv2.imread(pair.mask_path, cv2.IMREAD_GRAYSCALE)
	if mask is None:
		raise ValueError(f"Failed to read mask: {pair.mask_path}")

	image = center_crop_or_resize(image, target_height, target_width, cv2.INTER_LINEAR)
	mask = binarize_mask_from_label_ids(mask)
	mask = center_crop_or_resize(mask, target_height, target_width, cv2.INTER_NEAREST)
	mask = (mask > 0).astype(np.uint8)

	cv2.imwrite(output_image_path, image)
	cv2.imwrite(output_mask_path, mask)


def write_split(
	pairs: Sequence[Pair],
	output_root: str,
	output_split: str,
	target_height: int,
	target_width: int,
) -> None:
	"""Write one split to disk.

	Args:
		pairs: Pairs for the split.
		output_root: Output root directory.
		output_split: Output split name.
		target_height: Output height.
		target_width: Output width.
	"""
	split_dir = os.path.join(output_root, output_split)
	image_dir = os.path.join(split_dir, "image")
	mask_dir = os.path.join(split_dir, "mask")

	create_dir(image_dir)
	create_dir(mask_dir)

	for pair in pairs:
		filename = f"{pair.base}.png"
		output_image_path = os.path.join(image_dir, filename)
		output_mask_path = os.path.join(mask_dir, filename)
		prepare_pair(pair, output_image_path, output_mask_path, target_height, target_width)


def validate_outputs(
	output_root: str,
	output_splits: Iterable[str],
	target_height: int,
	target_width: int,
) -> None:
	"""Validate that each split has matching counts and expected shapes.

	Args:
		output_root: Output root directory.
		output_splits: Output split names.
		target_height: Expected height.
		target_width: Expected width.
	"""
	for split in output_splits:
		split_dir = os.path.join(output_root, split)
		image_dir = os.path.join(split_dir, "image")
		mask_dir = os.path.join(split_dir, "mask")
		image_names = sorted(name for name in os.listdir(image_dir) if name.endswith(".png"))
		mask_names = sorted(name for name in os.listdir(mask_dir) if name.endswith(".png"))

		if image_names != mask_names:
			raise RuntimeError(f"Mismatched image/mask files in split '{split}'.")

		for name in image_names:
			image_path = os.path.join(image_dir, name)
			mask_path = os.path.join(mask_dir, name)
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			if image is None:
				raise RuntimeError(f"Failed to read image: {image_path}")
			if mask is None:
				raise RuntimeError(f"Failed to read mask: {mask_path}")
			if image.shape[:2] != (target_height, target_width):
				raise RuntimeError(
					f"Unexpected image size {image.shape[:2]} in {image_path}"
				)
			if mask.shape != (target_height, target_width):
				raise RuntimeError(
					f"Unexpected mask size {mask.shape} in {mask_path}"
				)


def spot_check_masks(output_root: str, output_splits: Iterable[str], seed: int = 42) -> None:
	"""Spot-check a few masks for binary values.

	Args:
		output_root: Output root directory.
		output_splits: Output split names.
		seed: Random seed.
	"""
	rng = random.Random(seed)
	for split in output_splits:
		mask_dir = os.path.join(output_root, split, "mask")
		mask_names = [name for name in os.listdir(mask_dir) if name.endswith(".png")]
		if not mask_names:
			continue
		sample_names = rng.sample(mask_names, k=min(3, len(mask_names)))
		for name in sample_names:
			mask_path = os.path.join(mask_dir, name)
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			if mask is None:
				raise RuntimeError(f"Failed to read mask for validation: {mask_path}")
			unique_values = np.unique(mask)
			if not set(unique_values.tolist()).issubset({0, 1}):
				raise RuntimeError(f"Mask not binary: {mask_path}")


def summarize(split_pairs: dict[str, Sequence[Pair]], output_root: str) -> None:
	"""Print a summary of splits and example paths.

	Args:
		split_pairs: Mapping of output split names to pairs.
		output_root: Output root directory.
	"""
	total = sum(len(pairs) for pairs in split_pairs.values())
	print(f"Total paired samples: {total}")
	for split, pairs in split_pairs.items():
		print(f"Split '{split}' count: {len(pairs)}")
		if pairs:
			first_pair = pairs[0]
			image_path = os.path.join(output_root, split, "image", f"{first_pair.base}.png")
			mask_path = os.path.join(output_root, split, "mask", f"{first_pair.base}.png")
			print(f"Split '{split}' example image: {image_path}")
			print(f"Split '{split}' example mask:  {mask_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: CLI arguments.

	Returns:
		Parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Prepare Cityscapes data for human-only segmentation."
	)
	parser.add_argument(
		"--input-root",
		default=INPUT_ROOT_DEFAULT,
		help="Path to Cityscapes root directory.",
	)
	parser.add_argument(
		"--output-root",
		default=OUTPUT_ROOT_DEFAULT,
		help="Path to output root directory.",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=DEFAULT_HEIGHT,
		help="Output image height.",
	)
	parser.add_argument(
		"--width",
		type=int,
		default=DEFAULT_WIDTH,
		help="Output image width.",
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

	create_dir(args.output_root)
	for split in CITYSCAPES_SPLITS:
		output_split = OUTPUT_SPLIT_MAP[split]
		split_dir = os.path.join(args.output_root, output_split)
		create_dir(os.path.join(split_dir, "image"))
		create_dir(os.path.join(split_dir, "mask"))

	missing_total: list[str] = []
	split_pairs: dict[str, Sequence[Pair]] = {}

	for split in CITYSCAPES_SPLITS:
		pairs, missing = collect_pairs(args.input_root, split)
		if missing:
			print(f"Split '{split}': skipping {len(missing)} images without matching masks.")
			for base in missing[:5]:
				print(f"Missing mask for: {base}")
			if len(missing) > 5:
				print("Additional missing masks omitted...")
			missing_total.extend(missing)

		output_split = OUTPUT_SPLIT_MAP[split]
		if not pairs:
			print(f"Warning: no valid pairs found for split '{split}'.")
			split_pairs[output_split] = []
			continue
		write_split(pairs, args.output_root, output_split, args.height, args.width)
		split_pairs[output_split] = pairs

	output_splits = [OUTPUT_SPLIT_MAP[split] for split in CITYSCAPES_SPLITS]
	validate_outputs(args.output_root, output_splits, args.height, args.width)
	spot_check_masks(args.output_root, output_splits, seed=RANDOM_STATE)
	summarize(split_pairs, args.output_root)

	if missing_total:
		print(f"Total missing masks across splits: {len(missing_total)}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
