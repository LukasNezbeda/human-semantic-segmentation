"""
Prepare the Cityscapes dataset for binary human semantic segmentation.

Creates train/valid/test splits with paired images and binary masks.
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


IMAGE_ROOT_DEFAULT = os.path.join(
	"data",
	"cityscapes",
	"Cityscape Dataset",
	"leftImg8bit",
)
LABEL_ROOT_DEFAULT = os.path.join(
	"data",
	"cityscapes",
	"Fine Annotations",
	"gtFine",
)
OUTPUT_ROOT_DEFAULT = os.path.join("data", "cityscapes", "new_data")
INPUT_SPLITS = ("train", "val")
OUTPUT_SPLITS = ("train", "valid", "test")
SEED = 42
TARGET_HEIGHT = 512
TARGET_WIDTH = 1024
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
HUMAN_LABEL_IDS = (24, 25)


def create_dir(path: str) -> None:
	"""Create a directory if it does not exist.

	Args:
		path: Directory path.
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def is_non_empty_dir(path: str) -> bool:
	"""Return True if a directory exists and contains any entries.

	Args:
		path: Directory path.

	Returns:
		True if the directory exists and has any content.
	"""
	if not os.path.isdir(path):
		return False
	return any(os.scandir(path))


def collect_pairs(image_root: str, label_root: str) -> tuple[list[Pair], list[str]]:
	"""Collect image/mask pairs from Cityscapes.

	Args:
		image_root: Root directory for leftImg8bit images.
		label_root: Root directory for gtFine labelIds masks.

	Returns:
		A tuple of (pairs, missing_masks), where pairs are sorted by base name.
	"""
	if not os.path.isdir(image_root):
		raise ValueError(f"Image root not found: {image_root}")
	if not os.path.isdir(label_root):
		raise ValueError(f"Label root not found: {label_root}")

	pairs: list[Pair] = []
	missing_masks: list[str] = []

	for root, _, files in os.walk(image_root):
		for name in files:
			if not name.endswith("_leftImg8bit.png"):
				continue
			image_path = os.path.join(root, name)
			city = os.path.basename(root)
			split = os.path.basename(os.path.dirname(root))
			if split not in INPUT_SPLITS:
				continue
			base = name.replace("_leftImg8bit.png", "")
			mask_name = f"{base}_gtFine_labelIds.png"
			mask_path = os.path.join(label_root, split, city, mask_name)
			if not os.path.exists(mask_path):
				missing_masks.append(image_path)
				continue
			pairs.append(Pair(base=base, image_path=image_path, mask_path=mask_path))

	pairs.sort(key=lambda item: item.base)
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
		interpolation: OpenCV interpolation to use for resizing.

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


def binarize_mask(mask: np.ndarray) -> np.ndarray:
	"""Convert Cityscapes labelIds to human-only binary values.

	Args:
		mask: LabelIds mask array.

	Returns:
		Binary uint8 mask with values {0,1}.
	"""
	binary = np.isin(mask, HUMAN_LABEL_IDS).astype(np.uint8)
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

	image = center_crop_or_resize(
		image,
		target_height,
		target_width,
		interpolation=cv2.INTER_AREA,
	)
	mask = binarize_mask(mask)
	mask = center_crop_or_resize(
		mask,
		target_height,
		target_width,
		interpolation=cv2.INTER_NEAREST,
	)
	mask = binarize_mask(mask)

	if image.shape[:2] != (target_height, target_width):
		raise RuntimeError(
			f"Unexpected image shape {image.shape} for {pair.image_path}"
		)
	if mask.shape[:2] != (target_height, target_width):
		raise RuntimeError(
			f"Unexpected mask shape {mask.shape} for {pair.mask_path}"
		)

	cv2.imwrite(output_image_path, image)
	cv2.imwrite(output_mask_path, mask)


def split_pairs(
	pairs: Sequence[Pair],
	train_ratio: float,
	valid_ratio: float,
	seed: int,
) -> dict[str, list[Pair]]:
	"""Split pairs into train/valid/test sets.

	Args:
		pairs: List of pairs.
		train_ratio: Ratio for training split.
		valid_ratio: Ratio for validation split.
		seed: Random seed.

	Returns:
		Dictionary with train, valid, test splits.
	"""
	rng = random.Random(seed)
	shuffled = list(pairs)
	rng.shuffle(shuffled)

	total = len(shuffled)
	train_count = int(total * train_ratio)
	valid_count = int(total * valid_ratio)
	test_count = total - train_count - valid_count

	train_pairs = shuffled[:train_count]
	valid_pairs = shuffled[train_count : train_count + valid_count]
	test_pairs = shuffled[train_count + valid_count : train_count + valid_count + test_count]

	return {
		"train": train_pairs,
		"valid": valid_pairs,
		"test": test_pairs,
	}


def write_split(
	split_name: str,
	pairs: Sequence[Pair],
	output_root: str,
	target_height: int,
	target_width: int,
) -> None:
	"""Write split pairs to disk.

	Args:
		split_name: Split name (train/valid/test).
		pairs: Pair list.
		output_root: Output root directory.
		target_height: Output height.
		target_width: Output width.
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
		prepare_pair(pair, output_image_path, output_mask_path, target_height, target_width)

	validate_split_counts(output_root, split_name)


def validate_split_counts(output_root: str, split_name: str) -> None:
	"""Validate that split image and mask counts match.

	Args:
		output_root: Output root directory.
		split_name: Split name.
	"""
	split_dir = os.path.join(output_root, split_name)
	image_dir = os.path.join(split_dir, "image")
	mask_dir = os.path.join(split_dir, "mask")
	image_count = len([name for name in os.listdir(image_dir) if name.endswith(".png")])
	mask_count = len([name for name in os.listdir(mask_dir) if name.endswith(".png")])
	if image_count != mask_count:
		raise RuntimeError(
			f"Mismatched counts in {split_name}: {image_count} images vs {mask_count} masks"
		)


def spot_check_outputs(
	output_root: str,
	split_name: str,
	target_height: int,
	target_width: int,
	seed: int,
) -> None:
	"""Spot-check masks for binary values and correct shapes.

	Args:
		output_root: Output root directory.
		split_name: Split name.
		target_height: Output height.
		target_width: Output width.
		seed: Random seed.
	"""
	rng = random.Random(seed)
	split_dir = os.path.join(output_root, split_name)
	image_dir = os.path.join(split_dir, "image")
	mask_dir = os.path.join(split_dir, "mask")
	mask_names = [name for name in os.listdir(mask_dir) if name.endswith(".png")]
	if not mask_names:
		return
	mask_name = rng.choice(mask_names)
	mask_path = os.path.join(mask_dir, mask_name)
	image_path = os.path.join(image_dir, mask_name)

	mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	if mask is None or image is None:
		raise RuntimeError(f"Failed to read spot-check files: {image_path}, {mask_path}")
	if mask.shape[:2] != (target_height, target_width):
		raise RuntimeError(f"Mask has unexpected shape: {mask_path}")
	if image.shape[:2] != (target_height, target_width):
		raise RuntimeError(f"Image has unexpected shape: {image_path}")
	unique_values = np.unique(mask)
	if not set(unique_values.tolist()).issubset({0, 1}):
		raise RuntimeError(f"Mask not binary: {mask_path}")


def summarize(splits: dict[str, Sequence[Pair]], output_root: str) -> None:
	"""Print summary of splits and example paths.

	Args:
		splits: Mapping of split names to pairs.
		output_root: Output root directory.
	"""
	total = sum(len(pairs) for pairs in splits.values())
	print(f"Total paired samples: {total}")
	for split_name, pairs in splits.items():
		print(f"Split {split_name} count: {len(pairs)}")
		if pairs:
			first_pair = pairs[0]
			image_path = os.path.join(
				output_root,
				split_name,
				"image",
				f"{first_pair.base}.png",
			)
			mask_path = os.path.join(
				output_root,
				split_name,
				"mask",
				f"{first_pair.base}.png",
			)
			print(f"Split {split_name} example image: {image_path}")
			print(f"Split {split_name} example mask:  {mask_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: CLI arguments.

	Returns:
		Parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Prepare Cityscapes data for binary human segmentation."
	)
	parser.add_argument(
		"--image-root",
		default=IMAGE_ROOT_DEFAULT,
		help="Path to leftImg8bit root directory.",
	)
	parser.add_argument(
		"--label-root",
		default=LABEL_ROOT_DEFAULT,
		help="Path to gtFine root directory.",
	)
	parser.add_argument(
		"--output-root",
		default=OUTPUT_ROOT_DEFAULT,
		help="Path to output root directory.",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=TARGET_HEIGHT,
		help="Output height.",
	)
	parser.add_argument(
		"--width",
		type=int,
		default=TARGET_WIDTH,
		help="Output width.",
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

	if not os.path.isdir(args.image_root):
		print(f"Image root not found: {args.image_root}")
		return 1
	if not os.path.isdir(args.label_root):
		print(f"Label root not found: {args.label_root}")
		return 1

	pairs, missing_masks = collect_pairs(args.image_root, args.label_root)
	if missing_masks:
		print(f"Skipping {len(missing_masks)} images without matching masks.")
		for missing in missing_masks[:5]:
			print(f"Missing mask for: {missing}")
		if len(missing_masks) > 5:
			print("Additional missing masks omitted...")

	if not pairs:
		print("No valid image/mask pairs found.")
		return 1

	create_dir(args.output_root)
	splits = split_pairs(pairs, TRAIN_RATIO, VALID_RATIO, SEED)
	for split_name in OUTPUT_SPLITS:
		write_split(
			split_name,
			splits[split_name],
			args.output_root,
			args.height,
			args.width,
		)
		spot_check_outputs(
			args.output_root,
			split_name,
			args.height,
			args.width,
			seed=SEED,
		)

	summarize(splits, args.output_root) # type: ignore
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
