"""PennFudanPed dataset preparation for semantic segmentation.

This script prepares the PennFudanPed dataset into a train/valid/test split and writes
images and binary masks into a folder structure suitable for segmentation training.

Input (assumed to exist):
    data/penn_fudan/PennFudanPed/
        PNGImages/   (images: *.png)
        PedMasks/    (instance masks: *_mask.png)

Output (created by this script):
    data/penn_fudan/new_data/
        train/image, train/mask
        valid/image, valid/mask
        test/image,  test/mask

Masks are converted to binary semantic masks with values {0, 1}.
Images and masks are center-cropped to 512x512; if center-cropping is not possible,
the script falls back to resizing to 512x512 (masks use nearest-neighbor).

Run from repo root:
    python data/load_penn_fudan.py
"""

from __future__ import annotations

import os
import sys
import random
from glob import glob
from typing import Iterable

import cv2
import numpy as np


TARGET_HEIGHT = 512
TARGET_WIDTH = 512
DEFAULT_SEED = 42


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

    for root, _dirs, files in os.walk(path):
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

    # Fallback resize
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a mask to {0, 1} uint8 values."""
    return (mask > 0).astype(np.uint8)


def center_crop_or_resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Center-crop a mask to (height, width), fallback to resize if crop not possible.

    Resizing uses nearest-neighbor to preserve discrete labels.
    """
    h, w = mask.shape[:2]

    if h >= height and w >= width:
        y0 = (h - height) // 2
        x0 = (w - width) // 2
        return mask[y0 : y0 + height, x0 : x0 + width]

    # Fallback resize (nearest for masks)
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized


def list_image_mask_pairs(dataset_root: str) -> list[tuple[str, str, str]]:
    """List paired (image, mask) samples in the PennFudan dataset.

    Args:
        dataset_root: Path to `PennFudanPed`.

    Returns:
        List of tuples (base_name, image_path, mask_path).

    Raises:
        FileNotFoundError: If image or mask directories do not exist.
        RuntimeError: If no valid pairs are found.
    """
    images_dir = os.path.join(dataset_root, "PNGImages")
    masks_dir = os.path.join(dataset_root, "PedMasks")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    image_paths = sorted(glob(os.path.join(images_dir, "*.png")))

    pairs: list[tuple[str, str, str]] = []
    missing_masks: list[str] = []

    for image_path in image_paths:
        base = os.path.splitext(os.path.basename(image_path))[0]
        mask_filename = f"{base}_mask.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        if not os.path.exists(mask_path):
            missing_masks.append(image_path)
            continue
        pairs.append((base, image_path, mask_path))

    if missing_masks:
        print(f"Warning: {len(missing_masks)} images without a matching mask were skipped.")

    if not pairs:
        raise RuntimeError("No (image, mask) pairs found. Check dataset structure.")

    return pairs


def split_pairs(
    pairs: list[tuple[str, str, str]],
    seed: int,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Split paired samples into train/valid/test.

    Rounding is handled by assigning any remainder to the training split.
    """
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_valid = int(n_total * valid_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_valid - n_test

    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]

    return train, valid, test


def write_split(
    split_name: str,
    pairs: Iterable[tuple[str, str, str]],
    out_images_dir: str,
    out_masks_dir: str,
) -> int:
    """Write a dataset split to disk.

    Args:
        split_name: Human-readable split label for logging.
        pairs: Iterable of (base_name, image_path, mask_path).
        out_images_dir: Output directory for images.
        out_masks_dir: Output directory for masks.

    Returns:
        Number of samples written.

    Raises:
        RuntimeError: If any image/mask cannot be read or written.
    """
    create_dir(out_images_dir)
    create_dir(out_masks_dir)

    written = 0
    for base, image_path, mask_path in pairs:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # Convert instance mask to binary semantic mask (0/1)
        mask_bin = binarize_mask(mask)

        # Crop/resize
        image_out = center_crop_or_resize_image(image, TARGET_HEIGHT, TARGET_WIDTH)
        mask_out = center_crop_or_resize_mask(mask_bin, TARGET_HEIGHT, TARGET_WIDTH)

        # Guarantee binary values after any resize
        mask_out = binarize_mask(mask_out)

        out_image_path = os.path.join(out_images_dir, f"{base}.png")
        out_mask_path = os.path.join(out_masks_dir, f"{base}.png")

        ok_img = cv2.imwrite(out_image_path, image_out)
        ok_msk = cv2.imwrite(out_mask_path, mask_out)
        if not ok_img:
            raise RuntimeError(f"Failed to write image: {out_image_path}")
        if not ok_msk:
            raise RuntimeError(f"Failed to write mask: {out_mask_path}")

        written += 1

    print(f"{split_name}: wrote {written} samples")
    return written


def assert_split_counts_equal(images_dir: str, masks_dir: str) -> None:
    """Assert that image and mask counts match for a split."""
    image_files = sorted(glob(os.path.join(images_dir, "*.png")))
    mask_files = sorted(glob(os.path.join(masks_dir, "*.png")))
    if len(image_files) != len(mask_files):
        raise AssertionError(
            f"Mismatched counts in {os.path.dirname(images_dir)}: "
            f"{len(image_files)} images vs {len(mask_files)} masks"
        )


def main() -> int:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_root = os.path.join(project_root, "data", "penn_fudan", "PennFudanPed")
    output_root = os.path.join(project_root, "data", "penn_fudan", "new_data")

    if is_dir_nonempty(output_root):
        print(
            f"Refusing to continue: output directory already exists and is not empty: {output_root}\n"
            "Delete/move it first if you want to regenerate the dataset split."
        )
        return 1

    pairs = list_image_mask_pairs(dataset_root)
    train_pairs, valid_pairs, test_pairs = split_pairs(pairs, seed=DEFAULT_SEED)

    print(f"Found {len(pairs)} paired samples")
    print(f"Split sizes: train={len(train_pairs)}, valid={len(valid_pairs)}, test={len(test_pairs)}")

    # Output directory structure
    train_images_dir = os.path.join(output_root, "train", "image")
    train_masks_dir = os.path.join(output_root, "train", "mask")
    valid_images_dir = os.path.join(output_root, "valid", "image")
    valid_masks_dir = os.path.join(output_root, "valid", "mask")
    test_images_dir = os.path.join(output_root, "test", "image")
    test_masks_dir = os.path.join(output_root, "test", "mask")

    write_split("Train", train_pairs, train_images_dir, train_masks_dir)
    write_split("Valid", valid_pairs, valid_images_dir, valid_masks_dir)
    write_split("Test", test_pairs, test_images_dir, test_masks_dir)

    # Lightweight validation
    assert_split_counts_equal(train_images_dir, train_masks_dir)
    assert_split_counts_equal(valid_images_dir, valid_masks_dir)
    assert_split_counts_equal(test_images_dir, test_masks_dir)

    # Spot-check mask values on a single file (if available)
    sample_mask_files = glob(os.path.join(train_masks_dir, "*.png"))
    if sample_mask_files:
        sample_mask = cv2.imread(sample_mask_files[0], cv2.IMREAD_GRAYSCALE)
        if sample_mask is None:
            raise RuntimeError(f"Failed to read written mask: {sample_mask_files[0]}")
        unique_vals = set(np.unique(sample_mask).tolist())
        if not unique_vals.issubset({0, 1}):
            raise AssertionError(f"Output mask is not binary {unique_vals} in {sample_mask_files[0]}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
