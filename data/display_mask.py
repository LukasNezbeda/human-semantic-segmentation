"""Display an image mask in a new window.

This script is intended for quick inspection of segmentation masks, including
Cityscapes label-id PNGs (single-channel) and color masks (3-channel).

Usage:
	python data/display_mask.py path\\to\\mask.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _to_display_image(mask: np.ndarray, *, colorize_grayscale: bool) -> np.ndarray:
	"""Convert an arbitrary mask array into something `cv2.imshow` can display."""
	if mask.ndim == 2:
		if not colorize_grayscale:
			if mask.dtype == np.uint8:
				return mask
			mask_float = mask.astype(np.float32)
			mask_norm = cv2.normalize(mask_float, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
			return mask_norm.astype(np.uint8)

		# Colorize label-id masks for easier viewing.
		if mask.dtype != np.uint8:
			mask_float = mask.astype(np.float32)
			mask_u8 = cv2.normalize(mask_float, None, 0, 255, cv2.NORM_MINMAX).astype( # type: ignore
				np.uint8
			)
		else:
			max_val = int(mask.max()) if mask.size else 0
			if 0 < max_val < 255:
				scale = 255.0 / float(max_val)
				mask_u8 = np.clip(mask.astype(np.float32) * scale, 0, 255).astype(np.uint8)
			else:
				mask_u8 = mask

		# `applyColorMap` expects 8-bit single-channel.
		return cv2.applyColorMap(mask_u8, cv2.COLORMAP_TURBO)

	if mask.ndim == 3:
		# OpenCV uses BGR already (as read from cv2.imread), so show as-is.
		if mask.shape[2] == 4:
			# Drop alpha if present.
			return mask[:, :, :3]
		return mask

	raise ValueError(f"Unsupported mask shape: {mask.shape}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Display a segmentation mask image in a new window.",
	)
	parser.add_argument(
		"mask_path",
		type=Path,
		help="Path to the mask image (e.g., Cityscapes *_labelIds.png or *_color.png).",
	)
	parser.add_argument(
		"--no-colorize",
		action="store_true",
		help="Show single-channel masks as grayscale instead of applying a colormap.",
	)
	parser.add_argument(
		"--window-title",
		type=str,
		default=None,
		help="Optional window title (defaults to the filename).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	mask_path: Path = args.mask_path

	if not mask_path.exists():
		raise FileNotFoundError(f"Mask file not found: {mask_path}")

	mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
	if mask is None:
		raise ValueError(f"Failed to read image: {mask_path}")

	display = _to_display_image(mask, colorize_grayscale=not args.no_colorize)
	title = args.window_title or mask_path.name

	cv2.namedWindow(title, cv2.WINDOW_NORMAL)
	cv2.imshow(title, display)

	# Exit on any keypress or when the window is closed.
	while True:
		if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
			break
		if cv2.waitKey(50) != -1:
			break
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
