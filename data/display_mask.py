"""Display segmentation masks with a color palette.

Supports:
	- Cityscapes `*_gtFine_labelIds.png` masks (label IDs 0..33, ignore 255)
	- Cityscapes `*_gtFine_color.png` masks (already colored)
  - PennFudanPed `*_mask.png` masks (instance IDs 0..N)

Input:
  Mask image path

Output:
  A window displaying the mask with colors.

Usage:
	python data/display_mask.py --mask path/to/mask.png
	python data/display_mask.py --mask path/to/mask.png --dataset cityscapes
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

try:
	import cv2
except ModuleNotFoundError as exc:  # pragma: no cover
	raise ModuleNotFoundError(
		"Missing dependency: OpenCV (cv2). "
		"Install `opencv-python` (or run this script in the project's conda env)."
	) from exc
import numpy as np


Dataset = Literal["auto", "cityscapes", "pennfudan"]


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments."""
	parser = argparse.ArgumentParser(description="Display a segmentation mask with colors")
	parser.add_argument(
		"--mask",
		required=True,
		help="Path to a mask image (Cityscapes labelIds or PennFudan PedMasks)",
	)
	parser.add_argument(
		"--dataset",
		choices=["auto", "cityscapes", "pennfudan"],
		default="auto",
		help="How to interpret mask values (default: auto)",
	)
	parser.add_argument(
		"--title",
		default=None,
		help="Optional window title (default: derived from filename)",
	)
	return parser.parse_args()


def read_mask(mask_path: str) -> np.ndarray:
	"""Read a mask image.

	Args:
		mask_path: Path to the mask image.

	Returns:
		Mask as either a 2D array (label/instance ids) or a 3D uint8 image.
	"""
	if not os.path.exists(mask_path):
		raise FileNotFoundError(f"Mask not found: {mask_path}")

	mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
	if mask is None:
		raise ValueError(f"Failed to read mask: {mask_path}")
	return mask


def _hsv_to_rgb_uint8(h: float, s: float, v: float) -> tuple[int, int, int]:
	"""Convert HSV in [0,1] to RGB uint8."""
	# Manual conversion to avoid extra dependencies.
	if s <= 0:
		g = int(round(v * 255))
		return g, g, g

	h = (h % 1.0) * 6.0
	i = int(h)
	f = h - i
	p = v * (1.0 - s)
	q = v * (1.0 - s * f)
	t = v * (1.0 - s * (1.0 - f))

	if i == 0:
		r, g, b = v, t, p
	elif i == 1:
		r, g, b = q, v, p
	elif i == 2:
		r, g, b = p, v, t
	elif i == 3:
		r, g, b = p, q, v
	elif i == 4:
		r, g, b = t, p, v
	else:
		r, g, b = v, p, q

	return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def generate_distinct_palette(n: int) -> list[tuple[int, int, int]]:
	"""Generate a deterministic list of visually distinct RGB colors.

	Args:
		n: Number of colors to generate.

	Returns:
		List of RGB colors (uint8 tuples).
	"""
	golden_ratio = 0.618033988749895
	colors: list[tuple[int, int, int]] = []
	seen: set[tuple[int, int, int]] = set()

	for idx in range(n):
		# Spread hues using the golden ratio; tweak on collision.
		for attempt in range(200):
			h = (idx * golden_ratio + attempt * 0.01) % 1.0
			rgb = _hsv_to_rgb_uint8(h, s=0.85, v=0.95)
			# Avoid very dark colors and duplicates (after rounding).
			if rgb in seen:
				continue
			if max(rgb) < 60:
				continue
			seen.add(rgb)
			colors.append(rgb)
			break
		else:
			raise RuntimeError("Failed to generate a distinct color palette")

	return colors


def cityscapes_palette() -> np.ndarray:
	"""Return the official Cityscapes labelId->color palette.

	The `*_gtFine_color.png` files are generated from these colors. Using the
	same palette for `*_gtFine_labelIds.png` produces a matching visualization.
	"""
	# RGB colors for labelIds 0..33 from the official Cityscapes scripts.
	label_colors: list[tuple[int, int, int]] = [
		(0, 0, 0),  # 0 unlabeled
		(0, 0, 0),  # 1 ego vehicle
		(0, 0, 0),  # 2 rectification border
		(0, 0, 0),  # 3 out of roi
		(0, 0, 0),  # 4 static
		(111, 74, 0),  # 5 dynamic
		(81, 0, 81),  # 6 ground
		(128, 64, 128),  # 7 road
		(244, 35, 232),  # 8 sidewalk
		(250, 170, 160),  # 9 parking
		(230, 150, 140),  # 10 rail track
		(70, 70, 70),  # 11 building
		(102, 102, 156),  # 12 wall
		(190, 153, 153),  # 13 fence
		(180, 165, 180),  # 14 guard rail
		(150, 100, 100),  # 15 bridge
		(150, 120, 90),  # 16 tunnel
		(153, 153, 153),  # 17 pole
		(153, 153, 153),  # 18 polegroup
		(250, 170, 30),  # 19 traffic light
		(220, 220, 0),  # 20 traffic sign
		(107, 142, 35),  # 21 vegetation
		(152, 251, 152),  # 22 terrain
		(70, 130, 180),  # 23 sky
		(220, 20, 60),  # 24 person
		(255, 0, 0),  # 25 rider
		(0, 0, 142),  # 26 car
		(0, 0, 70),  # 27 truck
		(0, 60, 100),  # 28 bus
		(0, 0, 90),  # 29 caravan
		(0, 0, 110),  # 30 trailer
		(0, 80, 100),  # 31 train
		(0, 0, 230),  # 32 motorcycle
		(119, 11, 32),  # 33 bicycle
	]

	palette = np.zeros((256, 3), dtype=np.uint8)
	palette[: len(label_colors)] = np.array(label_colors, dtype=np.uint8)
	# Cityscapes uses -1 (stored as 255 in uint8) for license plate / ignore.
	palette[255] = (0, 0, 0)
	return palette


def colorize_cityscapes(mask: np.ndarray) -> np.ndarray:
	"""Colorize a Cityscapes labelId mask to RGB."""
	if mask.dtype.kind not in ("u", "i"):
		raise ValueError(f"Expected integer mask, got dtype={mask.dtype}")
	if np.max(mask) > 255 or np.min(mask) < 0:
		raise ValueError("Cityscapes labelIds are expected in range [0,255]")
	palette = cityscapes_palette()
	mask_u8 = mask.astype(np.uint8, copy=False)
	rgb = palette[mask_u8]
	return rgb


def _instance_id_to_rgb(instance_id: int) -> tuple[int, int, int]:
	"""Map an instance id to a deterministic bright RGB color."""
	if instance_id <= 0:
		return 0, 0, 0
	# Simple integer hash -> hue.
	h = ((instance_id * 2654435761) & 0xFFFFFFFF) / 2**32
	return _hsv_to_rgb_uint8(h, s=0.9, v=0.95)


def colorize_instances(mask: np.ndarray) -> np.ndarray:
	"""Colorize a mask where pixel values represent instance IDs."""
	if mask.dtype.kind not in ("u", "i"):
		raise ValueError(f"Expected integer mask, got dtype={mask.dtype}")

	unique_ids = np.unique(mask)
	# Build lookup for all ids present (fast for typical PennFudan masks).
	max_id = int(unique_ids.max())
	lookup = np.zeros((max_id + 1, 3), dtype=np.uint8)
	for instance_id in unique_ids:
		instance_int = int(instance_id)
		if instance_int < 0:
			continue
		r, g, b = _instance_id_to_rgb(instance_int)
		lookup[instance_int] = (r, g, b)

	mask_int = mask.astype(np.int64, copy=False)
	mask_int = np.clip(mask_int, 0, max_id)
	rgb = lookup[mask_int]
	return rgb


def detect_dataset(mask_path: str, mask: np.ndarray) -> Dataset:
	"""Heuristically detect dataset type from filename and values."""
	name = os.path.basename(mask_path).lower()
	if "gtfine_color" in name:
		return "cityscapes"
	if "labelids" in name:
		return "cityscapes"
	if name.endswith("_mask.png") or "pedmasks" in mask_path.replace("\\", "/").lower():
		return "pennfudan"

	unique = np.unique(mask)
	if 255 in unique:
		return "cityscapes"
	if unique.max() > 33:
		return "pennfudan"
	# Ambiguous small-id masks: default to PennFudan-style instance coloring.
	return "pennfudan"


def _as_2d(mask: np.ndarray) -> np.ndarray:
	"""Return a 2D view of a mask.

	Cityscapes/PennFudan id masks should be 2D. If a 3D array is provided, the
	first channel is used.
	"""
	if mask.ndim == 2:
		return mask
	if mask.ndim == 3:
		return mask[:, :, 0]
	raise ValueError(f"Unsupported mask shape: {mask.shape}")


def _cityscapes_color_to_labelids_path(mask_path: str) -> str:
	"""Derive the sibling labelIds path from a Cityscapes color mask path."""
	if mask_path.lower().endswith("_gtfine_color.png"):
		return mask_path[: -len("_gtFine_color.png")] + "_gtFine_labelIds.png"
	# Fallback: conservative replacement.
	return mask_path.replace("_gtFine_color.png", "_gtFine_labelIds.png")


def _is_all_black_bgr(mask: np.ndarray) -> bool:
	"""Return True if a BGR/BGRA image is entirely black in the color channels."""
	if mask.ndim != 3 or mask.shape[2] < 3:
		return False
	bgr = mask[:, :, :3]
	return int(bgr.max()) == 0


def show_rgb(rgb: np.ndarray, title: str) -> None:
	"""Display an RGB uint8 image in a window."""
	if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
		raise ValueError(f"Expected RGB uint8 image, got shape={rgb.shape}, dtype={rgb.dtype}")
	bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
	cv2.imshow(title, bgr)
	# Wait for any key press.
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def main() -> None:
	"""Entry point."""
	args = parse_args()
	mask_path: str = args.mask
	dataset: Dataset = args.dataset

	mask = read_mask(mask_path)
	if dataset == "auto":
		dataset = detect_dataset(mask_path, mask)

	if dataset == "cityscapes":
		# Cityscapes provides both labelId masks and already-colored masks.
		# In this repo's environment OpenCV may decode `*_gtFine_color.png` as all
		# zeros; when that happens (or when such a filename is passed), we fall back
		# to the sibling `*_gtFine_labelIds.png` and apply the official palette.
		name_lower = os.path.basename(mask_path).lower()
		if "_gtfine_color.png" in name_lower:
			fallback_path = _cityscapes_color_to_labelids_path(mask_path)
			if os.path.exists(fallback_path):
				mask = read_mask(fallback_path)
				mask2d = _as_2d(mask)
				rgb = colorize_cityscapes(mask2d)
			else:
				# Last resort: show what we loaded.
				rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB) if mask.ndim == 3 else colorize_cityscapes(_as_2d(mask))
		elif mask.ndim == 3 and mask.shape[2] >= 3:
			if _is_all_black_bgr(mask):
				# Try labelIds fallback if there is a sibling file.
				fallback_path = _cityscapes_color_to_labelids_path(mask_path)
				if os.path.exists(fallback_path):
					mask = read_mask(fallback_path)
					rgb = colorize_cityscapes(_as_2d(mask))
				else:
					rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
			else:
				# OpenCV loads as BGR.
				rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
		else:
			rgb = colorize_cityscapes(_as_2d(mask))
	elif dataset == "pennfudan":
		# Handles both binary and instance-id masks.
		rgb = colorize_instances(_as_2d(mask))
	else:
		raise ValueError(f"Unsupported dataset: {dataset}")

	if dataset == "cityscapes" and int(rgb.max()) == 0:
		try:
			ids = np.unique(_as_2d(mask))
			ids_preview = ", ".join(str(int(v)) for v in ids[:20])
			if ids.size > 20:
				ids_preview += ", ..."
			print(
				"[display_mask] Cityscapes visualization is all black. "
				"This usually means the mask contains only void/ROI labels (e.g., 0..3), "
				"which is common for the Cityscapes test split. "
				f"Unique ids: [{ids_preview}]\n"
				"Try a mask from `gtFine/train` or `gtFine/val` (or pass a `*_gtFine_labelIds.png` from there)."
			)
		except Exception:
			pass

	title = args.title
	if title is None:
		title = f"{dataset}: {os.path.basename(mask_path)}"
	show_rgb(rgb, title=title)


if __name__ == "__main__":
	main()
