"""
This file contains the training code for DeepLabV3+ on the PennFudan dataset
using 5-fold cross-validation.
"""

import os
import sys

# Add parent directory to path to enable imports
# Allows to reach the models and metrics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

# If GPU 0 is busy, use GPU 1 for training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import gc
from glob import glob
from typing import Sequence

import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import (  # type: ignore
	ModelCheckpoint,
	CSVLogger,
	ReduceLROnPlateau,
	EarlyStopping,
	TensorBoard,
)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.metrics import Recall, Precision  # type: ignore
from models.deeplabv3_plus import deeplabv3_plus
from metrics.metrics import dice_coef, iou, combined_loss

""" Global parameters """
H = 512
W = 512
FOLD_COUNT = 5


def create_dir(path: str) -> None:
	"""Create a directory if it does not exist.

	Args:
		path: Directory path to create.
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def shuffling(x: Sequence[str], y: Sequence[str]) -> tuple[list[str], list[str]]:
	"""Shuffle paired lists with a fixed random seed.

	Args:
		x: Image paths.
		y: Mask paths.

	Returns:
		Shuffled image and mask path lists.
	"""
	x_shuffled, y_shuffled = shuffle(list(x), list(y), random_state=42)  # type: ignore
	return list(x_shuffled), list(y_shuffled) # type: ignore


def load_data(path: str) -> tuple[list[str], list[str]]:
	"""Load image and mask paths for a fold.

	Args:
		path: Fold root directory containing image/ and mask/ subfolders.

	Returns:
		Sorted lists of image and mask paths.
	"""
	x = sorted(glob(os.path.join(path, "image", "*.png")))
	y = sorted(glob(os.path.join(path, "mask", "*.png")))

	return x, y


def read_image(path: bytes) -> np.ndarray:
	"""Read and normalize an RGB image.

	Args:
		path: Image file path as bytes.

	Returns:
		Float32 image array in [0, 1].
	"""
	path_str = path.decode()
	x = cv2.imread(path_str, cv2.IMREAD_COLOR)
	x = x / 255.0  # type: ignore
	x = x.astype(np.float32)
	return x


def read_mask(path: bytes) -> np.ndarray:
	"""Read a grayscale mask and add a channel dimension.

	Args:
		path: Mask file path as bytes.

	Returns:
		Float32 mask array with shape (H, W, 1).
	"""
	path_str = path.decode()
	y = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
	y = y.astype(np.float32)  # type: ignore
	y = np.expand_dims(y, axis=-1)
	return y


def tf_parse(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
	"""Parse paths into tensors using numpy-based loaders.

	Args:
		x: Image path tensor.
		y: Mask path tensor.

	Returns:
		Image and mask tensors with fixed shapes.
	"""

	def _parse(x_value, y_value):
		x_out = read_image(x_value)
		y_out = read_mask(y_value)
		return x_out, y_out

	x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
	x.set_shape([H, W, 3])
	y.set_shape([H, W, 1])
	return x, y


def tf_dataset(
	x: Sequence[str],
	y: Sequence[str],
	batch: int = 2,
) -> tf.data.Dataset:
	"""Build a TensorFlow dataset from image and mask paths.

	Args:
		x: Image paths.
		y: Mask paths.
		batch: Batch size.

	Returns:
		Prepared TensorFlow dataset.
	"""
	dataset = tf.data.Dataset.from_tensor_slices((list(x), list(y)))
	dataset = dataset.map(tf_parse)
	dataset = dataset.batch(batch)
	dataset = dataset.prefetch(10)

	return dataset


def get_fold_paths(dataset_root: str, fold_count: int) -> list[str]:
	"""Collect fold directories under the dataset root.

	Args:
		dataset_root: Root directory containing fold_* subdirectories.
		fold_count: Number of folds to collect.

	Returns:
		List of fold directory paths.
	"""
	return [os.path.join(dataset_root, f"fold_{i}") for i in range(fold_count)]


def validate_fold_paths(fold_paths: Sequence[str]) -> None:
	"""Ensure each fold directory contains image and mask subfolders.

	Args:
		fold_paths: Fold directory paths.
	"""
	for fold_path in fold_paths:
		image_dir = os.path.join(fold_path, "image")
		mask_dir = os.path.join(fold_path, "mask")
		if not os.path.isdir(image_dir):
			raise ValueError(f"Missing image directory: {image_dir}")
		if not os.path.isdir(mask_dir):
			raise ValueError(f"Missing mask directory: {mask_dir}")


def get_final_metric(history: tf.keras.callbacks.History, key: str) -> float:
	"""Fetch the last value for a metric from training history.

	Args:
		history: Keras training history.
		key: Metric key to retrieve.

	Returns:
		Final metric value or NaN if unavailable.
	"""
	values = history.history.get(key)
	if not values:
		print(f"Warning: Missing history for {key}")
		return float("nan")
	return float(values[-1])


def write_metrics_summary(
	output_path: str,
	rows: Sequence[dict[str, float | str]],
	metric_keys: Sequence[str],
) -> None:
	"""Write per-fold metrics summary to CSV.

	Args:
		output_path: Output CSV file path.
		rows: Metrics rows including the fold identifier.
		metric_keys: Ordered list of metric keys to include.
	"""
	fieldnames = ["fold", *metric_keys]
	with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def main() -> None:
	"""Train DeepLabV3+ with 5-fold cross-validation on PennFudan."""
	""" Seeding """
	np.random.seed(42)
	tf.random.set_seed(42)

	""" Hyperparameters """
	batch_size = 2
	lr = 1e-4
	num_epochs = 20

	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	dataset_root = os.path.join(project_root, "data", "penn_fudan", "new_data")
	runs_root = os.path.join(project_root, "runs", "dl3p_pennfud")
	create_dir(runs_root)

	fold_paths = get_fold_paths(dataset_root, FOLD_COUNT)
	validate_fold_paths(fold_paths)
	print(f"Dataset root: {dataset_root}")

	metrics_rows: list[dict[str, float | str]] = []
	metric_keys = [
		"val_loss",
		"val_dice_coef",
		"val_iou",
		"val_recall",
		"val_precision",
	]

	for fold_index in range(FOLD_COUNT):
		np.random.seed(42)
		tf.random.set_seed(42)

		train_x: list[str] = []
		train_y: list[str] = []
		val_x: list[str] = []
		val_y: list[str] = []

		for index, fold_path in enumerate(fold_paths):
			fold_x, fold_y = load_data(fold_path)
			if index == fold_index:
				val_x, val_y = fold_x, fold_y
			else:
				train_x.extend(fold_x)
				train_y.extend(fold_y)

		train_x, train_y = shuffling(train_x, train_y)

		print(f"Fold {fold_index} training samples: {len(train_x)} | {len(train_y)}")
		print(f"Fold {fold_index} validation samples: {len(val_x)} | {len(val_y)}")

		train_dataset = tf_dataset(train_x, train_y, batch_size)
		val_dataset = tf_dataset(val_x, val_y, batch_size)

		fold_dir = os.path.join(runs_root, f"fold_{fold_index}")
		create_dir(fold_dir)

		model_path = os.path.join(fold_dir, "deeplabv3_plus.h5")
		csv_path = os.path.join(fold_dir, "training_log.csv")
		tensor_logs = os.path.join(fold_dir, "tensor_logs")
		create_dir(tensor_logs)

		model = deeplabv3_plus((H, W, 3))
		model.compile(
			loss=combined_loss,
			optimizer=Adam(lr),
			metrics=[dice_coef, iou, Recall(), Precision()],
		)

		callbacks = [
			ModelCheckpoint(model_path, verbose=1, save_best_only=True),
			ReduceLROnPlateau(
				monitor="val_loss",
				factor=0.1,
				patience=5,
				min_lr=1e-7,
				verbose=1,
			),
			CSVLogger(csv_path),
			# TensorBoard(log_dir=tensor_logs),
			EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=False),
		]

		history = model.fit(
			train_dataset,
			epochs=num_epochs,
			validation_data=val_dataset,
			callbacks=callbacks,
		)

		metrics_rows.append(
			{
				"fold": f"fold_{fold_index}",
				"val_loss": get_final_metric(history, "val_loss"),
				"val_dice_coef": get_final_metric(history, "val_dice_coef"),
				"val_iou": get_final_metric(history, "val_iou"),
				"val_recall": get_final_metric(history, "val_recall"),
				"val_precision": get_final_metric(history, "val_precision"),
			}
		)

		tf.keras.backend.clear_session()
		del model
		gc.collect()

	average_row: dict[str, float | str] = {"fold": "average"}
	for key in metric_keys:
		values = [row[key] for row in metrics_rows if isinstance(row.get(key), float)]
		average_row[key] = float(np.nanmean(values)) if values else float("nan") # type: ignore

	metrics_rows.append(average_row)
	summary_path = os.path.join(runs_root, "metrics_summary.csv")
	write_metrics_summary(summary_path, metrics_rows, metric_keys)

	return


if __name__ == "__main__":
	main()
