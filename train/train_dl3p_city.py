"""
This file contains the training code for DeepLabV3+ on the Cityscapes dataset.
"""

import os
import sys

# Add parent directory to path to enable imports
# Allows access to the models and metrics modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

# If GPU 0 is busy, use GPU 1 for training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
W = 1024


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
	x_shuffled, y_shuffled = shuffle(list(x), list(y), random_state=42) # type: ignore
	return list(x_shuffled), list(y_shuffled) # type: ignore


def load_data(path: str) -> tuple[list[str], list[str]]:
	"""Load image and mask paths for a split.

	Args:
		path: Split root directory containing image/ and mask/ subfolders.

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
	batch: int = 8,
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


def main() -> None:
	"""Train DeepLabV3+ on Cityscapes."""
	""" Seeding """
	np.random.seed(42)
	tf.random.set_seed(42)

	""" Hyperparameters """
	batch_size = 8
	lr = 1e-3
	num_epochs = 100

	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	dataset_path = os.path.join(project_root, "data", "cityscapes", "new_data")
	runs_root = os.path.join(project_root, "runs", "dl3p_city")
	create_dir(runs_root)

	model_path = os.path.join(runs_root, "deeplabv3_plus.h5")
	csv_path = os.path.join(runs_root, "training_log.csv")
	tensor_logs = os.path.join(runs_root, "tensor_logs")
	create_dir(tensor_logs)

	""" Dataset """
	print(f"Dataset path: {dataset_path}")
	train_path = os.path.join(dataset_path, "train")
	val_path = os.path.join(dataset_path, "valid")

	train_x, train_y = load_data(train_path)
	train_x, train_y = shuffling(train_x, train_y)

	val_x, val_y = load_data(val_path)

	print(f"Training samples: {len(train_x)} | {len(train_y)}")
	print(f"Validation samples: {len(val_x)} | {len(val_y)}")

	train_dataset = tf_dataset(train_x, train_y, batch_size)
	val_dataset = tf_dataset(val_x, val_y, batch_size)

	""" Model """
	model = deeplabv3_plus((H, W, 3))
	model.compile(
		loss=combined_loss,
		optimizer=Adam(lr),
		metrics=[dice_coef, iou, Recall(), Precision()],
	)

	""" Callbacks """
	callbacks = [
		ModelCheckpoint(model_path, verbose=1, save_best_only=True),
		ReduceLROnPlateau(
			monitor="val_loss",
			factor=0.5,
			patience=5,
			min_lr=5e-5,
			verbose=1,
		),
		CSVLogger(csv_path),
		# TensorBoard(log_dir=tensor_logs),
		EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=False),
	]

	""" Training """
	model.fit(
		train_dataset,
		epochs=num_epochs,
		validation_data=val_dataset,
		callbacks=callbacks,
	)

	return


if __name__ == "__main__":
	main()
