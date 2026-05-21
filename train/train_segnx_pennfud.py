"""
This file contains the training code for SegNeXt on the Penn Fudan dataset.
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

import albumentations as A
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
from models.segnext import segnext
from metrics.metrics import dice_coef, iou, combined_loss

""" Global parameters """
H = 512
W = 512


"""Albumentations augmentation policy.

Applied on-the-fly to the training split only. Augmentations are never written
to disk; decoded samples are cached in RAM and augmented each time they are
iterated.
"""
transform = A.Compose( 
	[
		A.AdditiveNoise(
			noise_type="gaussian",
			noise_params={"mean_range": (0, 0), "std_range": (0.05, 0.15)},
			p=0.20,
		),
		A.RandomBrightnessContrast(
			brightness_limit=(-0.20, 0.20),
			contrast_limit=(-0.20, 0.20),
			p=0.30,
		),
		A.HorizontalFlip(p=0.50),
		A.Affine(
			scale=(0.90, 1.10),
			translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
			rotate=(-10, 10),
			shear={"x": (-5, 5), "y": (-5, 5)},
			interpolation=cv2.INTER_LINEAR,
			mask_interpolation=cv2.INTER_NEAREST,
			border_mode=cv2.BORDER_CONSTANT,
			fill=0,
			fill_mask=0,
			p=0.30,
		),
		A.ColorJitter(
			brightness=(0.20),
			contrast=(0.20),
			saturation=(0.20),
			hue=(0.10),
			p=0.30,
		),
	]
)


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


def augment_np(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Apply Albumentations to one image/mask pair.

	Args:
		image: Float32 image in [0, 1] with shape (H, W, 3).
		mask: Float32 mask with shape (H, W, 1) and values in {0, 1}.

	Returns:
		Tuple of (image, mask) as float32 with shapes (H, W, 3) and (H, W, 1).
	"""
	# Albumentations generally expects uint8 images. Keep masks categorical.
	image_u8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
	mask_2d = (np.squeeze(mask, axis=-1) > 0.5).astype(np.uint8)

	augmented = transform(image=image_u8, mask=mask_2d)
	image_aug = augmented["image"].astype(np.float32) / 255.0
	mask_aug = (augmented["mask"] > 0).astype(np.float32)
	mask_aug = np.expand_dims(mask_aug, axis=-1)
	return image_aug, mask_aug


def tf_augment(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
	"""Wrap Albumentations augmentation for use in `tf.data`.

	Uses a NumPy-based augmentation boundary, then re-attaches static shapes.

	Args:
		image: Float32 image tensor with shape (H, W, 3).
		mask: Float32 mask tensor with shape (H, W, 1).

	Returns:
		Augmented (image, mask) tensors with fixed shapes.
	"""
	image, mask = tf.numpy_function(augment_np, [image, mask], [tf.float32, tf.float32])
	image.set_shape([H, W, 3])
	mask.set_shape([H, W, 1])
	return image, mask


def tf_dataset(
	x: Sequence[str],
	y: Sequence[str],
	batch: int = 8,
	augment: bool = False,
) -> tf.data.Dataset:
	"""Build a TensorFlow dataset from image and mask paths.

	Args:
		x: Image paths.
		y: Mask paths.
		batch: Batch size.
		augment: Whether to apply Albumentations to the dataset.

	Returns:
		Prepared TensorFlow dataset.
	"""
	dataset = tf.data.Dataset.from_tensor_slices((list(x), list(y)))
	dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
	# Cache decoded samples in RAM. Augmentations are applied after this boundary.
	dataset = dataset.cache()
	if augment:
		dataset = dataset.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.batch(batch)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)

	return dataset


def main() -> None:
	"""Train SegNeXt on Penn Fudan."""
	""" Seeding """
	np.random.seed(42)
	tf.random.set_seed(42)

	""" Hyperparameters """
	batch_size = 8
	lr = 1e-3
	num_epochs = 100

	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	dataset_path = os.path.join(project_root, "data", "penn_fudan", "new_data")
	runs_root = os.path.join(project_root, "runs", "segnx_pennfud")
	create_dir(runs_root)

	model_path = os.path.join(runs_root, "segnext.weights.h5")
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

	train_dataset = tf_dataset(train_x, train_y, batch_size, augment=True)
	val_dataset = tf_dataset(val_x, val_y, batch_size, augment=False)

	""" Model """
	model = segnext((H, W, 3))
	model.compile(
		loss=combined_loss,
		optimizer=Adam(lr),
		metrics=[dice_coef, iou, Recall(), Precision()],
	)

	""" Callbacks """
	callbacks = [
		ModelCheckpoint(model_path, 
			verbose=1, 
			save_best_only=True,
            save_weights_only=True
        ),
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
