"""
This file contains the evaluation code for SegFormer on the Cityscapes dataset.
"""

import os
import sys

# Add parent directory to path to enable imports
# Allows to reach the metrics and train modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# If GPU 0 is busy, use GPU 1 for training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	jaccard_score,
	precision_score,
	recall_score,
)
from tqdm import tqdm

from metrics.metrics import dice_coef, dice_loss, iou
from models.segformer import segformer_b0
from train.train_segf_city import load_data

""" Swapping models and datasets """
# 1) Global parameters
# 2) Cityscapes Testing
# 3) Loading Data

""" Global parameters """
# H = 512
# W = 1024
ZERO_DIVISION = 0

""" Global params (Penn Fudan)"""
H = 512
W = 512

# Get the project root directory (parent of the train folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cityscapes Testing
model_path = os.path.join(project_root, "runs", "segf_city", "segformer_b0.h5")
# results_root = os.path.join(project_root, "results", "segf_city")

# Pennfudan Testing
# model_path = os.path.join(project_root, "runs", "segf_pennfud", "segformer_b0.h5")
results_root = os.path.join(project_root, "results", "segf_pennfud")


""" Directory Creation """
def create_dir(path: str) -> None:
	"""Create a directory if it does not exist.

	Args:
		path: Directory path to create.
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def save_results(
	image: np.ndarray,
	mask: np.ndarray,
	y_pred: np.ndarray,
	save_path: str,
) -> None:
	"""Save visualization tiles for image, mask, prediction, and masked image.

	Args:
		image: RGB image array.
		mask: Binary mask array.
		y_pred: Binary prediction array.
		save_path: Output image path.
	"""
	line = np.ones((H, 10, 3)) * 128  # Grey image

	mask = np.expand_dims(mask, axis=-1)
	mask = np.concatenate([mask, mask, mask], axis=-1)
	mask = mask * 255

	y_pred = np.expand_dims(y_pred, axis=-1)
	y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

	masked_image = image * y_pred
	y_pred = y_pred * 255

	cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
	cv2.imwrite(save_path, cat_images)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
	"""Convert mask to binary values {0,1}.

	Args:
		mask: Grayscale mask array.

	Returns:
		Binary mask array with values {0,1}.
	"""
	return (mask > 0).astype(np.int32)


if __name__ == "__main__":
	""" Seeding """
	np.random.seed(42)
	tf.random.set_seed(42)

	create_dir(results_root)

	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model weights not found: {model_path}")

	""" Loading model """
	model: Model = segformer_b0((H, W, 3))
	model.load_weights(model_path)

	""" Loading data """
	# dataset_path = os.path.join(project_root, "data", "cityscapes", "new_data")
	dataset_path = os.path.join(project_root, "data", "penn_fudan", "new_data")

	print(f"Dataset path: {dataset_path}")

	# test_path = os.path.join(dataset_path, "valid")
	test_path = os.path.join(dataset_path, "test")
	test_x, test_y = load_data(test_path)

	print(f"Test samples: {len(test_x)} | {len(test_y)}")

	""" Evaluation and Prediction """
	SCORE = []

	for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
		""" Name Extraction """
		name = os.path.splitext(os.path.basename(x))[0]

		""" Reading the image """
		image = cv2.imread(x, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError(f"Failed to read image: {x}")
		x_img = image / 255.0
		x_img = np.expand_dims(x_img, axis=0)

		""" Reading the mask """
		mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
		if mask is None:
			raise ValueError(f"Failed to read mask: {y}")
		mask = binarize_mask(mask)

		""" Prediction """
		y_pred = model.predict(x_img)[0]
		y_pred = np.squeeze(y_pred, axis=-1)
		y_pred = y_pred > 0.5
		y_pred = y_pred.astype(np.int32)

		""" Saving the prediction """
		save_image_path = os.path.join(results_root, f"{name}.png")
		save_results(image, mask, y_pred, save_image_path)

		""" Flatten Arrays """
		mask_flat = mask.flatten()
		y_pred_flat = y_pred.flatten()

		""" Metrics Calculation """
		acc_value = accuracy_score(mask_flat, y_pred_flat)
		f1_value = f1_score(
			mask_flat,
			y_pred_flat,
			labels=[0, 1],
			average="binary",
			zero_division=ZERO_DIVISION,
		)
		jac_value = jaccard_score(
			mask_flat,
			y_pred_flat,
			labels=[0, 1],
			average="binary",
			zero_division=ZERO_DIVISION,
		)
		recall_value = recall_score(
			mask_flat,
			y_pred_flat,
			labels=[0, 1],
			average="binary",
			zero_division=ZERO_DIVISION,
		)
		precision_value = precision_score(
			mask_flat,
			y_pred_flat,
			labels=[0, 1],
			average="binary",
			zero_division=ZERO_DIVISION,
		)

		SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

	""" Metrics values """
	score = [s[1:] for s in SCORE]
	score = np.mean(score, axis=0)
	print(f"Accuracy: {score[0]:0.5f}")
	print(f"F1-Score: {score[1]:0.5f}")
	print(f"Jaccard-Score: {score[2]:0.5f}")
	print(f"Recall: {score[3]:0.5f}")
	print(f"Precision: {score[4]:0.5f}")

	df = pd.DataFrame(
		SCORE,
		columns=["Name", "Accuracy", "F1-Score", "Jaccard-Score", "Recall", "Precision"],
	)
	df.to_csv(os.path.join(results_root, "metrics.csv"), index=False)
