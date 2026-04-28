"""Evaluation for DeepLabV3+ on the Cityscapes dataset.

Evaluates a trained DeepLabV3+ model on the prepared Cityscapes test split and
saves qualitative prediction tiles plus a CSV of per-image metrics.
"""

import os
import sys

# Add parent directory to path to enable imports
# Allows to reach the metrics and train modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from sklearn.metrics import (
	accuracy_score,
	f1_score,
	jaccard_score,
	precision_score,
	recall_score,
)
from tqdm import tqdm

from metrics.metrics import dice_coef, dice_loss, iou
from models.deeplabv3_plus import deeplabv3_plus
from train.train_dl3p_city import load_data


""" Global parameters """
H = 512
W = 1024

# Get the project root directory (parent of the train folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_root = os.path.join(project_root, "data", "cityscapes", "new_data")
runs_root = os.path.join(project_root, "runs", "dl3p_city")
model_path = os.path.join(runs_root, "deeplabv3_plus.h5")
results_root = os.path.join(project_root, "results", "dl3p_city")


""" Directory Creation """
def create_dir(path: str) -> None:
	"""Create a directory if it does not exist.

	Args:
		path: Directory path to create.
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def save_results(image: np.ndarray, mask: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
	"""Save visualization tiles for image, mask, prediction, and masked image.

	Args:
		image: RGB image array.
		mask: Binary mask array with values {0,1}.
		y_pred: Binary prediction array with values {0,1}.
		save_path: Output image path.
	"""
	line = np.ones((H, 10, 3)) * 128  # Grey image

	mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)
	mask = np.concatenate([mask, mask, mask], axis=-1)  # (H, W, 3)
	mask = mask * 255

	y_pred = np.expand_dims(y_pred, axis=-1)  # (H, W, 1)
	y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  # (H, W, 3)

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

	""" Storing files """
	create_dir(results_root)

	print(f"Dataset path: {dataset_root}")
	print(f"Runs path: {runs_root}")
	print(f"Results path: {results_root}")

	if not os.path.exists(model_path):
		raise FileNotFoundError(
			"Missing Cityscapes model weights at: "
			f"{model_path}. "
			"Expected from training script output in runs/dl3p_city. "
			"If you trained elsewhere, you may have weights under results/, e.g. "
			"results/2026-04-26/dl3p_city/deeplabv3_plus.h5."
		)

	""" Loading model """
	model: Model = deeplabv3_plus((H, W, 3))
	model.load_weights(model_path)

	""" Loading data """
	test_path = os.path.join(dataset_root, "test")
	test_x, test_y = load_data(test_path)
	print(f"Test samples: {len(test_x)} | {len(test_y)}")

	""" Evaluation and Prediction """
	SCORE: list[list[object]] = []

	for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
		""" Name Extraction """
		name = os.path.splitext(os.path.basename(x))[0]

		""" Reading the image """
		image = cv2.imread(x, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError(f"Failed to read image: {x}")
		x_img = image / 255.0  # type: ignore
		x_img = np.expand_dims(x_img, axis=0)

		""" Reading the mask """
		mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
		if mask is None:
			raise ValueError(f"Failed to read mask: {y}")
		mask_bin = binarize_mask(mask)

		""" Prediction """
		y_pred = model.predict(x_img, verbose=0)[0] # type: ignore
		y_pred = np.squeeze(y_pred, axis=-1)
		y_pred = y_pred > 0.5
		y_pred = y_pred.astype(np.int32)

		""" Saving the prediction """
		save_image_path = os.path.join(results_root, f"{name}.png")
		save_results(image, mask_bin, y_pred, save_image_path)

		""" Flatten Arrays """
		mask_flat = mask_bin.flatten()
		y_pred_flat = y_pred.flatten()

		""" Metrics Calculation """
		acc_value = accuracy_score(mask_flat, y_pred_flat)
		f1_value = f1_score(mask_flat, y_pred_flat, labels=[0, 1], average="binary")
		jac_value = jaccard_score(mask_flat, y_pred_flat, labels=[0, 1], average="binary")
		recall_value = recall_score(mask_flat, y_pred_flat, labels=[0, 1], average="binary")
		precision_value = precision_score(mask_flat, y_pred_flat, labels=[0, 1], average="binary")

		SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

	if not SCORE:
		raise RuntimeError(f"No samples found under: {test_path}")

	""" Metrics values """
	score = [s[1:] for s in SCORE]
	score = np.mean(score, axis=0) # type: ignore
	print(f"Accuracy: {score[0]:0.5f}")
	print(f"F1-Score: {score[1]:0.5f}")
	print(f"Jaccard-Score: {score[2]:0.5f}")
	print(f"Recall: {score[3]:0.5f}")
	print(f"Precision: {score[4]:0.5f}")

	df = pd.DataFrame(
		SCORE,
		columns=["Name", "Accuracy", "F1-Score", "Jaccard-Score", "Recall", "Precision"],
	)
	df.to_csv(os.path.join(results_root, "metrics.csv"))

