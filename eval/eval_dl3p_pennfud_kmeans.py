"""
This file contains the evaluation code for DeepLabV3+ on the PennFudan dataset.

Evaluates 5-fold cross-validation models on their corresponding fold data.
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

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model #type: ignore
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics.metrics import dice_loss, dice_coef, iou
from models.deeplabv3_plus import deeplabv3_plus
from train.train_dl3p_pennfud_kmeans import load_data


""" Global parameters """
H = 512
W = 512
FOLD_COUNT = 5

# Get the project root directory (parent of the train folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_root = os.path.join(project_root, "data", "penn_fudan", "new_data_kmeans")
runs_root = os.path.join(project_root, "runs", "dl3p_pennfud")
results_root = os.path.join(project_root, "results", "dl3p_pennfud")


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
		mask: Binary mask array.
		y_pred: Binary prediction array.
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

	create_dir(results_root)

	print(f"Dataset path: {dataset_root}")
	print(f"Runs path: {runs_root}")
	print(f"Results path: {results_root}")

	summary_rows = []

	for fold_index in range(FOLD_COUNT):
		""" Loading model """
		fold_dir = os.path.join(runs_root, f"fold_{fold_index}")
		model_path = os.path.join(fold_dir, "deeplabv3_plus.h5")
		if not os.path.exists(model_path):
			raise FileNotFoundError(
				f"Missing model weights for fold {fold_index}: {model_path}"
			)

		model: Model = deeplabv3_plus((H, W, 3))
		model.load_weights(model_path)

		""" Loading data """
		fold_data_path = os.path.join(dataset_root, f"fold_{fold_index}")
		test_x, test_y = load_data(fold_data_path)
		print(f"Fold {fold_index} samples: {len(test_x)} | {len(test_y)}")

		""" Evaluation and Prediction """
		score_rows = []
		fold_results_dir = os.path.join(results_root, f"fold_{fold_index}")
		create_dir(fold_results_dir)

		for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
			""" Name Extraction """
			name = os.path.splitext(os.path.basename(x))[0]

			""" Reading the image """
			image = cv2.imread(x, cv2.IMREAD_COLOR)
			x_img = image / 255.0  # type: ignore
			x_img = np.expand_dims(x_img, axis=0)

			""" Reading the mask """
			mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
			mask = binarize_mask(mask) #type: ignore

			""" Prediction """
			y_pred = model.predict(x_img)[0]
			y_pred = np.squeeze(y_pred, axis=-1)
			y_pred = y_pred > 0.5
			y_pred = y_pred.astype(np.int32)

			""" Saving the prediction """
			save_image_path = os.path.join(fold_results_dir, f"{name}.png")
			save_results(image, mask, y_pred, save_image_path) #type: ignore

			""" Flatten Arrays """
			mask_flat = mask.flatten()
			y_pred_flat = y_pred.flatten()

			""" Metrics Calculation """
			acc_value = accuracy_score(mask_flat, y_pred_flat)
			f1_value = f1_score(mask_flat, y_pred_flat, labels=[0, 1], average="binary")
			jac_value = jaccard_score(mask_flat, y_pred_flat, labels=[0, 1], average="binary")
			recall_value = recall_score(mask_flat, y_pred_flat, labels=[0, 1], average="binary")
			precision_value = precision_score(mask_flat, y_pred_flat, labels=[0, 1], average="binary")

			score_rows.append(
				[name, acc_value, f1_value, jac_value, recall_value, precision_value]
			)

		""" Metrics values """
		score = [s[1:] for s in score_rows]
		score = np.mean(score, axis=0)
		print(f"Fold {fold_index} Accuracy: {score[0]:0.5f}")
		print(f"Fold {fold_index} F1-Score: {score[1]:0.5f}")
		print(f"Fold {fold_index} Jaccard-Score: {score[2]:0.5f}")
		print(f"Fold {fold_index} Recall: {score[3]:0.5f}")
		print(f"Fold {fold_index} Precision: {score[4]:0.5f}")

		df = pd.DataFrame(
			score_rows,
			columns=["Name", "Accuracy", "F1-Score", "Jaccard-Score", "Recall", "Precision"],
		)
		df.to_csv(os.path.join(fold_results_dir, "metrics.csv"))

		summary_rows.append(
			{
				"fold": f"fold_{fold_index}",
				"accuracy": float(score[0]),
				"f1_score": float(score[1]),
				"jaccard_score": float(score[2]),
				"recall": float(score[3]),
				"precision": float(score[4]),
			}
		)

	summary_df = pd.DataFrame(summary_rows)
	avg_row = {
		"fold": "average",
		"accuracy": float(summary_df["accuracy"].mean()),
		"f1_score": float(summary_df["f1_score"].mean()),
		"jaccard_score": float(summary_df["jaccard_score"].mean()),
		"recall": float(summary_df["recall"].mean()),
		"precision": float(summary_df["precision"].mean()),
	}
	summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)
	summary_df.to_csv(os.path.join(results_root, "metrics_summary.csv"))
