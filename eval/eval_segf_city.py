"""
This file contains the evaluation code for SegFormer on the Cityscapes dataset.
"""

import os
import sys
import time

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
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
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


def create_frozen_inference_function(model: Model) -> tf.types.experimental.ConcreteFunction:
	@tf.function
	def tf_func_call(inp: tf.Tensor) -> tf.Tensor:
		return model(inp, training=False)

	input_tensor_spec = tf.TensorSpec(shape=(None, H, W, 3), dtype=tf.float32)
	concrete_function = tf_func_call.get_concrete_function(input_tensor_spec) # type: ignore
	return convert_variables_to_constants_v2(concrete_function) # type: ignore


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

	frozen_func = create_frozen_inference_function(model)

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
	inference_times = []

	for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
		""" Name Extraction """
		name = os.path.splitext(os.path.basename(x))[0]

		""" Reading the image """
		image = cv2.imread(x, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError(f"Failed to read image: {x}")
		# Normalize and ensure float32 for TensorFlow
		x_img = (image / 255.0).astype(np.float32)
		x_img = np.expand_dims(x_img, axis=0)
		x_tensor = tf.convert_to_tensor(x_img)

		""" Reading the mask """
		mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
		if mask is None:
			raise ValueError(f"Failed to read mask: {y}")
		mask = binarize_mask(mask)

		""" Prediction """
		# Measure model forward-pass latency for this image.
		inference_start = time.perf_counter()
		try:
			out = frozen_func(x_tensor)
			if isinstance(out, (list, tuple)):
				y_pred = out[0].numpy()
			else:
				y_pred = out.numpy()
		except Exception:
			y_pred = model.predict(x_img, verbose=0)[0]
		inference_time = time.perf_counter() - inference_start
		inference_times.append(inference_time)
		if isinstance(y_pred, np.ndarray) and y_pred.ndim == 4 and y_pred.shape[0] == 1:
			y_pred = y_pred[0]
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

		SCORE.append([
			name, 
			acc_value, 
			f1_value, 
			jac_value, 
			recall_value, 
			precision_value,
			inference_time
		])

	""" Metrics values """
	score = [s[1:6] for s in SCORE]
	score = np.mean(score, axis=0)
	total_inference_time = float(np.sum(inference_times))
	avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0
	avg_inference_time_ms = avg_inference_time * 1000.0
	fps = (1.0 / avg_inference_time) if avg_inference_time > 0 else 0.0
	print(f"Accuracy: {score[0]:0.5f}")
	print(f"F1-Score: {score[1]:0.5f}")
	print(f"Jaccard-Score: {score[2]:0.5f}")
	print(f"Recall: {score[3]:0.5f}")
	print(f"Precision: {score[4]:0.5f}")
	print(f"Total Inference Time (s): {total_inference_time:0.5f}")
	print(f"Average Inference Time per Image (s): {avg_inference_time:0.5f}")
	print(f"Average Inference Time per Image (ms): {avg_inference_time_ms:0.2f}")
	print(f"FPS: {fps:0.2f}")

	df = pd.DataFrame(
		SCORE,
		columns=[
			"Name", 
			"Accuracy", 
			"F1-Score", 
			"Jaccard-Score", 
			"Recall", 
			"Precision",
			"Inference Time (s)"
		],
	)
	df.to_csv(os.path.join(results_root, "metrics.csv"), index=False)
