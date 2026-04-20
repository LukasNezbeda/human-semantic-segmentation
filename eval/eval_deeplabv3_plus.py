"""
This file contains the evaluation code for DeepLabV3+ on the people_segmentation dataset.

"""

import os
import sys

# Add parent directory to path to enable imports
# Allows to reach the metrics and train modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
# from tensorflow.keras.utils import CustomObjectScope # Deprecated
from keras.models import Model
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics.metrics import dice_loss, dice_coef, iou
from models.deeplabv3_plus import deeplabv3_plus
from train.train_deeplabv3_plus import load_data


""" Global parameters """
H = 512
W = 512
USE_PENN_FUDAN_FOLDS = False

# Get the project root directory (parent of the train folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(project_root, "runs", "deeplabv3_plus.h5")

""" Directory Creation """
def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_path):
    ## i - m - yp - yp*i
    # Same dimensions
    # 3 channels

    line = np.ones((H, 10, 3)) * 128 # Grey image

    mask = np.expand_dims(mask, axis=-1) ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1) # (512, 512, 3)
    mask = mask * 255 # Scale it to 255 for visualization

    y_pred = np.expand_dims(y_pred, axis=-1) ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) # (512, 512, 3)

    masked_image = image * y_pred # Image and mask view
    y_pred = y_pred * 255 # Scale it to 255 for visualization

    cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
    cv2.imwrite(save_path, cat_images)


def list_fold_dirs(penn_root: str) -> list[str]:
    """Return sorted fold directories from a PennFudan root."""
    if not os.path.isdir(penn_root):
        return []
    fold_entries: list[tuple[int, str]] = []
    for name in os.listdir(penn_root):
        if not name.startswith("fold_"):
            continue
        fold_path = os.path.join(penn_root, name)
        if not os.path.isdir(fold_path):
            continue
        try:
            fold_index = int(name.split("_", 1)[1])
        except ValueError:
            continue
        fold_entries.append((fold_index, fold_path))
    fold_entries.sort(key=lambda item: item[0])
    return [path for _, path in fold_entries]


def evaluate_dataset(
    model: Model,
    image_paths: list[str],
    mask_paths: list[str],
    results_dir: str,
) -> list[list[float | str]]:
    """Evaluate a model on paired image and mask paths."""
    scores: list[list[float | str]] = []
    create_dir(results_dir)

    for x, y in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        """ Name Extraction """
        name = os.path.splitext(os.path.basename(x))[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0 # type: ignore
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Prediction """
        y_pred = model.predict(x)[0] # Extract first item from the list
        y_pred = np.squeeze(y_pred, axis=-1) # Squeeze it
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = os.path.join(results_dir, f"{name}.png")
        save_results(image, mask, y_pred, save_image_path)

        """ Flatten Arrays """
        mask = mask.flatten() # type: ignore
        y_pred = y_pred.flatten()

        """ Metrics Calculation """
        acc_value = accuracy_score(mask, y_pred)
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average='binary') # 0 for background, 1 for foreground
        jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average='binary')
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average='binary')
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average='binary')

        scores.append([name, acc_value, f1_value, jac_value, recall_value, precision_value]) # type: ignore

    return scores


def mean_scores(scores: list[list[float | str]]) -> np.ndarray:
    """Compute mean metrics from a list of per-image scores."""
    if not scores:
        return np.zeros(5, dtype=np.float32)
    values = [score[1:] for score in scores]
    return np.mean(values, axis=0) # type: ignore

        
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """ Storing files """
    create_dir("results")
    
    # """ Loading model (legacy) """
    # with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou': iou}):
    #     model: Model = tf.keras.models.load_model(model_path)

    penn_fudan_root = os.path.join(project_root, "data", "penn_fudan", "new_data")
    fold_model_root = os.path.join(project_root, "runs", "deeplabv3_plus")

    fold_dirs = list_fold_dirs(penn_fudan_root) if USE_PENN_FUDAN_FOLDS else []
    if fold_dirs:
        fold_scores: list[list[float | str]] = []

        for fold_dir in fold_dirs:
            fold_name = os.path.basename(fold_dir)
            fold_model_path = os.path.join(fold_model_root, fold_name, "deeplabv3_plus.h5")
            if not os.path.exists(fold_model_path):
                print(f"Missing model for {fold_name}: {fold_model_path}")
                continue

            model: Model = deeplabv3_plus((H, W, 3))
            model.load_weights(fold_model_path)

            fold_x, fold_y = load_data(fold_dir)
            print(f"{fold_name} samples: {len(fold_x)} | {len(fold_y)}")
            if not fold_x:
                print(f"Skipping {fold_name}: no samples found")
                continue

            fold_results_dir = os.path.join("results", fold_name)
            scores = evaluate_dataset(model, fold_x, fold_y, fold_results_dir)
            fold_mean = mean_scores(scores)
            fold_scores.append([fold_name, *fold_mean.tolist()])

        if fold_scores:
            df = pd.DataFrame(
                fold_scores,
                columns=["Fold", "Accuracy", "F1-Score", "Jaccard-Score", "Recall", "Precision"],
            )
            mean_row = [
                "average",
                df["Accuracy"].mean(),
                df["F1-Score"].mean(),
                df["Jaccard-Score"].mean(),
                df["Recall"].mean(),
                df["Precision"].mean(),
            ]
            df.loc[len(df)] = mean_row
            df.to_csv("results/metrics.csv", index=False)
        else:
            print("No fold scores computed; falling back to legacy evaluation.")
            fold_dirs = []

    if not fold_dirs:
        """ Loading model """
        model: Model = deeplabv3_plus((H, W, 3))
        model.load_weights(model_path)
        # model.summary() # Checking if model loaded correctly

        """ Loading data """
        dataset_path = os.path.join(project_root, "data", "person_segmentation", "new_data")
        print(f"Dataset path: {dataset_path}")
        
        val_path = os.path.join(dataset_path, "test")
        test_x, test_y = load_data(val_path)

        print(f"Test samples: {len(test_x)} | {len(test_y)}")

        """ Evaluation and Prediction """
        scores = evaluate_dataset(model, test_x, test_y, "results")

        """ Metrics values """
        score = mean_scores(scores)
        print(f"Accuracy: {score[0]:0.5f}")
        print(f"F1-Score: {score[1]:0.5f}")
        print(f"Jaccard-Score: {score[2]:0.5f}")
        print(f"Recall: {score[4]:0.5f}")
        print(f"Precision: {score[3]:0.5f}")

        df = pd.DataFrame(scores, columns=["Name", "Accuracy", "F1-Score", "Jaccard-Score" , "Recall", "Precision"])
        df.to_csv("results/metrics.csv", index=False)

    """ About the CSV file """
    # The CSV file allows you to get insight on how each image in the testing set performed.
    # This shows the capabilities of your model.