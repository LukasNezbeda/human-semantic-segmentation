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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from metrics.metrics import dice_loss, dice_coef, iou
from models.deeplabv3_plus import deeplabv3_plus
from train.deeplabv3_plus.train_dl3p_person_seg import load_data


""" Global parameters """
H = 512
W = 512

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
    y_pred = y_pred * 255 # Scale it to 255 for visualization

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_path, cat_images)

        
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """ Storing files """
    create_dir("results")
    
    # """ Loading model (legacy) """
    # with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou': iou}):
    #     model: Model = tf.keras.models.load_model(model_path)

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
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Name Extraction """
        name = os.path.splitext(os.path.basename(x))[0]
        print(name)

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0 # type: ignore

        # print(x.shape) # Before dimension expansion

        x = np.expand_dims(x, axis=0)

        # print(x.shape) # After dimension expansion

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Prediction """
        y_pred = model.predict(x)[0] # Extract first item from the list
        y_pred = np.squeeze(y_pred, axis=-1) # Squeeze it
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = f"results/{name}.png"
        save_results(image, mask, y_pred, save_image_path)


        break