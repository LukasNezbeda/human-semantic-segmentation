import os
import sys

from models.deeplabv3_plus import deeplabv3_plus

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
from keras.models import Model
from metrics.metrics import dice_loss, dice_coef, iou
from train.deeplabv3_plus.train_deeplabv3_plus import load_data, create_dir

""" Global parameters """
H = 512
W = 512

# Get the project root directory (parent of the train folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(project_root, "runs", "deeplabv3_plus.h5")

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """ Storing files """
    create_dir("data/person_segmentation/test_images/mask")
    
    # """ Loading model (legacy) """
    # with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou': iou}):
    #     model: Model = tf.keras.models.load_model(model_path)

    """ Loading model """
    model: Model = deeplabv3_plus((H, W, 3))
    model.load_weights(model_path)

    """ Load dataset """
    data_x = glob(os.path.join(project_root, "data", "person_segmentation", "test_images", "image", "*.jpg"))

    for path in tqdm(data_x, total=len(data_x)):
        """ Name Extraction """
        name = os.path.splitext(os.path.basename(path))[0]

        """ Reading the image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape # type: ignore
        x = cv2.resize(image, (W, H)) # type: ignore
        x = x / 255.0 # Normalization
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        y = model.predict(x)[0] # type: ignore
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1) # Prediction mask resized to original image

        """ Save the masked image """
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1) # type: ignore
        cv2.imwrite(f"data/person_segmentation/test_images/mask/{name}.png", cat_images)
