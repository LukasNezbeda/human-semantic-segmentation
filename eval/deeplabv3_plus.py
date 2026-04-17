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
from tensorflow.keras.models import load_model, Model # type: ignore
from keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from metrics.metrics import dice_loss, dice_coef, iou
from train.deeplabv3_plus.train_dl3p_person_seg import load_data

""" Global parameters """
H = 512
W = 512

# Get the project root directory (parent of the train folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(project_root, "runs", "deeplabv3_plus.h5")

""" Directory Creation """
def create_dir (path):
    if not os.path.exists(path):
        os.makedirs(path)
        
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    
    """ Storing files """
    create_dir("results")
    
    """ Loading model """
    
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = load_model(model_path, compile=False)
        model.summary()