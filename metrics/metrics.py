"""
This file contains the metrics and loss function used for training DeepLabV3+ on the people_segmentation dataset.

"""

import numpy as np
import tensorflow as tf
from keras.layers import Flatten
from keras import backend as K

smooth = 1e-15


""" Adjusted iou function because the previous one did not work """
# iou() used tf.numpy_function
# yields unknown type
# better to be replacede with a pure-Tensorflow implementation
def iou(y_true, y_pred):
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

""" Default metrics and loss function """
# Common metric for segmentation tasks
# def iou(y_true, y_pred): 
#     def f(y_true, y_pred):
#         intersection = (y_true * y_pred).sum()
#         union = y_true.sum() + y_pred.sum() - intersection
        
#         x = (intersection + smooth) / (union + smooth)
#         x = x.astype(np.float32)
        
#         return x
#     return tf.numpy_function(f, [y_true, y_pred], tf.float32) 

# Calculates performance of segmentation tasks
def dice_coef(y_true, y_pred): 
    # Flatten (Stejná funkcionalita)
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Main loss function
def dice_loss(y_true, y_pred): 
    return 1.0 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred):
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)