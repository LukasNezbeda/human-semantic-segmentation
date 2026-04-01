"""
This file contains the training code for DeepLabV3+ on the people_segmentation dataset.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.metrics import Recall, Precision
from models.deeplabv3_plus import deeplabv3_plus
from metrics.metrics import dice_loss, dice_coef, iou, combined_loss

""" Global parameters """
H = 512
W = 512

""" Directory creation """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def shuffling(x, y):
    # Might sound "None" is not iterable
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.png")))
    y = sorted(glob(os.path.join(path, "mask", "*.png")))

    return x, y

def read_image(path):
    path = path.decode() # Convert bytes to string
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x


# Normalizing depends on mask format, binary mask works for values (0 to 255) which does require it
def read_mask(path):
    path = path.decode() # Convert bytes to string
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    y = y.astype(np.float32)
    y = np.expand_dims(y, axis=-1) # Add channel dimension
    return y

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    # read_image and read_mask are outside of tensorflow environment
    # tf.numpy_function is needed to wrap a python function and use it as Tensorflow op
    # May sound that it is not iterable
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

# Formatting dataset to be tensorflow compatible
def tf_dataset(x, y, batch=2):
    datasset = tf.data.Dataset.from_tensor_slices((x, y))
    datasset = datasset.map(tf_parse)
    dataset = datasset.batch(batch)
    dataset = dataset.prefetch(10)
    
    return dataset

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # """ Directory for storing files """
    # create_dir("files")
    
    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 20
    model_path = os.path.join("..", "models", "deeplabv3_plus.h5")
    csv_path = os.path.join("..", "runs", "training_log.csv")
    
    tensor_logs = os.path.join("..", "runs", "tensor_logs")
    create_dir(tensor_logs)
    
    """ Dataset"""
    dataset_path = "new_data"
    
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "test")
    
    # Load and shuffle training data
    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    
    # No need to shuffle validation data
    val_x, val_y = load_data(val_path)
    
    # Might sound that it cannot be sized
    print(f"Training samples: {len(train_x)} | {len(train_y)}")
    print(f"Validation samples: {len(val_x)} | {len(val_y)}")
    
    train_dataset = tf_dataset(train_x, train_y, batch_size)
    val_dataset = tf_dataset(val_x, val_y, batch_size)
    
    """ DEBUG: Check dataset shape """
    # for x, y in train_dataset:
        # (batch, height, width, channels) (batch, height, width, channels)
        # print(x.shape, y.shape)
        
        # break
        
    """ Model """
    model = deeplabv3_plus((H, W, 3))
    model.compile(loss=combined_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])
    
    """ Callbacks """
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        # TensorBoard(log_dir=tensor_logs),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=False),
        # TODO: Callback pro vizualizaci vstupu, predikce a ground truth v každé epoše
    ]
    
    """ Training """
    model.fit(
        train_dataset, 
        epochs=num_epochs,
        validation_data=val_dataset, 
        callbacks=callbacks)