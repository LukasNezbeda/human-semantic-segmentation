"""
This file is a data loader for people_segmentation dataset to be used for DeepLabV3+

It can use improvements
- Data augmentation done in RAM instead of on disk.
- Adjusting filepaths for workspace folder structure.
"""

import os

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, ChannelShuffle, CoarseDropout, CenterCrop, Rotate, GridDistortion, OpticalDistortion



""" Automating Data Preparation """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_data(path, split=0.1):
    # Sorting to avoid misalignment
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))
    
    """ Splitting the dataset """
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y)

    """ DEBUG:  Testing reading images and masks"""
    # for x, y in zip(X, Y):
    #     print(f"Image: {x}, Mask: {y}")
        
    #     x = cv2.imread(x)
    #     cv2.imwrite("sample_image.jpg", x)

    #     y = cv2.imread(y)
    #     cv2.imwrite("sample_mask.png", y)
        
    #     break # To read just one image
        
def augment_data(images, masks, save_path, augment=True):
    # Fixed dataset size for augmentation
    H = 512
    W = 512
    
    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extracting the filename  """
        # Windows handling
        # name = x.split("\\")[-1].split(".") # Extract the last element of the path [filename], [extension] (windows)
        name = x.split("\\")[-1].split(".")[0] # Extract the last element of the path [filename] (windows)
        # print(f"\n{name}")
        
        # Linux handling
        # name = x.split("/")[-1].split(".") # Extract the last element of the path [filename], [extension] (Linux)
        # name = x.split("/")[-1].split(".")[0] # Extract the last element of the path [filename] (Linux)
        # print(name)
        
        """ Reading the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        
        """ Data Augmentation """
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
            
            # Greyscale augmentation
            x2 = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            y2 = y
            
            # Channel shuffling
            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]
            
            # CoarseDropout
            aug = CoarseDropout(
                p=1.0,
                num_holes_range=(5, 10), 
                hole_height_range=(16, 32), 
                hole_width_range=(16, 32))
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]
            
            # Rotation
            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]
            
        else: # Resize and save without augmentation
            X = [x]
            Y = [y]
        
        index = 0
        for i, m in zip(X, Y):
            try:
                """ Center Cropping """ # Avoids compression and information loss
                aug = CenterCrop(H, W, p=1.0)
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]
                
            except Exception as e: # Resize as a fallback if cropping fails
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))
            
            
            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"
        
            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)
            
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
        
            index += 1
        
        """ DEBUG: Testing saving augmented images and masks """
        # break
        
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    
    """ Dataset loading """
    data_path = "people_segmentation"
    load_data(data_path)
    
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    
    print(f"Training samples:\t {len(train_x)} | {len(train_y)}")
    print(f"Testing samples:\t {len(test_x)} | {len(test_y)}")
    
    """ Create directories to save augmented data """
    create_dir("new_data/train/image")
    create_dir("new_data/train/mask")
    create_dir("new_data/test/image")
    create_dir("new_data/test/mask")
    
    """ Data augmentation """
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=True)