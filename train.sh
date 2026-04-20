#!/bin/bash

# run experiments
# python -u train/train_deeplabv3_plus.py --dataset penn_fudan > log.txt
python -u train/train_deeplabv3_plus.py --dataset person_segmentation > log.txt

