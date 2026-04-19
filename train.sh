#!/bin/bash

# run experiments
python -u train/deeplabv3_plus/train_deeplabv3_plus.py --dataset penn_fudan > log.txt

