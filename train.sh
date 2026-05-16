#!/bin/bash

# run experiments
# python -u train/train_dl3p_pseg.py > log.txt
# python -u train/train_dl3p_pennfud.py > log.txt
# python -u train/train_dl3p_city.py > log.txt
# python -u train/train_segf_city.py > log.txt
# python -u train/train_segnx_city.py > log.txt
python -u train/train_unet3p_city.py > log.txt

