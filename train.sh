#!/bin/bash
# This script is for training an image diffusion model on GF3 dataset
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_P2P_DISABLE=1
export OPENAI_LOGDIR='log/south_pole/03.27'
MODEL_FLAGS='--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma False'
DIFFUSION_FLAGS='--diffusion_steps 2000 --noise_schedule linear'
TRAIN_FLAGS='--lr 1e-4 --batch_size 6'
mpiexec -n 1 python scripts/image_train.py --data_dir_sar './datasets/south_pole/03.27/sar/train' --data_dir_opt './datasets/south_pole/03.27/nac/train'  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS