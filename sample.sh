#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_P2P_DISABLE=1
# 储存采样结果的路径（log.txt, progress.csv, samples.npz）在logger.Configure()时被获取以确认log的存储路径
export OPENAI_LOGDIR='results/south_pole/03.27'
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma False"
DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
SAMPLE_FLAGS="--batch_size 2 --num_samples 256 --timestep_respacing 250"
PATH_FLAGS="--model_path log/south_pole/03.27/model040000.pt --data_path ./datasets/south_pole/03.27/test"  # 要使用的模型路径 要使用的test影像路径
RESULT_FLAGS="--sample_results_folder sample_results/south_pole/03.27"  # 储存采样结果影像的路径
python scripts/image_sample_realtime.py $PATH_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $RESULT_FLAGS