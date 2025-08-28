#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=AttentionProfiler_GPU
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=24GB
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate nlp-hw1

cd /Users/prabhavsingh/Documents/CLASSES/Fall2025/601.771-HW1/question1
cp config_gpu.py config.py
python self_attention_profiler.py